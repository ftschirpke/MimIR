#include "mim/plug/gpu/phase/lower_map_reduce.h"

#include <algorithm>
#include <unordered_map>

#include <fe/log.h>
#include <fe/vector.h>

#include <mim/driver.h>
#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/btensor/btensor.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

namespace {

using fe::Vector;

/// Whether an explicit, user-written `%gpu.init` is reachable from `def`.
bool contains_gpu_init(const Def* def, DefSet& seen) {
    if (auto [_, ins] = seen.emplace(def); !ins) return false;
    if (Axm::isa<gpu::init>(def)) return true;
    for (auto d : def->deps())
        if (contains_gpu_init(d, seen)) return true;
    return false;
}

/// Bounds LowerMapReduce::classify_map_reduce_calls's forward use-walk; a chain longer than this is
/// left un-classified (conservatively kept host-visible).
constexpr int Max_Consumer_Hops = 128;

struct Consumers {
    DefSet defs;
    bool overflowed = false;
};

/// The real (non-`Extract`/`Tuple`-wrapper) consumers of `def`, found via `sched`'s direct-use info.
Consumers real_consumers(Scheduler& sched, const Def* def) {
    Consumers res;
    DefSet seen;
    int hops = 0;
    Vector<const Def*> stack{def};
    while (!stack.empty()) {
        auto d = stack.back();
        stack.pop_back();
        for (auto use : sched.uses(d)) {
            if (++hops > Max_Consumer_Hops) {
                res.overflowed = true;
                return res;
            }
            auto u = use.def();
            if (!seen.emplace(u).second) continue;
            if (u->isa<Extract>() || u->isa<Tuple>())
                stack.push_back(u);
            else
                res.defs.emplace(u);
        }
    }
    return res;
}

/// Whether `to` is reachable from `from` by repeatedly following other `map_reduce_post` calls' real
/// consumers — i.e. whether the two calls have a data dependency (in either direction is checked by the
/// caller). Conservative: an overflowed walk anywhere along the way counts as "reaches".
bool reaches(Scheduler& sched, const Def* from, const Def* to, DefSet& seen) {
    if (!seen.emplace(from).second) return false;
    auto consumers = real_consumers(sched, from);
    if (consumers.overflowed) return true;
    for (auto d : consumers.defs) {
        if (d == to) return true;
        if (Axm::isa<btensor::map_reduce_post>(d) && reaches(sched, d, to, seen)) return true;
    }
    return false;
}

/// `real_consumers` unwraps a *use*'s `Extract`/`Tuple` wrappers to find a real consumer; this unwraps the
/// opposite direction, peeling the `Extract` that pulled a `map_reduce_post` call's buffer result out of
/// its `[mem, %buffer.Buf ...]` pair back down to the call itself.
const Def* peel_extract(const Def* def) {
    while (auto ex = def->isa<Extract>())
        def = ex->tuple();
    return def;
}

/// Mirrors `btensor::phase::LowerMapReduce`'s helper of the same name: a counting `%affine.For` loop body
std::pair<Lam*, const Def*> counting_for(const Def* bound, const Def* acc, const Def* exit, Sym name) {
    auto& w       = bound->world();
    auto acc_ty   = acc->type();
    auto body     = w.mut_con({/* iter */ w.type_i64(), /* acc */ acc_ty, /* return */ w.cn(acc_ty)})->set(name);
    auto for_loop = w.call<affine::For>(body, exit, Defs{w.lit_i64(0), bound, w.lit_i64(1), acc});
    return {body, for_loop};
}

/// Mirrors `btensor::phase::LowerMapReduce`'s `affine_map` lambda
std::pair<const Def*, const Def*>
affine_map(const Def* f, const Def* m, const Def* n, const Def* sin, const Def* sout, const Def* idxs, const Def* mem) {
    auto& w = mem->world();
    auto a  = w.app(w.annex<affine::map>(), Defs{m, n});
    a       = w.app(a, Defs{sin, sout});
    a       = w.app(a, f);
    a       = w.app(a, idxs);
    a       = w.app(a, mem->type()->as<App>()->arg());
    return w.app(a, mem)->projs<2>();
}

const Def* fold_index(const Def* shape, const Def* idx) {
    auto& w = shape->world();
    auto r  = shape->num_projs();
    DefVec out;
    bool dropped = false;
    for (size_t i = 0; i != r; ++i)
        if (auto l = Lit::isa<nat_t>(shape->proj(r, i)); l && *l == 1)
            dropped = true;
        else
            out.push_back(idx->proj(r, i));
    // Without dropped axes the tuple below would just eta-reduce back to `idx` — but only after
    // World::tuple's pack normalization has alpha-compared the projections, which walks `idx`'s whole
    // (mem-threaded, Var-dependent) coordinate chain per elem pair — exponentially. Return `idx` directly.
    if (!dropped) return idx;
    return w.tuple(out);
}

/// Chained `%mem.lea` over a coordinate tuple.
const Def* op_lea_tuple(const Def* ptr, const Def* tuple) {
    auto n       = tuple->num_projs();
    auto element = ptr;
    for (size_t i = 0; i != n; ++i)
        element = mem::op_lea(element, tuple->proj(n, i));
    return element;
}

/// Scalarize may flatten an escaped `comb`/`post` from `Cn [[mem, T, ins], Cn ret]` to `Cn [mem, T, ins, Cn ret]`.
Lam* rebuild_lam_global_mem(Lam* lam, const Def* Tout, Sym name) {
    auto& w        = lam->world();
    auto global_ty = w.annex<gpu::GlobalM>();

    Lam* new_lam;
    if (lam->num_vars() == 2) {
        auto [_, Tin, extra_ty] = lam->var(0)->type()->projs<3>();
        new_lam = w.mut_con(Defs{w.sigma({global_ty, Tin, extra_ty}), w.cn({global_ty, Tout})})->set(name);
    } else {
        auto Tin      = lam->var(1)->type();
        auto extra_ty = lam->var(2)->type();
        new_lam       = w.mut_con(Defs{global_ty, Tin, extra_ty, w.cn({global_ty, Tout})})->set(name);
    }
    new_lam->set(true, lam->reduce_body(new_lam->var()));
    return new_lam;
}

/// Mirrors `btensor::phase::LowerMapReduce`'s helper of the same name.
void apply_cps(World& w, Lam* mut, const Def* f, DefVec parts, const Def* k) {
    auto dom = f->type()->as<Pi>()->dom();
    if (dom->num_projs() == parts.size() + 1) {
        parts.emplace_back(k);
        mut->app(true, f, parts);
    } else {
        mut->app(true, f, Defs{w.tuple(parts), k});
    }
}

Vector<nat_t> row_major_strides(const Vector<nat_t>& dims) {
    Vector<nat_t> strides(dims.size());
    nat_t acc = 1;
    for (auto i = dims.size(); i-- != 0;) {
        strides[i] = acc;
        acc *= dims[i];
    }
    return strides;
}

std::pair<const Def*, DefVec> unflatten_index(World& w, const Def* flat, const Vector<nat_t>& dims, const Def* mem) {
    auto strides = row_major_strides(dims);
    DefVec coords(dims.size());
    for (size_t d = 0; d != dims.size(); ++d) {
        auto [m1, q] = w.call(core::div::udiv, Defs{mem, w.tuple({flat, w.lit_i64(strides[d])})})->projs<2>();
        auto [m2, r] = w.call(core::div::urem, Defs{m1, w.tuple({q, w.lit_i64(dims[d])})})->projs<2>();
        mem          = m2;
        coords[d]    = w.call(core::conv::u, w.lit_nat(dims[d]), r);
    }
    return {mem, coords};
}

struct InputDesc {
    DefVec rs, ss, ts, accs;
};

InputDesc extract_input_desc(nat_t n, const Def* Rs, const Def* Ss, const Def* Ts, const Def* accs) {
    InputDesc desc{DefVec(n), DefVec(n), DefVec(n), DefVec(n)};
    for (nat_t i = 0; i != n; ++i) {
        desc.rs[i]   = Rs->proj(n, i);
        desc.ss[i]   = Ss->proj(n, i);
        desc.ts[i]   = Ts->proj(n, i);
        desc.accs[i] = accs->proj(n, i);
    }
    return desc;
}

struct Inputs {
    const Def* mem;
    const Def* global;
    DefVec dptrs;
};

/// `%gpu.GlobalPtr` is a `lam` macro for `%mem.Ptr (T, %gpu.addr_space_global)` and reduces away on
/// construction, so a constructed device pointer's type no longer carries a distinct `gpu::GlobalPtr` tag
/// — check the underlying `%mem.Ptr`'s address space instead.
bool is_device_ptr(World& w, const Def* ty) {
    auto ptr = Axm::isa<mem::Ptr>(ty);
    if (!ptr) return false;
    auto as = Lit::isa<nat_t>(ptr->arg(1));
    return as && *as == Lit::as(w.annex<gpu::addr_space_global>());
}

/// A producer classified `DeviceOnly` returns its output as a raw device pointer, not a `%buffer.Buf`.
/// The caller resolves each slot of `ins` accordingly: the plain rewrite of the consumer's own argument
/// tuple for an ordinary host input (still `%buffer.Buf`-typed, since `%buffer.lower_ptr` hasn't run yet),
/// or the producer's own already-rewritten result for a device-resident one — only the latter carries
/// its true (raw-pointer) type, since a wholesale rewrite of the argument tuple's declared type can't
/// special-case one slot's producer the way rewriting that producer's call itself does.
/// `stream`, if non-null, routes the upload through the asynchronous axiom variant on that stream.
Inputs alloc_copy_inputs(World& w,
                         const Def* m0,
                         const Def* m1,
                         Defs ris,
                         Defs sis,
                         Defs tis,
                         Defs ins,
                         const Def* stream) {
    DefVec dptrs(ris.size());
    for (size_t i = 0; i != ris.size(); ++i) {
        auto in = ins[i];
        if (is_device_ptr(w, in->type())) {
            dptrs[i] = in; // already device-resident: no host round-trip needed
            continue;
        }
        const Def* alloc_copy;
        if (stream)
            alloc_copy = w.app(w.app(w.annex<gpu::buf_alloc_copy>(gpu::buf_alloc_copy::asyn), {ris[i], sis[i], tis[i]}),
                               {m0, m1, in, stream});
        else
            alloc_copy
                = w.app(w.app(w.annex<gpu::buf_alloc_copy>(gpu::buf_alloc_copy::block), {ris[i], sis[i], tis[i]}),
                        {m0, m1, in});
        auto [m2, g2, ptr] = alloc_copy->projs<3>();
        m0                 = m2;
        m1                 = g2;
        dptrs[i]           = ptr;
    }
    return {m0, m1, dptrs};
}

const Def* output_arr_ty(World& w, const Def* elem_ty, const Def* So, nat_t ro) {
    auto arr_ty = elem_ty;
    for (auto d = ro; d-- != 0;)
        arr_ty = w.arr(So->proj(ro, d), arr_ty);
    return arr_ty;
}

std::pair<const Def*, const Def*> alloc_output(World& w, const Def* m1, const Def* arr_ty, const Def* stream) {
    if (stream) return w.app(w.app(w.annex<gpu::alloc>(gpu::alloc::asyn), arr_ty), w.tuple({m1, stream}))->projs<2>();
    return w.app(w.app(w.annex<gpu::alloc>(gpu::alloc::block), arr_ty), m1)->projs<2>();
}

/// Creates and initializes a fresh `%gpu.Stream`, threading `(mem, global)`.
std::tuple<const Def*, const Def*, const Def*> create_stream(World& w, const Def* mem, const Def* global) {
    auto [m1, ptr]    = mem::op_alloc(w.annex<gpu::Stream>(), mem)->projs<2>();
    auto [m2, g2]     = w.app(w.annex<gpu::stream_init>(), Defs{m1, global, ptr})->projs<2>();
    auto [m3, stream] = w.call<mem::load>(Defs{m2, ptr})->projs<2>();
    return {m3, g2, stream};
}

const Def* free_ptr(World& w, const Def* global, const Def* ptr, const Def* stream) {
    if (stream) return w.call(gpu::free::asyn, Defs{global, ptr, stream});
    return w.call(gpu::free::block, Defs{global, ptr});
}

struct Grid {
    nat_t n_groups, n_items, total;
};

Grid grid_layout(const Vector<nat_t>& out_dims) {
    nat_t total = 1;
    for (auto d : out_dims)
        total *= d;
    nat_t n_items  = std::min<nat_t>(total, 1024);
    nat_t n_groups = (total + n_items - 1) / n_items;
    return {n_groups, n_items, total};
}

/// Per-input state for `build_kernel`: `InputDesc`'s shapes/access-functions plus `alloc_copy_inputs`'s pointers.
struct Mapped {
    DefVec rs, ss, dptrs, accs;
    nat_t n() const { return dptrs.size(); }
};

/// Builds the kernel: one thread per output point, reducing sequentially over the `rr` reduction dims.
Lam* build_kernel(World& w,
                  const Def* Ro,
                  nat_t rr,
                  const Vector<nat_t>& out_dims,
                  const Def* Sr,
                  const Def* So,
                  const Mapped& ins,
                  const Def* To,
                  const Def* acc_out,
                  const Def* init,
                  Lam* global_comb,
                  const Mapped& post_ins,
                  Lam* global_post,
                  const Def* Tp,
                  const Def* out_dptr,
                  const Grid& grid) {
    auto nis        = ins.n();
    auto nps        = post_ins.n();
    auto ro         = out_dims.size();
    auto nloops_nat = ro + rr;
    auto n          = w.lit_nat(nloops_nat);

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("mapReduceKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);

    auto group_i64 = grid.n_groups == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), group_id);
    auto item_i64  = grid.n_items == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), item_id);
    auto flat
        = w.call(core::wrap::add, core::Mode::none,
                 Defs{w.call(core::wrap::mul, core::Mode::none, Defs{group_i64, w.lit_i64(grid.n_items)}), item_i64});

    auto in_range     = w.call(core::icmp::ul, Defs{flat, w.lit_i64(grid.total)});
    auto early_return = w.mut_con(w.sigma(Defs{}))->set("outOfRange");
    early_return->app(true, k_ret, Defs{k_global, k_shared, k_const, k_local});
    auto body = w.mut_con(w.sigma(Defs{}))->set("inRange");
    kernel->set(true, w.app(w.extract(w.tuple({early_return, body}), in_range), w.tuple()));

    auto write_back           = w.mut_con(Defs{global_ty, To})->set("writeBack");
    auto [wb_mem, acc_final]  = write_back->vars<2>();
    auto [wb_mem2, wb_coords] = unflatten_index(w, flat, out_dims, wb_mem);
    DefVec wb_idx             = wb_coords;
    for (size_t j = 0; j != rr; ++j)
        wb_idx.push_back(w.call(core::conv::u, Sr->proj(nloops_nat, ro + j), w.lit_i64(0)));
    auto [wc_mem, write_coords] = affine_map(acc_out, Ro, n, Sr, So, w.tuple(wb_idx), wb_mem2);

    auto pcur = wc_mem;
    DefVec post_elems(nps);
    for (size_t j = 0; j != nps; ++j) {
        auto [pc_mem, pcoords]
            = affine_map(post_ins.accs[j], post_ins.rs[j], Ro, So, post_ins.ss[j], write_coords, pcur);
        pcur = pc_mem;
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{pcur, op_lea_tuple(k_post_dptrs[j], fold_index(post_ins.ss[j], pcoords))})
                  ->projs<2>();
        pcur          = rd_mem;
        post_elems[j] = rd_val;
    }

    auto after_post            = w.mut_con(Defs{global_ty, Tp})->set("afterPost");
    auto [post_mem, elem_post] = after_post->vars<2>();
    auto final_mem
        = w.call<mem::store>(Defs{post_mem, op_lea_tuple(k_out_dptr, fold_index(So, write_coords)), elem_post});
    after_post->app(true, k_ret, Defs{final_mem, k_shared, k_const, k_local});
    apply_cps(w, write_back, global_post, {pcur, acc_final, w.tuple(post_elems)}, after_post);

    const Def* acc   = w.tuple({k_global, init});
    const Def* cont  = write_back;
    Lam* current_mut = body;
    DefVec red_iters;
    red_iters.reserve(rr);
    for (size_t j = 0; j != rr; ++j) {
        auto dim                    = Sr->proj(nloops_nat, ro + j);
        auto bound                  = w.call<core::bitcast>(w.type_i64(), dim);
        auto [rbody, for_call]      = counting_for(bound, acc, cont, w.sym("forRed_" + std::to_string(j)));
        auto [iter, new_acc, yield] = rbody->vars<3>();
        cont                        = yield;
        red_iters.push_back(w.call(core::conv::u, dim, iter));
        acc = new_acc;
        current_mut->set(true, for_call);
        current_mut = rbody;
    }
    auto [red_mem, elem_acc] = acc->projs<2>();

    auto [body_mem, body_coords] = unflatten_index(w, flat, out_dims, red_mem);
    DefVec iters_v               = body_coords;
    iters_v.insert(iters_v.end(), red_iters.begin(), red_iters.end());
    auto iters = w.tuple(iters_v);

    auto cur = body_mem;
    DefVec input_elems(nis);
    for (size_t i = 0; i != nis; ++i) {
        auto [mc_mem, coords] = affine_map(ins.accs[i], ins.rs[i], n, Sr, ins.ss[i], iters, cur);
        cur                   = mc_mem;
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{cur, op_lea_tuple(k_dptrs[i], fold_index(ins.ss[i], coords))})->projs<2>();
        cur            = rd_mem;
        input_elems[i] = rd_val;
    }

    apply_cps(w, current_mut, global_comb, {cur, elem_acc, w.tuple(input_elems)}, cont);

    return kernel;
}

/// `materialize` copies the output back to a host `%buffer.Buf` (today's behavior); when false, the raw
/// device pointer is handed to `cont` instead. `consumer_frees_output` is set for a `DeviceOnly` result
/// whose single real consumer takes over freeing it once it has issued the kernel that reads it; otherwise
/// (host-visible, or truly unused) this call frees it itself. `stream`, if non-null, is this call's own
/// dedicated stream: the copy-back and every free run asynchronously on it, followed by a `stream_sync`
/// before the result becomes host-observable (materialized) and a `stream_deinit` before `auto_deinit`.
Lam* build_teardown(World& w,
                    const Def* Ro,
                    const Def* So,
                    const Def* Tp,
                    Defs dptrs,
                    Defs post_dptrs,
                    const Def* out_dptr,
                    const Def* cont,
                    bool materialize,
                    bool consumer_frees_output,
                    const Def* stream) {
    auto global_ty = w.annex<gpu::GlobalM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto mem_ty    = w.call<mem::M>(0);

    auto after_launch                        = w.mut_con(Defs{mem_ty, global_ty, const_ty})->set("afterLaunch");
    auto [post_mem, post_global, post_const] = after_launch->vars<3>();

    auto cur_mem    = post_mem;
    auto cur_global = post_global;
    const Def* result;
    if (materialize) {
        auto [alloc_mem, host_buf] = buffer::op_alloc(Ro, So, Tp, cur_mem)->projs<2>();
        const Def* copy_back;
        if (stream)
            copy_back = w.app(w.app(w.annex<gpu::buf_copy_to_host>(gpu::buf_copy_to_host::asyn), {Ro, So, Tp}),
                              {alloc_mem, cur_global, out_dptr, host_buf, stream});
        else
            copy_back = w.app(w.app(w.annex<gpu::buf_copy_to_host>(gpu::buf_copy_to_host::block), {Ro, So, Tp}),
                              {alloc_mem, cur_global, out_dptr, host_buf});
        auto [cb_mem, cb_global] = copy_back->projs<2>();
        cur_mem                  = cb_mem;
        cur_global               = cb_global;
        result                   = host_buf;
    } else {
        result = out_dptr;
    }

    // A device-resident producer's pointer may appear more than once among `dptrs`/`post_dptrs` (e.g. the
    // same intermediate result read both as a mapped input and again in the epilogue): free each once.
    DefSet freed;
    auto free_once = [&](const Def* dptr) {
        if (freed.emplace(dptr).second) cur_global = free_ptr(w, cur_global, dptr, stream);
    };
    for (auto dptr : dptrs)
        free_once(dptr);
    for (auto dptr : post_dptrs)
        free_once(dptr);
    if (!consumer_frees_output) free_once(out_dptr);

    if (stream) {
        if (materialize) {
            // The async copy above doesn't block the host: sync before `result` becomes observable.
            auto [sm, sg] = w.app(w.annex<gpu::stream_sync>(), Defs{cur_mem, cur_global, stream})->projs<2>();
            cur_mem       = sm;
            cur_global    = sg;
        }
        auto [dm, dg] = w.app(w.annex<gpu::stream_deinit>(), Defs{cur_mem, cur_global, stream})->projs<2>();
        cur_mem       = dm;
        cur_global    = dg;
    }

    auto final_mem = w.app(w.annex<gpu::auto_deinit>(), Defs{cur_mem, cur_global, post_const});
    after_launch->app(true, cont, Defs{final_mem, result});
    return after_launch;
}

} // namespace

void LowerMapReduce::start() {
    DefSet seen;
    auto has_gpu_init
        = std::ranges::any_of(old_world().roots(), [&](auto def) { return contains_gpu_init(def, seen); });
    if (has_gpu_init) {
        log().w("not lowering any map-reduce operations to GPU: the program already contains an explicit `%gpu.init`");
        return;
    }
    classify_map_reduce_calls();
    Super::start();
}

/// Classifies every reachable `%btensor.map_reduce_post` call by how its result is consumed, so
/// `lower_map_reduce_post` can skip the host round-trip for a result that only feeds another such call.
void LowerMapReduce::classify_map_reduce_calls() {
    auto& ow = old_world();
    nest_    = std::make_unique<Nest>(ow);
    sched_   = std::make_unique<Scheduler>(*nest_);

    Vector<const App*> calls;
    DefSet seen;
    auto roots = ow.roots();
    Vector<const Def*> stack(roots.begin(), roots.end());
    while (!stack.empty()) {
        auto def = stack.back();
        stack.pop_back();
        if (auto [_, ins] = seen.emplace(def); !ins) continue;
        if (auto app = Axm::isa<btensor::map_reduce_post>(def)) calls.push_back(app);
        for (auto d : def->deps())
            stack.push_back(d);
    }

    for (auto call : calls) {
        auto consumers = real_consumers(*sched_, call);
        CallInfo info;
        if (consumers.overflowed) {
            info.cls = Classification::HostVisible;
        } else if (consumers.defs.empty()) {
            info.cls = Classification::Dead;
        } else if (consumers.defs.size() == 1) {
            if (auto mr = Axm::isa<btensor::map_reduce_post>(*consumers.defs.begin())) {
                info.cls             = Classification::DeviceOnly;
                info.single_consumer = mr;
            } else {
                info.cls = Classification::HostVisible;
            }
        } else {
            info.cls = Classification::HostVisible;
        }
        call_info_.emplace(call, info);
    }

    // Within one straight-line block (same `early` node, no loop/recursion), a second `HostVisible` call
    // provably independent of another one gets its own stream so the two launches can overlap.
    std::unordered_map<const Nest::Node*, Vector<const App*>> by_node;
    for (auto call : calls) {
        auto& info = call_info_.at(call);
        if (info.cls != Classification::HostVisible) continue;
        auto node = sched_->early(call);
        if (node->loop_depth() != 0 || node->is_recursive()) continue;
        by_node[node].push_back(call);
    }
    for (auto& [node, group] : by_node) {
        if (group.size() < 2) continue;
        auto a = group[0];
        for (size_t i = 1; i != group.size(); ++i) {
            auto b = group[i];
            DefSet seen_ab, seen_ba;
            if (!reaches(*sched_, a, b, seen_ab) && !reaches(*sched_, b, a, seen_ba)) {
                call_info_.at(b).own_stream = true;
                break;
            }
        }
    }
}

const LowerMapReduce::CallInfo& LowerMapReduce::call_info(const App* app) const {
    static const CallInfo default_info{};
    auto i = call_info_.find(app);
    return i != call_info_.end() ? i->second : default_info;
}

const Def* LowerMapReduce::rewrite_imm_App(const App* app) {
    if (Axm::isa<btensor::map_reduce_post>(app)) return lower_map_reduce_post(app);
    return Super::rewrite_imm_App(app);
}

const Def* LowerMapReduce::lower_map_reduce_post(const App* app) {
    if (is_bootstrapping()) return Super::rewrite_imm_App(app);

    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [nis_nps, meta, shapes, in_tys, comb_init, acc_out, accs_all] = c->uncurry_args<7>();
    auto [nis, nps]                     = nis_nps->projs<2>([](auto d) { return Lit::isa(d); });
    auto [To, Tp, Ro, Rn, sched_ty]     = meta->projs<5>();
    auto [So, Sr, sched]                = shapes->projs<3>();
    auto [Tis, Ris, Sis, Tps, Rps, Sps] = in_tys->projs<6>();
    auto [comb, init, post]             = comb_init->projs<3>();
    auto [accs, post_accs]              = accs_all->projs<2>();

    auto ro_l = Lit::isa<nat_t>(Ro);
    auto rn_l = Lit::isa<nat_t>(Rn);
    if (!nis || !nps || !ro_l || !rn_l || *rn_l < *ro_l) {
        log().w("{} doesn't have lowering-time known rank counts (nis/nps/Ro/Rn)", app);
        return Super::rewrite_imm_App(app);
    }
    auto nis_n = *nis;
    auto nps_n = *nps;
    auto ro    = *ro_l;
    auto rr    = *rn_l - *ro_l;

    Vector<nat_t> out_dims(ro);
    nat_t out_total = 1;
    for (nat_t d = 0; d != ro; ++d) {
        auto l = Lit::isa<nat_t>(Sr->proj(ro + rr, d));
        if (!l) {
            log().w("{} doesn't have a lowering-time known output (grid) shape", app);
            return Super::rewrite_imm_App(app);
        }
        out_dims[d] = *l;
        out_total *= *l;
    }
    if (out_total == 0) {
        log().w("{} has a zero-sized output, skipping GPU lowering", app);
        return Super::rewrite_imm_App(app);
    }

    auto comb_lam = comb->isa_mut<Lam>();
    auto post_lam = post->isa_mut<Lam>();
    if (!comb_lam || !post_lam) {
        log().w("{} doesn't have a lowering-time known combiner/epilogue", app);
        return Super::rewrite_imm_App(app);
    }

    auto& info                 = call_info(app);
    bool materialize           = info.cls == Classification::HostVisible;
    bool consumer_frees_output = info.cls == Classification::DeviceOnly;
    auto out_arr_ty            = output_arr_ty(w, Tp, So, ro);
    auto mem_ty                = w.call<mem::M>(0);
    // `%btensor.map_reduce_post` is mem-threaded (`[%mem.M 0, %buffer.Buf ...]`); a `DeviceOnly` result
    // keeps that same `[mem, value]` shape, just with a raw device pointer instead of a host buffer.
    auto result_ty = materialize ? rewrite(app->type()) : w.sigma({mem_ty, w.call<gpu::GlobalPtr>(out_arr_ty)});

    auto rewritten_arg                             = rewrite(app->arg());
    auto [_, rewritten_inputs, rewritten_post_ins] = rewritten_arg->projs<3>();
    auto fun = w.mut_fun(w.sigma({mem_ty, rewritten_inputs->type(), rewritten_post_ins->type()}), result_ty)
                   ->set("mapReduceAffGpu");
    auto call                                = w.app(cps::op_cps2ds_dep(fun), rewritten_arg);
    auto [fun_mem, new_inputs, new_post_ins] = fun->var(0_n)->projs<3>();
    auto cont                                = fun->var(1);

    auto [h_mem, h_global, h_const] = w.app(w.annex<gpu::auto_init>(), fun_mem)->projs<3>();

    const Def* stream = nullptr;
    if (info.own_stream) std::tie(h_mem, h_global, stream) = create_stream(w, h_mem, h_global);

    // A `DeviceOnly` producer's slot must be resolved by rewriting its own (old-world) call directly:
    // `new_inputs`/`new_post_ins` only carry `fun`'s declared (still `%buffer.Buf`-shaped) parameter type.
    auto [_old_mem_arg, old_ins, old_post_ins] = app->arg()->projs<3>();
    auto resolve_slot                          = [&](const Def* old_slot, const Def* new_slot) {
        auto mr = Axm::isa<btensor::map_reduce_post>(peel_extract(old_slot));
        return (mr && call_info(mr).cls == Classification::DeviceOnly) ? rewrite(old_slot) : new_slot;
    };
    DefVec in_vals(nis_n), post_in_vals(nps_n);
    for (nat_t i = 0; i != nis_n; ++i)
        in_vals[i] = resolve_slot(old_ins->proj(nis_n, i), new_inputs->proj(nis_n, i));
    for (nat_t j = 0; j != nps_n; ++j)
        post_in_vals[j] = resolve_slot(old_post_ins->proj(nps_n, j), new_post_ins->proj(nps_n, j));

    auto in_desc = extract_input_desc(nis_n, Ris, Sis, Tis, accs);
    auto inputs  = alloc_copy_inputs(w, h_mem, h_global, in_desc.rs, in_desc.ss, in_desc.ts, in_vals, stream);

    auto post_desc   = extract_input_desc(nps_n, Rps, Sps, Tps, post_accs);
    auto post_inputs = alloc_copy_inputs(w, inputs.mem, inputs.global, post_desc.rs, post_desc.ss, post_desc.ts,
                                         post_in_vals, stream);

    auto [out_global, out_dptr] = alloc_output(w, post_inputs.global, out_arr_ty, stream);

    auto global_comb = rebuild_lam_global_mem(comb_lam, To, w.sym("combGlobal"));
    auto global_post = rebuild_lam_global_mem(post_lam, Tp, w.sym("postGlobal"));

    auto grid = grid_layout(out_dims);

    auto kernel = build_kernel(w, Ro, rr, out_dims, Sr, So, Mapped{in_desc.rs, in_desc.ss, inputs.dptrs, in_desc.accs},
                               To, acc_out, init, global_comb,
                               Mapped{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.accs}, global_post, Tp,
                               out_dptr, grid);

    DefVec kernel_arg_tys(nis_n + nps_n + 1);
    for (nat_t i = 0; i != nis_n; ++i)
        kernel_arg_tys[i] = inputs.dptrs[i]->type();
    for (nat_t j = 0; j != nps_n; ++j)
        kernel_arg_tys[nis_n + j] = post_inputs.dptrs[j]->type();
    kernel_arg_tys[nis_n + nps_n] = out_dptr->type();

    auto launch = w.app(w.annex<gpu::launch>(), Defs{w.lit_nat(nis_n + nps_n + 1), w.tuple(kernel_arg_tys)});
    launch      = w.app(launch, Defs{w.lit_nat(grid.n_groups), w.lit_nat(grid.n_items),
                                stream ? stream : w.annex<gpu::default_stream>(), w.lit_ff(), w.tuple()});
    launch      = w.app(launch, kernel);

    DefVec kernel_args = inputs.dptrs;
    kernel_args.insert(kernel_args.end(), post_inputs.dptrs.begin(), post_inputs.dptrs.end());
    kernel_args.push_back(out_dptr);
    launch = w.app(launch, kernel_args);

    auto after_launch = build_teardown(w, Ro, So, Tp, inputs.dptrs, post_inputs.dptrs, out_dptr, cont, materialize,
                                       consumer_frees_output, stream);
    auto launch_call  = w.app(launch, Defs{w.tuple({post_inputs.mem, out_global, h_const}), after_launch});
    fun->set(true, launch_call);

    return call;
}

} // namespace mim::plug::gpu::phase
