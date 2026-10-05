#include "mim/plug/gpu/phase/lower_map_reduce.h"

#include <algorithm>

#include <fe/log.h>
#include <fe/vector.h>

#include <mim/check.h>
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

/// Counts, for every Def reachable from `def`, how many times it appears as an operand of some
/// other reachable Def -- lets `chainable_producer` tell whether a `%btensor.map_reduce_post`
/// result is consumed only by the one call chaining into it (safe to keep device-resident) or also
/// needed elsewhere (must go through the usual host round-trip).
void count_uses(const Def* def, DefMap<nat_t>& uses, DefSet& seen) {
    if (auto [_, ins] = seen.emplace(def); !ins) return;
    for (auto d : def->deps()) {
        ++uses[d];
        count_uses(d, uses, seen);
    }
}

/// If `d` is exactly the `%buffer.Buf` result of a `%btensor.map_reduce_post` application used
/// nowhere else, returns that application so its output can be chained device-resident into the
/// call reading `d` instead of round-tripping through host memory.
const App* chainable_producer(const Def* d, const DefMap<nat_t>& uses) {
    auto it = uses.find(d);
    if (it == uses.end() || it->second != 1) return nullptr;
    auto ex = d->isa<Extract>();
    if (!ex) return nullptr;
    auto app = ex->tuple()->isa<App>();
    if (!app || !Axm::isa<btensor::map_reduce_post>(app)) return nullptr;
    return app;
}

/// True iff `producer`'s buffer output is exactly one of `consumer`'s own `is`/`post_is` arguments
/// AND singly used program-wide -- i.e. `producer` is genuinely absorbed into `consumer`'s body by
/// `lower_map_reduce_chained`, not merely a mem-chain-adjacent predecessor that happens to feed some
/// unrelated node's data (e.g. a sibling branch of a `%%tensor.concat` sharing the same mem chain).
bool chains_into(const App* producer, const App* consumer, const DefMap<nat_t>& uses) {
    auto d = producer->proj(2, 1);
    if (chainable_producer(d, uses) != producer) return false;
    auto [mem, is_t, post_t] = consumer->arg()->projs<3>();
    for (auto e : is_t->projs())
        if (e == d) return true;
    for (auto e : post_t->projs())
        if (e == d) return true;
    return false;
}

/// Walks a call's `mem` input back through any `%btensor.map_reduce_post` producer that
/// `chains_into` confirms is genuinely absorbed into `consumer`'s own body, all the way to the
/// genuine predecessor outside the chain. Needed because that producer is getting absorbed into the
/// consumer's own body rather than materialized as a separate call: calling the generic `rewrite()`
/// on its `mem` output (as if it were an ordinary boundary value) would independently trigger its
/// *other*, host-round-tripping lowering too, duplicating the whole call. `consumer` is updated to
/// each newly found producer as the walk recurses, since deeper hops are absorbed into that
/// producer's own body, not into the original top-level call.
const Def* find_root_mem(const Def* mem_def, const App* consumer, const DefMap<nat_t>& uses) {
    auto ex = mem_def->isa<Extract>();
    if (!ex) return mem_def;
    auto app = ex->tuple()->isa<App>();
    if (!app || !Axm::isa<btensor::map_reduce_post>(app)) return mem_def;
    if (!chains_into(app, consumer, uses)) return mem_def;
    return find_root_mem(app->arg()->proj(3, 0), app, uses);
}

/// Whether an op's schedule type `TSched` is a `%gpu.Sched` (a mutable Pi, so compared up to α-equivalence).
bool is_gpu_sched(const Def* TSched) {
    return Checker::alpha<Checker::Mode::Test>(TSched, TSched->world().annex<gpu::Sched>());
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

std::pair<const Def*, const Def*> alloc_output(World& w, const Def* m1, const Def* elem_ty, const Def* So, nat_t ro) {
    auto arr_ty = elem_ty;
    for (auto d = ro; d-- != 0;)
        arr_ty = w.arr(So->proj(ro, d), arr_ty);
    return w.app(w.app(w.annex<gpu::alloc>(gpu::alloc::block), arr_ty), m1)->projs<2>();
}

/// The op-specific, decision-free pieces a `%gpu.mr_kernel` builds its kernels from (see gpu.mim).
/// The kernel receives the device pointers as its argument tuple: the `nis` inputs, the `nps`
/// epilogue inputs, and the output.
class Pieces {
public:
    Pieces(World& w,
           nat_t ro,
           nat_t rr,
           const Def* Ro,
           const Def* Sr,
           const Def* So,
           const InputDesc& ins,
           const InputDesc& post_ins,
           const Def* acc_out,
           Lam* comb,
           Lam* post,
           const Def* Tp)
        : w_(w)
        , ro_(ro)
        , rn_(ro + rr)
        , Ro_(Ro)
        , Sr_(Sr)
        , So_(So)
        , ins_(ins)
        , post_ins_(post_ins)
        , acc_out_(acc_out)
        , comb_(comb)
        , post_(post)
        , Tp_(Tp) {}

    /// `Fn [GlobalM, args, To, iv] → [GlobalM, To]`: fold all inputs read at the loop vector `iv`.
    Lam* cell(const Pi* ty) {
        auto lam                = w_.mut_lam(ty)->set("cell");
        auto [in, k]            = lam->vars<2>();
        auto [g, args, acc, iv] = in->projs<4>();
        auto [mem, elems]       = read_all(g, args, iv);
        apply_cps(w_, lam, comb_, {mem, acc, elems}, k);
        return lam;
    }

    /// `Fn [GlobalM, args, iv] → [GlobalM, Tis#i]`: read input `i` at the loop vector `iv`.
    Lam* rd(const Pi* ty, nat_t i) {
        auto lam           = w_.mut_lam(ty)->set("rd" + std::to_string(i));
        auto [in, k]       = lam->vars<2>();
        auto [g, args, iv] = in->projs<3>();
        auto [mem, val]    = read(g, args, i, iters(iv));
        lam->app(true, k, Defs{mem, val});
        return lam;
    }

    /// `Fn [GlobalM, To, [T0, T1]] → [GlobalM, To]`: the combiner itself.
    Lam* comb(const Pi* ty) {
        auto lam          = w_.mut_lam(ty)->set("comb");
        auto [in, k]      = lam->vars<2>();
        auto [g, acc, xs] = in->projs<3>();
        apply_cps(w_, lam, comb_, {g, acc, xs}, k);
        return lam;
    }

    /// `Fn [GlobalM, args, To, iv] → GlobalM`: run the epilogue on the folded value of the output
    /// cell at `iv` (whose reduction coordinates are ignored) and store the result.
    Lam* wb(const Pi* ty) {
        auto lam              = w_.mut_lam(ty)->set("wb");
        auto [in, k]          = lam->vars<2>();
        auto [g, args, v, iv] = in->projs<4>();
        auto n_args           = n_args_();

        DefVec wb_idx(rn_);
        for (nat_t d = 0; d != rn_; ++d)
            wb_idx[d] = w_.call(core::conv::u, Sr_->proj(rn_, d), d < ro_ ? iv->proj(rn_, d) : w_.lit_i64(0));
        auto [mem, coords] = affine_map(acc_out_, Ro_, w_.lit_nat(rn_), Sr_, So_, w_.tuple(wb_idx), g);

        auto nps = post_ins_.accs.size();
        DefVec post_elems(nps);
        for (size_t j = 0; j != nps; ++j) {
            auto [pc_mem, pcoords]
                = affine_map(post_ins_.accs[j], post_ins_.rs[j], Ro_, So_, post_ins_.ss[j], coords, mem);
            auto ptr = args->proj(n_args, ins_.accs.size() + j);
            std::tie(mem, post_elems[j])
                = w_.call<mem::load>(Defs{pc_mem, op_lea_tuple(ptr, fold_index(post_ins_.ss[j], pcoords))})->projs<2>();
        }

        auto after_post        = w_.mut_con(Defs{w_.annex<gpu::GlobalM>(), Tp_})->set("afterPost");
        auto [post_mem, value] = after_post->vars<2>();
        auto out_ptr           = op_lea_tuple(args->proj(n_args, n_args - 1), fold_index(So_, coords));
        after_post->app(true, k, w_.call<mem::store>(Defs{post_mem, out_ptr, value}));
        apply_cps(w_, lam, post_, {mem, v, w_.tuple(post_elems)}, after_post);
        return lam;
    }

private:
    nat_t n_args_() const { return ins_.accs.size() + post_ins_.accs.size() + 1; }

    const Def* iters(const Def* iv) {
        DefVec res(rn_);
        for (nat_t d = 0; d != rn_; ++d)
            res[d] = w_.call(core::conv::u, Sr_->proj(rn_, d), iv->proj(rn_, d));
        return w_.tuple(res);
    }

    std::pair<const Def*, const Def*> read(const Def* mem, const Def* args, nat_t i, const Def* iters) {
        auto [mc_mem, coords] = affine_map(ins_.accs[i], ins_.rs[i], w_.lit_nat(rn_), Sr_, ins_.ss[i], iters, mem);
        auto ptr              = op_lea_tuple(args->proj(n_args_(), i), fold_index(ins_.ss[i], coords));
        return w_.call<mem::load>(Defs{mc_mem, ptr})->projs<2>();
    }

    std::pair<const Def*, const Def*> read_all(const Def* mem, const Def* args, const Def* iv) {
        auto it = iters(iv);
        DefVec elems(ins_.accs.size());
        for (size_t i = 0; i != elems.size(); ++i)
            std::tie(mem, elems[i]) = read(mem, args, i, it);
        return {mem, w_.tuple(elems)};
    }

    World& w_;
    nat_t ro_, rn_;
    const Def *Ro_, *Sr_, *So_;
    const InputDesc &ins_, &post_ins_;
    const Def* acc_out_;
    Lam *comb_, *post_;
    const Def* Tp_;
};

/// Materializes `out_dptr` to a fresh host buffer, frees every device pointer this call (and
/// whatever it chained into) allocated, closes the session, and hands the result to `cont` (the
/// enclosing function's own return continuation).
const Def* build_teardown(World& w,
                          const Def* Ro,
                          const Def* So,
                          const Def* Tp,
                          const Def* mem,
                          const Def* global,
                          const Def* const_tok,
                          const DefVec& to_free,
                          const Def* out_dptr,
                          const Def* cont) {
    auto [alloc_mem, host_buf] = buffer::op_alloc(Ro, So, Tp, mem)->projs<2>();
    auto copy_back
        = w.app(w.app(w.annex<gpu::buf_copy_to_host>(), {Ro, So, Tp}), {alloc_mem, global, out_dptr, host_buf});
    auto [cb_mem, cb_global] = copy_back->projs<2>();

    auto cur_global = cb_global;
    for (auto dptr : to_free)
        cur_global = w.call(gpu::free::block, Defs{cur_global, dptr});
    cur_global = w.call(gpu::free::block, Defs{cur_global, out_dptr});

    auto final_mem = w.app(w.annex<gpu::auto_deinit>(), Defs{cb_mem, cur_global, const_tok});
    return w.app(cont, Defs{final_mem, host_buf});
}

} // namespace

void LowerMapReduce::start() {
    DefSet seen;
    if (std::ranges::any_of(old_world().roots(), [&](auto def) { return contains_gpu_init(def, seen); })) {
        log().w("not lowering any map-reduce operations to GPU: the program already contains an explicit `%gpu.init`");
        neutralize_only_ = true;
    } else {
        DefSet use_seen;
        for (auto def : old_world().roots())
            count_uses(def, use_count_, use_seen);
    }

    Super::start();
}

const Def* LowerMapReduce::rewrite_imm_App(const App* app) {
    if (!is_bootstrapping() && Axm::isa<btensor::map_reduce_post>(app))
        return neutralize_only_ ? neutralize(app) : lower_map_reduce_post(app);
    return Super::rewrite_imm_App(app);
}

/// The op `app` with a `%gpu.Sched` schedule replaced by the neutral `%btensor.mk_sched`, for
/// `%btensor.lower_map_reduce` to lower instead.
const Def* LowerMapReduce::neutralize(const App* app) {
    auto& w = new_world();
    DefVec args;
    for (const App* h = app; h; h = h->callee()->isa<App>())
        args.emplace_back(rewrite(h->arg()));
    std::ranges::reverse(args);

    auto [To, Tp, Ro, Rn, TSched] = args[1]->projs<5>();
    if (!is_gpu_sched(TSched)) return Super::rewrite_imm_App(app);

    auto [So, Sr, sched] = args[2]->projs<3>();
    auto neutral         = w.app(w.annex<btensor::mk_sched>(), {Rn, w.lit_nat_0()});
    args[1]              = w.tuple({To, Tp, Ro, Rn, neutral->type()});
    args[2]              = w.tuple({So, Sr, neutral});

    const Def* res = w.annex<btensor::map_reduce_post>();
    for (auto a : args)
        res = w.app(res, a);
    return res;
}

/// Destructures `app` (the same axiom-arg unpacking `lower_map_reduce_post` does) and builds
/// everything from input resolution through the kernel launch, sharing `mem`/`global`/`const_tok`
/// with whatever already-open session the caller is in -- either the top-level call's own fresh
/// `%%gpu.auto_init` session, or a consumer chaining this app's output in device-resident.
///
/// Each primary/post input is resolved by `resolve_is`: if the OLD (pre-rewrite) value at that
/// position is itself a `%btensor.map_reduce_post` application used nowhere else
/// (`chainable_producer`), this recurses into `lower_map_reduce_chained` for it instead of the
/// generic `rewrite()` path, so its result never leaves the device -- the recursive call's own
/// launch gets threaded in as "what happens before the rest of resolving *this* app's inputs" via
/// `cont`. A non-chainable input falls back to the ordinary alloc + host-to-device copy.
///
/// `resolved_is`/`resolved_post_is`, when given, override a non-chained slot's resolved value with
/// the caller's own already-rewritten Def (used by the top-level call so a genuinely host-resident
/// input flows through `fun`'s real formal parameter, matching `%%cps.cps2ds_dep`'s calling
/// convention, rather than a second, redundant free-variable reference to the same rewritten Def).
///
/// Returns `std::nullopt` if `app` itself doesn't have lowering-time known metadata (mirrors
/// `lower_map_reduce_post`'s own early bail-outs) -- only possible when called recursively for a
/// chained producer, in which case the caller falls back to treating that input as non-chained.
std::optional<const Def*> LowerMapReduce::lower_map_reduce_chained(const App* app,
                                                                   const Def* mem,
                                                                   const Def* global,
                                                                   const Def* const_tok,
                                                                   DefVec& to_free,
                                                                   const DeviceCont& cont,
                                                                   const DefVec* resolved_is,
                                                                   const DefVec* resolved_post_is) {
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
    if (!nis || !nps || !ro_l || !rn_l || *rn_l < *ro_l) return std::nullopt;
    auto nis_n = *nis;
    auto nps_n = *nps;
    auto ro    = *ro_l;
    auto rr    = *rn_l - *ro_l;

    nat_t out_total = 1;
    for (nat_t d = 0; d != ro; ++d) {
        auto l = Lit::isa<nat_t>(Sr->proj(ro + rr, d));
        if (!l) return std::nullopt;
        out_total *= *l;
    }
    if (out_total == 0) return std::nullopt;

    auto comb_lam = comb->isa_mut<Lam>();
    auto post_lam = post->isa_mut<Lam>();
    if (!comb_lam || !post_lam) return std::nullopt;

    auto in_desc   = extract_input_desc(nis_n, Ris, Sis, Tis, accs);
    auto post_desc = extract_input_desc(nps_n, Rps, Sps, Tps, post_accs);

    auto [old_mem, old_is, old_post_is] = app->arg()->projs<3>();

    DefVec input_dptrs(nis_n), post_dptrs(nps_n);

    auto alloc_one
        = [&](const Def* mem, const Def* global, const Def* r, const Def* s, const Def* t, const Def* host_val) {
              auto alloc_copy = w.app(w.app(w.annex<gpu::buf_alloc_copy>(), {r, s, t}), {mem, global, host_val});
              return alloc_copy->projs<3>();
          };

    // Resolves post_dptrs[j..nps_n), then builds and launches this app's own kernel, then hands the
    // result to `cont`. Defined before `resolve_in` since `resolve_in`'s base case starts it.
    auto resolve_post
        = [&](this auto&& self, nat_t j, const Def* mem, const Def* global, const Def* const_tok) -> const Def* {
        if (j != nps_n) {
            auto old_j = old_post_is->proj(nps_n, j);
            if (auto producer = chainable_producer(old_j, use_count_)) {
                auto sub = lower_map_reduce_chained(producer, mem, global, const_tok, to_free,
                                                    [&, j](const Def* m2, const Def* g2, const Def* c2, const Def* out,
                                                           const Def* p_ro, const Def* p_So) -> const Def* {
                                                        post_dptrs[j] = out;
                                                        // A schedule can pack a producer's output differently than a
                                                        // fresh host allocation of the same logical tensor would be
                                                        // shaped -- index by its *actual* shape, not this slot's
                                                        // declared (and now stale) `Sps`. Only when the rank agrees: a
                                                        // rank mismatch means `post_accs[j]` itself already reshapes
                                                        // between the producer's coordinate space and its own declared
                                                        // `Rps[j]`-dimensional one, and `Sps[j]` is what that reshaped
                                                        // output actually indexes into.
                                                        if (Lit::isa<nat_t>(p_ro) == Lit::isa<nat_t>(post_desc.rs[j]))
                                                            post_desc.ss[j] = p_So;
                                                        to_free.push_back(out);
                                                        return self(j + 1, m2, g2, c2);
                                                    });
                if (sub) return *sub;
                // `producer`'s own metadata wasn't lowering-time known after all -- fall back below.
            }
            auto host_val      = resolved_post_is ? (*resolved_post_is)[j] : rewrite(old_j);
            auto [m2, g2, ptr] = alloc_one(mem, global, post_desc.rs[j], post_desc.ss[j], post_desc.ts[j], host_val);
            post_dptrs[j]      = ptr;
            to_free.push_back(ptr);
            return self(j + 1, m2, g2, const_tok);
        }

        auto [out_global, out_dptr] = alloc_output(w, global, Tp, So, ro);
        global                      = out_global;

        DefVec kernel_args = input_dptrs;
        kernel_args.insert(kernel_args.end(), post_dptrs.begin(), post_dptrs.end());
        kernel_args.push_back(out_dptr);
        DefVec kernel_arg_tys;
        for (auto arg : kernel_args)
            kernel_arg_tys.push_back(arg->type());
        auto n  = w.lit_nat(kernel_args.size());
        auto Ts = w.tuple(kernel_arg_tys);

        // The op's schedule is a chooser among %gpu.mr_kernel's strategies, selected in the frontend;
        // an op scheduled for the CPU (or one no longer fitting the two-input strategies) gets the
        // generic kernel, via %gpu.cpu_sched for the former.
        auto pieces    = Pieces(w, ro, rr, Ro, Sr, So, in_desc, post_desc, acc_out,
                                rebuild_lam_global_mem(comb_lam, To, w.sym("combGlobal")),
                                rebuild_lam_global_mem(post_lam, Tp, w.sym("postGlobal")), Tp);
        auto gpu_sched = is_gpu_sched(sched_ty);
        const Def* handlers;
        if (gpu_sched && nis_n == 2) {
            auto mk  = w.app(w.annex<gpu::mr_kernel>(), {Rn, Ro, Sr, To, n, Ts, Tis->proj(2, 0), Tis->proj(2, 1)});
            auto dom = mk->type()->as<Pi>()->dom();
            auto ty  = [&](nat_t i) { return dom->proj(6, i)->as<Pi>(); };
            handlers = w.app(mk, {init, pieces.cell(ty(1)), pieces.rd(ty(2), 0), pieces.rd(ty(3), 1),
                                  pieces.comb(ty(4)), pieces.wb(ty(5))});
        } else {
            auto mk  = w.app(w.annex<gpu::mr_kernel_generic>(), {Rn, Ro, Sr, To, n, Ts});
            auto dom = mk->type()->as<Pi>()->dom();
            handlers = w.app(mk, {init, pieces.cell(dom->proj(3, 1)->as<Pi>()), pieces.wb(dom->proj(3, 2)->as<Pi>())});
        }
        auto L      = w.app(w.annex<gpu::Launch>(), {n, Ts});
        auto chosen = gpu_sched ? handlers : w.app(w.app(w.annex<gpu::cpu_sched>(), {Ro, Rn, L}), handlers->proj(3, 0));
        auto launch = w.app(w.app(w.app(sched, L), chosen), kernel_args);

        auto global_ty                     = w.annex<gpu::GlobalM>();
        auto const_ty                      = w.annex<gpu::ConstM>();
        auto mem_ty                        = w.call<mem::M>(0);
        auto after_launch                  = w.mut_con(Defs{mem_ty, global_ty, const_ty})->set("afterLaunch");
        auto [al_mem, al_global, al_const] = after_launch->vars<3>();
        after_launch->set(true, cont(al_mem, al_global, al_const, out_dptr, w.lit_nat(ro), So));

        return w.app(launch, Defs{w.tuple({mem, global, const_tok}), after_launch});
    };

    // Resolves input_dptrs[i..nis_n), continuing into resolve_post once done.
    auto resolve_in
        = [&](this auto&& self, nat_t i, const Def* mem, const Def* global, const Def* const_tok) -> const Def* {
        if (i == nis_n) return resolve_post(0, mem, global, const_tok);
        auto old_i = old_is->proj(nis_n, i);
        if (auto producer = chainable_producer(old_i, use_count_)) {
            auto sub = lower_map_reduce_chained(producer, mem, global, const_tok, to_free,
                                                [&, i](const Def* m2, const Def* g2, const Def* c2, const Def* out,
                                                       const Def* p_ro, const Def* p_So) -> const Def* {
                                                    input_dptrs[i] = out;
                                                    // A schedule can pack a producer's output differently than a fresh
                                                    // host allocation of the same logical tensor would be shaped --
                                                    // index by its *actual* shape, not this slot's declared (and now
                                                    // stale) `Sis`. Only when the rank agrees: a rank mismatch means
                                                    // `accs[i]` itself already reshapes between the producer's
                                                    // coordinate space and its own declared `Ris[i]`-dimensional one,
                                                    // and `Sis[i]` is what that reshaped output actually indexes into.
                                                    if (Lit::isa<nat_t>(p_ro) == Lit::isa<nat_t>(in_desc.rs[i]))
                                                        in_desc.ss[i] = p_So;
                                                    to_free.push_back(out);
                                                    return self(i + 1, m2, g2, c2);
                                                });
            if (sub) return *sub;
        }
        auto host_val      = resolved_is ? (*resolved_is)[i] : rewrite(old_i);
        auto [m2, g2, ptr] = alloc_one(mem, global, in_desc.rs[i], in_desc.ss[i], in_desc.ts[i], host_val);
        input_dptrs[i]     = ptr;
        to_free.push_back(ptr);
        return self(i + 1, m2, g2, const_tok);
    };

    return resolve_in(0, mem, global, const_tok);
}

const Def* LowerMapReduce::lower_map_reduce_post(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [nis_nps, meta, shapes, in_tys, comb_init, acc_out, accs_all] = c->uncurry_args<7>();
    auto [nis, nps]                 = nis_nps->projs<2>([](auto d) { return Lit::isa(d); });
    auto [To, Tp, Ro, Rn, sched_ty] = meta->projs<5>();
    auto [So, Sr, sched]            = shapes->projs<3>();
    auto [comb, init, post]         = comb_init->projs<3>();
    auto result_ty                  = rewrite(app->type());

    auto ro_l = Lit::isa<nat_t>(Ro);
    auto rn_l = Lit::isa<nat_t>(Rn);
    if (!nis || !nps || !ro_l || !rn_l || *rn_l < *ro_l) {
        log().w("{} doesn't have lowering-time known rank counts (nis/nps/Ro/Rn)", app);
        return neutralize(app);
    }
    auto nis_n = *nis;
    auto nps_n = *nps;
    auto ro    = *ro_l;
    auto rr    = *rn_l - *ro_l;

    nat_t out_total = 1;
    for (nat_t d = 0; d != ro; ++d) {
        auto l = Lit::isa<nat_t>(Sr->proj(ro + rr, d));
        if (!l) {
            log().w("{} doesn't have a lowering-time known output (grid) shape", app);
            return neutralize(app);
        }
        out_total *= *l;
    }
    if (out_total == 0) {
        log().w("{} has a zero-sized output, skipping GPU lowering", app);
        return neutralize(app);
    }

    auto comb_lam = comb->isa_mut<Lam>();
    auto post_lam = post->isa_mut<Lam>();
    if (!comb_lam || !post_lam) {
        log().w("{} doesn't have a lowering-time known combiner/epilogue", app);
        return neutralize(app);
    }

    // Only the primary/post inputs that AREN'T themselves an in-chain producer become `fun`'s
    // actual formal parameters -- a chained input's whole subgraph (further map_reduce_post calls)
    // gets built directly into `fun`'s body by `lower_map_reduce_chained`, its device output
    // flowing straight into this call's kernel without ever crossing back through host memory.
    auto [old_mem, old_is, old_post_is] = app->arg()->projs<3>();

    Vector<const App*> chain_in(nis_n), chain_post(nps_n);
    for (nat_t i = 0; i != nis_n; ++i)
        chain_in[i] = chainable_producer(old_is->proj(nis_n, i), use_count_);
    for (nat_t j = 0; j != nps_n; ++j)
        chain_post[j] = chainable_producer(old_post_is->proj(nps_n, j), use_count_);

    DefVec host_is, host_post_is;
    for (nat_t i = 0; i != nis_n; ++i)
        if (!chain_in[i]) host_is.push_back(rewrite(old_is->proj(nis_n, i)));
    for (nat_t j = 0; j != nps_n; ++j)
        if (!chain_post[j]) host_post_is.push_back(rewrite(old_post_is->proj(nps_n, j)));

    auto mem_ty             = w.call<mem::M>(0);
    auto rewritten_mem      = rewrite(find_root_mem(old_mem, app, use_count_));
    auto rewritten_inputs   = w.tuple(host_is);
    auto rewritten_post_ins = w.tuple(host_post_is);
    auto rewritten_arg      = w.tuple({rewritten_mem, rewritten_inputs, rewritten_post_ins});

    auto fun  = w.mut_fun(w.sigma({mem_ty, rewritten_inputs->type(), rewritten_post_ins->type()}), result_ty)
                    ->set("mapReduceAffGpu");
    auto call = w.app(cps::op_cps2ds_dep(fun), rewritten_arg);
    auto [fun_mem, new_host_inputs, new_host_post_ins] = fun->var(0_n)->projs<3>();
    auto cont                                          = fun->var(1);

    DefVec resolved_is(nis_n, nullptr), resolved_post_is(nps_n, nullptr);
    for (nat_t i = 0, k = 0; i != nis_n; ++i)
        if (!chain_in[i]) resolved_is[i] = new_host_inputs->proj(host_is.size(), k++);
    for (nat_t j = 0, k = 0; j != nps_n; ++j)
        if (!chain_post[j]) resolved_post_is[j] = new_host_post_ins->proj(host_post_is.size(), k++);

    auto [h_mem, h_global, h_const] = w.app(w.annex<gpu::auto_init>(), fun_mem)->projs<3>();

    DefVec to_free;
    auto top_cont = [&](const Def* mem, const Def* global, const Def* const_tok, const Def* out_dptr, const Def*,
                        const Def*) -> const Def* {
        return build_teardown(w, Ro, So, Tp, mem, global, const_tok, to_free, out_dptr, cont);
    };

    auto body
        = lower_map_reduce_chained(app, h_mem, h_global, h_const, to_free, top_cont, &resolved_is, &resolved_post_is);
    if (!body) {
        // Can't happen: the checks above are exactly the ones `lower_map_reduce_chained` itself
        // performs on this very `app`, so it never returns `std::nullopt` for the top-level call.
        log().w("{} unexpectedly failed GPU lowering", app);
        return neutralize(app);
    }
    fun->set(true, *body);

    return call;
}

} // namespace mim::plug::gpu::phase
