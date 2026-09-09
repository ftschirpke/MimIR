#include "mim/plug/gpu/phase/lower_map_reduce.h"

#include <algorithm>
#include <numeric>

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

/// Recovers the frontend's `(vdim, unroll)` schedule choice by applying `sched` to `%gpu.sched_probe`
/// instead of binding it to a nest, so the decision `%tensor.dot_product_impl` already made can be
/// read rather than rediscovered from the op's operands.
std::optional<std::pair<nat_t, nat_t>> probe_schedule(const Def* sched) {
    auto& w      = sched->world();
    auto probe   = w.annex<gpu::sched_probe>();
    auto pair_ty = probe->type()->as<Pi>()->codom();
    auto pair    = w.app(w.app(sched, pair_ty), probe);
    auto [vdim, unroll] = pair->projs<2>();
    auto vdim_l   = Lit::isa<nat_t>(vdim);
    auto unroll_l = Lit::isa<nat_t>(unroll);
    if (!vdim_l || !unroll_l) return std::nullopt;
    return std::pair{*vdim_l, *unroll_l};
}

/// Whether `acc`'s read coordinates depend on loop position `pos`: evaluate it on two loop-vectors
/// that differ only at `pos` (via distinct `%affine.lit` markers -- the same technique
/// `%tensor.fastest_axis`'s normalizer uses to reflect an access map's structure without assuming
/// which op produced it) and check whether the result changes. General enough to recognize a
/// strip-mined axis, whose read is an affine combination of two positions (`o#d · ts + o#(d+1)`, per
/// `tensor.mim`'s `split_iota`), not just a pure single-axis permutation.
bool depends_on_axis(const Def* acc, nat_t rn, nat_t pos) {
    auto& w = acc->world();
    DefVec base(rn);
    for (nat_t i = 0; i != rn; ++i)
        base[i] = w.call<affine::lit>(w.lit_nat(i + 1));
    auto varied = base;
    varied[pos] = w.call<affine::lit>(w.lit_nat(rn + 1));
    return w.app(acc, w.tuple(base)) != w.app(acc, w.tuple(varied));
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

/// Row-major 2-way split of a flat I64 index into `(outer, inner)` via a single div/mod by `inner_extent`.
std::tuple<const Def*, const Def*, const Def*> divmod_i64(World& w, const Def* mem, const Def* flat, nat_t inner_extent) {
    auto [m1, q] = w.call(core::div::udiv, Defs{mem, w.tuple({flat, w.lit_i64(inner_extent)})})->projs<2>();
    auto [m2, r] = w.call(core::div::urem, Defs{m1, w.tuple({flat, w.lit_i64(inner_extent)})})->projs<2>();
    return {m2, q, r};
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

Inputs alloc_copy_inputs(World& w, const Def* m0, const Def* m1, Defs ris, Defs sis, Defs tis, const Def* inputs) {
    DefVec dptrs(ris.size());
    for (size_t i = 0; i != ris.size(); ++i) {
        auto alloc_copy    = w.app(w.app(w.annex<gpu::buf_alloc_copy>(), {ris[i], sis[i], tis[i]}),
                                   {m0, m1, inputs->proj(ris.size(), i)});
        auto [m2, g2, ptr] = alloc_copy->projs<3>();
        m0                 = m2;
        m1                 = g2;
        dptrs[i]           = ptr;
    }
    return {m0, m1, dptrs};
}

std::pair<const Def*, const Def*> alloc_output(World& w, const Def* m1, const Def* elem_ty, const Def* So, nat_t ro) {
    auto arr_ty = elem_ty;
    for (auto d = ro; d-- != 0;)
        arr_ty = w.arr(So->proj(ro, d), arr_ty);
    return w.app(w.app(w.annex<gpu::alloc>(gpu::alloc::block), arr_ty), m1)->projs<2>();
}

/// `dptr`'s pointee is the *whole* rank-`rank_def` array (`GlobalPtr «Sis[i]; Tis[i]»`, per
/// `%gpu.buf_alloc_copy`'s result type) -- peel that many array levels to reach the scalar type a
/// shared-memory tile cell actually holds.
/// `World::seq` folds an extent-1 level away entirely (`«1; T»` is built as plain `T`), so a
/// dimension of `shape_def` that is literally `1` contributes no `Seq` to peel.
const Def* scalar_elem_ty(const Def* dptr, const Def* rank_def, const Def* shape_def) {
    auto ty = Axm::as<mem::Ptr>(dptr->type())->arg(0);
    auto r  = *Lit::isa<nat_t>(rank_def);
    for (nat_t i = 0; i != r; ++i)
        if (*Lit::isa<nat_t>(shape_def->proj(r, i)) != 1) ty = ty->as<Seq>()->body();
    return ty;
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

/// The recognized shape of a 2-input contraction where each input's own (non-reduction) axis is one
/// or more loop positions and *every* reduction position is shared by both inputs: GEMM/BMM (of which
/// `%tensor.dot_product`/`product_2d`/`bmm` are the frontend's only current producers) with `rr == 1`,
/// but also convolution's shape with `rr >= 2` (`cin, kh, kw` all shared by the data and weight
/// inputs) -- detected structurally, not by axiom identity or reduction-dim count, so any op with
/// this shape is picked up and tiled the same way, treating the whole reduction group as one
/// flattened GEMM "K" (mirroring how `m_positions`/`n_positions` already flatten "M"/"N").
/// `m_positions`/`n_positions` are ascending, and their sizes need not be 1: a schedule like
/// `dot_schedule_kvec` strip-mines the two logical output dims into (block, within-block) pairs
/// *before* GPU ever sees the op, so a real redvec call typically shows up with sizes of 2, not 1.
struct GemmShape {
    Vector<nat_t> m_positions; ///< ascending loop positions (within `[0, ro)`) forming input0's own axis
    Vector<nat_t> n_positions; ///< ditto for input1; together with `m_positions` partitions `[0, ro)`
};

std::optional<GemmShape> detect_gemm(const Mapped& ins, nat_t ro, nat_t rr, nat_t rn) {
    if (ins.n() != 2 || rr < 1 || ro < 2) return std::nullopt;
    for (nat_t i = 0; i != 2; ++i)
        for (nat_t p = ro; p != ro + rr; ++p)
            if (!depends_on_axis(ins.accs[i], rn, p)) return std::nullopt;
    auto positions_of = [&](const Def* acc) -> std::optional<Vector<nat_t>> {
        Vector<nat_t> pos;
        for (nat_t p = 0; p != ro; ++p)
            if (depends_on_axis(acc, rn, p)) pos.push_back(p);
        if (pos.empty()) return std::nullopt;
        return pos;
    };
    auto m_pos = positions_of(ins.accs[0]);
    auto n_pos = positions_of(ins.accs[1]);
    if (!m_pos || !n_pos) return std::nullopt;

    Vector<bool> covered(ro, false);
    for (auto p : *m_pos) {
        if (covered[p]) return std::nullopt;
        covered[p] = true;
    }
    for (auto p : *n_pos) {
        if (covered[p]) return std::nullopt;
        covered[p] = true;
    }
    for (auto c : covered)
        if (!c) return std::nullopt;

    return GemmShape{*m_pos, *n_pos};
}

/// The recognized shape of a 2-input, multi-reduction-dim contraction where one input's own
/// (non-reduction) axis is a single position (`weight_position` -- e.g. `cout`) and the other's is
/// one or more positions (`data_positions` -- e.g. `n, oh, ow`), detected the same way `GemmShape` is
/// (structurally, via `depends_on_axis`) but generalized to `rr >= 2` shared reduction positions (e.g.
/// `cin, kh, kw`) instead of exactly one. This is convolution's shape (and would match any future op
/// with the same structure), distinguished from a GEMM/BMM by having more than one reduction position
/// and an asymmetric (1 vs many) split of the parallel positions -- `conv_schedule` never picks a
/// reduction-range `vdim` the way `dot_schedule`/`dot_schedule_kvec` do, so unlike `GemmShape` this
/// isn't gated on `probe_schedule`'s redvec signal at all, only on this structural check.
struct ConvShape {
    nat_t weight_idx, data_idx;      ///< which of the two inputs is which
    nat_t weight_position;           ///< the single loop position (within `[0, ro)`) the weight reads
    Vector<nat_t> data_positions;    ///< ascending loop positions the data input reads; the *last* one
                                      ///< is treated as the fastest-varying (coalesced) axis, since a
                                      ///< schedule only ever strip-mines a *lower*-numbered position,
                                      ///< inserting new ones without reordering the rest (see `tensor.mim`'s
                                      ///< `strip_mine_par`/`split_iota`), so the highest original position
                                      ///< (`ow`) always sorts last regardless of how much splitting happened
};

std::optional<ConvShape> detect_conv(const Mapped& ins, nat_t ro, nat_t rr, nat_t rn) {
    if (ins.n() != 2 || rr < 2 || ro < 2) return std::nullopt;
    for (nat_t i = 0; i != 2; ++i)
        for (nat_t p = ro; p != ro + rr; ++p)
            if (!depends_on_axis(ins.accs[i], rn, p)) return std::nullopt;

    auto parallel_positions_of = [&](const Def* acc) {
        Vector<nat_t> pos;
        for (nat_t p = 0; p != ro; ++p)
            if (depends_on_axis(acc, rn, p)) pos.push_back(p);
        return pos;
    };
    auto pos0 = parallel_positions_of(ins.accs[0]);
    auto pos1 = parallel_positions_of(ins.accs[1]);
    if (pos0.empty() || pos1.empty()) return std::nullopt;

    Vector<bool> covered(ro, false);
    for (auto p : pos0) {
        if (covered[p]) return std::nullopt;
        covered[p] = true;
    }
    for (auto p : pos1) {
        if (covered[p]) return std::nullopt;
        covered[p] = true;
    }
    for (auto c : covered)
        if (!c) return std::nullopt;

    if (pos0.size() == 1 && pos1.size() != 1) return ConvShape{0, 1, pos0[0], pos1};
    if (pos1.size() == 1 && pos0.size() != 1) return ConvShape{1, 0, pos1[0], pos0};
    return std::nullopt; // ambiguous (both size 1, i.e. rr>=2 GEMM-shaped -- doesn't happen for conv/pool)
}

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

/// Thread-block edge length for `build_kernel_gemm_tiled`'s shared-memory tiles.
constexpr nat_t Gemm_Tile = 16;

/// A shared-memory-tiled kernel for a detected 2-input contraction (`GemmShape`): each
/// `Gemm_Tile`x`Gemm_Tile` thread block covers a tile of the two parallel dims and streams the whole
/// reduction group -- flattened into one GEMM "K", exactly like `m_extents`/`n_extents` flatten "M"/
/// "N" -- through cooperative loads into shared memory `Gemm_Tile` elements at a time, instead of
/// every thread re-reading the same rows/columns from global memory independently. This is what
/// makes convolution (`rr` = `cin, kh, kw`) tile the same way GEMM/BMM (`rr` = 1) does: the K-tile
/// loop doesn't care how many reduction positions it flattens, only that every one of them is shared
/// by both inputs (`detect_gemm` already checked that). Requires both parallel dims and the flattened
/// reduction extent to be exact multiples of `Gemm_Tile` -- the caller falls back to `build_kernel`
/// otherwise. Reuses the same write-back/epilogue shape as `build_kernel`, just fed by the tiled
/// reduction instead of the sequential one, so an existing epilogue fusion (e.g. bias + activation)
/// keeps working unchanged.
std::pair<Lam*, const Def*> build_kernel_gemm_tiled(World& w,
                             const Def* Ro,
                             nat_t ro,
                             nat_t rr,
                             const Def* Sr,
                             const Def* So,
                             const Mapped& ins,
                             const GemmShape& shape,
                             const Vector<nat_t>& m_extents,
                             const Vector<nat_t>& n_extents,
                             const Def* To,
                             const Def* acc_out,
                             const Def* init,
                             Lam* global_comb,
                             const Mapped& post_ins,
                             Lam* global_post,
                             const Def* Tp,
                             const Def* out_dptr) {
    auto nis  = ins.n();
    auto nps  = post_ins.n();
    auto rn   = ro + rr;
    auto n    = w.lit_nat(rn);
    auto m_total = std::accumulate(m_extents.begin(), m_extents.end(), nat_t{1}, std::multiplies<>{});
    auto n_total = std::accumulate(n_extents.begin(), n_extents.end(), nat_t{1}, std::multiplies<>{});
    Vector<nat_t> k_extents(rr);
    for (nat_t j = 0; j != rr; ++j)
        k_extents[j] = *Lit::isa<nat_t>(Sr->proj(rn, ro + j));
    auto dimk = std::accumulate(k_extents.begin(), k_extents.end(), nat_t{1}, std::multiplies<>{});

    auto tile      = Gemm_Tile;
    auto n_blocks0 = m_total / tile;
    auto n_blocks1 = n_total / tile;
    auto n_ktiles  = dimk / tile;
    Grid grid{n_blocks0 * n_blocks1, tile * tile, m_total * n_total};

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();

    auto elem_ty0       = scalar_elem_ty(ins.dptrs[0], ins.rs[0], ins.ss[0]);
    auto elem_ty1       = scalar_elem_ty(ins.dptrs[1], ins.rs[1], ins.ss[1]);
    // Padding the row stride to `tile + 1` staggers same-column accesses across banks: every thread
    // in a warp reading column `kk_idx` of the same tile row would otherwise land in the same bank.
    auto tile_ty = [&](const Def* elem_ty) { return w.arr(w.lit_nat(tile), w.arr(w.lit_nat(tile + 1), elem_ty)); };
    auto shared_pack_ty = w.sigma({tile_ty(elem_ty0), tile_ty(elem_ty1)});
    auto shared_ptr_ty  = w.call<gpu::SharedPtr>(shared_pack_ty);

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma({shared_ptr_ty}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("gemmTiledKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);

    auto shared_pack_ptr = k_shared_ptrs->proj(1, 0);
    auto shared0_ptr      = mem::op_lea_unsafe(shared_pack_ptr, u64{0});
    auto shared1_ptr      = mem::op_lea_unsafe(shared_pack_ptr, u64{1});

    // --- Decompose (group_id, item_id) into (block0, block1, ty, tx); input0 always owns the `ty`
    // (row) role and input1 the `tx` (col) role -- fixed by construction, not derived from the op. ---
    auto entry = w.mut_con(w.sigma(Defs{}))->set("gemmEntry");
    kernel->set(true, w.app(entry, w.tuple()));

    auto group_i64 = grid.n_groups == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), group_id);
    auto item_i64  = grid.n_items == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), item_id);

    auto [m_bt, block0_i64, block1_i64] = divmod_i64(w, k_global, group_i64, n_blocks1);
    auto [m_tt, ty_i64, tx_i64]         = divmod_i64(w, m_bt, item_i64, tile);

    auto par_local0_idx = w.call(core::conv::u, w.lit_nat(tile), ty_i64);
    auto par_local1_idx = w.call(core::conv::u, w.lit_nat(tile), tx_i64);
    auto k_local0_idx    = w.call(core::conv::u, w.lit_nat(tile + 1), tx_i64);
    auto k_local1_idx    = w.call(core::conv::u, w.lit_nat(tile + 1), ty_i64);
    auto k_local0_i64    = tx_i64;
    auto k_local1_i64    = ty_i64;

    auto mul_i64 = [&](const Def* a, const Def* b) { return w.call(core::wrap::mul, core::Mode::none, Defs{a, b}); };
    auto add_i64 = [&](const Def* a, const Def* b) { return w.call(core::wrap::add, core::Mode::none, Defs{a, b}); };

    // `m_flat`/`n_flat` are this thread's own (global, tile-relative) logical M/N coordinate; each was
    // split into `m_extents`/`n_extents` many loop positions by the schedule's own strip-mining, so
    // decompose them the same way (most-significant position first) rather than assuming just one.
    auto m_flat_global = add_i64(mul_i64(block0_i64, w.lit_i64(tile)), ty_i64);
    auto n_flat_global = add_i64(mul_i64(block1_i64, w.lit_i64(tile)), tx_i64);
    auto [m_after, m_sub_coords] = unflatten_index(w, m_flat_global, m_extents, m_tt);
    auto [n_after, n_sub_coords] = unflatten_index(w, n_flat_global, n_extents, m_after);

    /// The full `rn`-length loop-vector for reading one input at the given global indices: `own_pos`'s
    /// positions get `own_coords` (already `Idx`-typed, from `unflatten_index`), `[ro, ro+rr)` get
    /// `k_vals` (ditto, one per reduction position), and any remaining position (belonging to the
    /// *other* input's own axis) is unused by this input's access map (checked by `detect_gemm`), so
    /// any in-bounds value is fine.
    auto build_iters = [&](const Vector<nat_t>& own_pos, const DefVec& own_coords, const DefVec& k_vals) {
        DefVec iv(rn);
        Vector<bool> filled(rn, false);
        for (size_t j = 0; j != own_pos.size(); ++j) {
            iv[own_pos[j]]    = own_coords[j];
            filled[own_pos[j]] = true;
        }
        for (nat_t j = 0; j != rr; ++j) {
            iv[ro + j]     = k_vals[j];
            filled[ro + j] = true;
        }
        for (nat_t p = 0; p != ro; ++p)
            if (!filled[p]) iv[p] = w.call(core::conv::u, Sr->proj(rn, p), w.lit_i64(0));
        return w.tuple(iv);
    };

    // --- Write-back (reused verbatim in shape from `build_kernel`, just carrying `shared` too). ---
    auto write_back           = w.mut_con(Defs{global_ty, shared_ty, To})->set("gemmWriteBack");
    auto [wb_mem, wb_shared, acc_final] = write_back->vars<3>();
    DefVec wb_idx(rn);
    for (size_t j = 0; j != shape.m_positions.size(); ++j)
        wb_idx[shape.m_positions[j]] = m_sub_coords[j];
    for (size_t j = 0; j != shape.n_positions.size(); ++j)
        wb_idx[shape.n_positions[j]] = n_sub_coords[j];
    for (nat_t j = 0; j != rr; ++j)
        wb_idx[ro + j] = w.call(core::conv::u, Sr->proj(rn, ro + j), w.lit_i64(0));
    auto [wc_mem, write_coords] = affine_map(acc_out, Ro, n, Sr, So, w.tuple(wb_idx), wb_mem);

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

    auto after_post                        = w.mut_con(Defs{global_ty, Tp})->set("gemmAfterPost");
    auto [post_mem, elem_post]             = after_post->vars<2>();
    auto final_mem
        = w.call<mem::store>(Defs{post_mem, op_lea_tuple(k_out_dptr, fold_index(So, write_coords)), elem_post});
    after_post->app(true, k_ret, Defs{final_mem, wb_shared, k_const, k_local});
    apply_cps(w, write_back, global_post, {pcur, acc_final, w.tuple(post_elems)}, after_post);

    // --- after_inner: barrier once the tile's reduction is folded, then continue the k-tile loop. ---
    auto after_inner = w.mut_con(Defs{global_ty, To})->set("gemmAfterInner");
    auto [ai_mem, ai_acc] = after_inner->vars<2>();

    // --- Outer k-tile loop: cooperative load, barrier, tiled reduction, barrier. ---
    const Def* outer_init             = w.tuple({n_after, k_shared, init});
    auto [outer_body, outer_for_call] = counting_for(w.lit_i64(n_ktiles), outer_init, write_back, w.sym("gemmKTile"));
    entry->set(true, outer_for_call);
    auto [kt_iter, outer_acc, outer_yield] = outer_body->vars<3>();
    auto [g0, s0, accval0]                 = outer_acc->projs<3>();

    auto k_tile_base = mul_i64(kt_iter, w.lit_i64(tile));
    auto k0_global   = add_i64(k_tile_base, k_local0_i64);
    auto k1_global   = add_i64(k_tile_base, k_local1_i64);

    // The global flat K position decomposes into the reduction group's own sub-positions (e.g.
    // `cin, kh, kw`) the same way `m_flat_global`/`n_flat_global` decompose above -- the shared-memory
    // tile itself stays a plain flat `(par_local, k_local)` array regardless of how many positions K
    // flattens, so only this global-memory read side needs the decomposition.
    auto [g0k, k0_vals] = unflatten_index(w, k0_global, k_extents, g0);
    auto [g1, coords0]  = affine_map(ins.accs[0], ins.rs[0], n, Sr, ins.ss[0],
                                     build_iters(shape.m_positions, m_sub_coords, k0_vals), g0k);
    auto [rd0, val0] = w.call<mem::load>(Defs{g1, op_lea_tuple(k_dptrs[0], fold_index(ins.ss[0], coords0))})->projs<2>();
    auto [rd0k, k1_vals] = unflatten_index(w, k1_global, k_extents, rd0);
    auto [g2, coords1]   = affine_map(ins.accs[1], ins.rs[1], n, Sr, ins.ss[1],
                                     build_iters(shape.n_positions, n_sub_coords, k1_vals), rd0k);
    auto [rd1, val1] = w.call<mem::load>(Defs{g2, op_lea_tuple(k_dptrs[1], fold_index(ins.ss[1], coords1))})->projs<2>();

    auto s1 = w.call<mem::store>(Defs{s0, op_lea_tuple(shared0_ptr, w.tuple({par_local0_idx, k_local0_idx})), val0});
    auto s2 = w.call<mem::store>(Defs{s1, op_lea_tuple(shared1_ptr, w.tuple({par_local1_idx, k_local1_idx})), val1});

    auto [g3, s3] = w.app(w.annex<gpu::sync_work_items>(), Defs{rd1, s2})->projs<2>();

    auto [g4, s4] = w.app(w.annex<gpu::sync_work_items>(), Defs{ai_mem, s3})->projs<2>();
    after_inner->app(true, outer_yield, Defs{g4, s4, ai_acc});

    auto [inner_body, inner_for_call] = counting_for(w.lit_i64(tile), w.tuple({g3, accval0}), after_inner, w.sym("gemmInnerK"));
    outer_body->set(true, inner_for_call);
    auto [kk_iter, inner_acc, inner_yield] = inner_body->vars<3>();
    auto [g_in, accval_in]                 = inner_acc->projs<2>();
    auto kk_idx                             = w.call(core::conv::u, w.lit_nat(tile + 1), kk_iter);

    auto [rm0, v0] = w.call<mem::load>(Defs{s3, op_lea_tuple(shared0_ptr, w.tuple({par_local0_idx, kk_idx}))})->projs<2>();
    auto [rm1, v1] = w.call<mem::load>(Defs{rm0, op_lea_tuple(shared1_ptr, w.tuple({par_local1_idx, kk_idx}))})->projs<2>();

    apply_cps(w, inner_body, global_comb, {g_in, accval_in, w.tuple({v0, v1})}, inner_yield);

    return {kernel, shared_pack_ty};
}

/// Cap on `build_kernel_conv_tiled`'s thread-block size (the `ow` extent): a CUDA block is at most 1024
/// threads.
constexpr nat_t Conv_Max_Ow = 1024;

/// Cap on the cooperative weight load's round count (`ceil(filter_total / n_items)`, `n_items = ow`):
/// only `ow` sits on the thread dimension, so for late-network shapes (small spatial extent, large
/// channel count -- e.g. VGG16's 512x512x3x3 filter at a 28x28 or 14x14 layer) the load can need
/// *hundreds* of sequential, sub-warp-occupancy rounds, measured to cost 100s of ms for a single
/// call (388ms for 512x512x3x3 @ 28x28, batch 10) versus microseconds for an early, wide-`ow` layer
/// with the same kernel. Rather than fix the block shape now (folding more axes onto the thread
/// dimension), decline the tiled path outright when it would need too many rounds and fall back to
/// `build_kernel`, which has no such blind spot.
constexpr nat_t Conv_Max_Load_Iters = 8;

/// A shared-memory kernel for a detected convolution-shaped reduction (`ConvShape`): the whole weight
/// filter for one thread block's `cout` is cooperatively loaded into shared memory once (`filter_total`
/// elements, `n_items` at a time) and read from there by every thread in the block instead of every
/// thread re-reading it independently from global memory -- the classic conv reuse opportunity, since
/// the same filter is read by every output pixel of that `(n, cout, ...)` slice. The data input is
/// still read through the generic (global-memory) access-map machinery, but threads are laid out so
/// that `ow` -- the fastest-varying axis by construction, since a schedule only ever strip-mines a
/// *lower*-numbered position (see `ConvShape::data_positions`) -- maps directly onto the thread index,
/// so adjacent threads read adjacent (stride-1) addresses: coalesced access without any extra work.
/// Reuses the same write-back/epilogue shape as `build_kernel`.
std::pair<Lam*, const Def*> build_kernel_conv_tiled(World& w,
                             const Def* Ro,
                             nat_t ro,
                             nat_t rr,
                             const Def* Sr,
                             const Def* So,
                             const Mapped& ins,
                             const ConvShape& shape,
                             const Vector<nat_t>& out_dims,
                             const Def* To,
                             const Def* acc_out,
                             const Def* init,
                             Lam* global_comb,
                             const Mapped& post_ins,
                             Lam* global_post,
                             const Def* Tp,
                             const Def* out_dptr) {
    auto nis = ins.n();
    auto nps = post_ins.n();
    auto rn  = ro + rr;
    auto n   = w.lit_nat(rn);

    auto ow_pos    = shape.data_positions.back();
    auto ow_extent = out_dims[ow_pos];

    Vector<nat_t> outer_positions{shape.weight_position};
    for (auto p : shape.data_positions)
        if (p != ow_pos) outer_positions.push_back(p);
    std::sort(outer_positions.begin(), outer_positions.end());
    Vector<nat_t> outer_extents;
    for (auto p : outer_positions)
        outer_extents.push_back(out_dims[p]);
    auto n_groups = std::accumulate(outer_extents.begin(), outer_extents.end(), nat_t{1}, std::multiplies<>{});

    Vector<nat_t> filter_extents(rr);
    for (nat_t j = 0; j != rr; ++j)
        filter_extents[j] = *Lit::isa<nat_t>(Sr->proj(rn, ro + j));
    auto filter_total   = std::accumulate(filter_extents.begin(), filter_extents.end(), nat_t{1}, std::multiplies<>{});
    auto filter_strides = row_major_strides(filter_extents);

    Grid grid{n_groups, ow_extent, n_groups * ow_extent};

    // The cache is padded up to an exact multiple of the block size so the cooperative load below can
    // run unconditionally every round instead of needing a per-iteration bounds branch (see there).
    auto max_load_iters = (filter_total + grid.n_items - 1) / grid.n_items;
    auto padded_total    = max_load_iters * grid.n_items;

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();

    auto weight_elem_ty = scalar_elem_ty(ins.dptrs[shape.weight_idx], ins.rs[shape.weight_idx], ins.ss[shape.weight_idx]);
    auto shared_pack_ty = w.arr(w.lit_nat(padded_total), weight_elem_ty);
    auto shared_ptr_ty  = w.call<gpu::SharedPtr>(shared_pack_ty);

    DefVec arg_tys(nis + nps + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    for (size_t j = 0; j != nps; ++j)
        arg_tys[nis + j] = post_ins.dptrs[j]->type();
    arg_tys[nis + nps] = out_dptr->type();

    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma({shared_ptr_ty}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("convTiledKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + nps + 1, i);
    DefVec k_post_dptrs(nps);
    for (size_t j = 0; j != nps; ++j)
        k_post_dptrs[j] = k_args->proj(nis + nps + 1, nis + j);
    auto k_out_dptr = k_args->proj(nis + nps + 1, nis + nps);
    auto shared_ptr = k_shared_ptrs->proj(1, 0);

    auto mul_i64 = [&](const Def* a, const Def* b) { return w.call(core::wrap::mul, core::Mode::none, Defs{a, b}); };
    auto add_i64 = [&](const Def* a, const Def* b) { return w.call(core::wrap::add, core::Mode::none, Defs{a, b}); };

    // --- Entry: decompose (group_id, item_id) into every "outer" position's own coordinate plus `ow`. ---
    auto entry = w.mut_con(w.sigma(Defs{}))->set("convEntry");
    kernel->set(true, w.app(entry, w.tuple()));

    auto group_i64           = grid.n_groups == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), group_id);
    auto item_i64             = grid.n_items == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), item_id);
    auto [entry_mem, outer_coords] = unflatten_index(w, group_i64, outer_extents, k_global);
    auto ow_idx               = w.call(core::conv::u, Sr->proj(rn, ow_pos), item_i64);

    DefVec pos_val(ro);
    for (size_t j = 0; j != outer_positions.size(); ++j)
        pos_val[outer_positions[j]] = outer_coords[j];
    pos_val[ow_pos] = ow_idx;

    /// The full `rn`-length loop-vector for reading either input at the given reduction coordinates:
    /// `pos_val` (fixed for the whole kernel invocation) covers every parallel position, and the
    /// caller-supplied `reduction_vals` cover the trailing `rr` reduction positions.
    auto build_full_iters = [&](const DefVec& reduction_vals) {
        DefVec iv(rn);
        for (nat_t p = 0; p != ro; ++p)
            iv[p] = pos_val[p];
        for (nat_t j = 0; j != rr; ++j)
            iv[ro + j] = reduction_vals[j];
        return w.tuple(iv);
    };

    // --- Write-back (reused verbatim in shape from `build_kernel`): shared memory is only written
    // during the cooperative load below, so no updated token needs to flow back out here. ---
    auto write_back          = w.mut_con(Defs{global_ty, To})->set("convWriteBack");
    auto [wb_mem, acc_final] = write_back->vars<2>();
    DefVec wb_idx(rn);
    for (nat_t p = 0; p != ro; ++p)
        wb_idx[p] = pos_val[p];
    for (nat_t j = 0; j != rr; ++j)
        wb_idx[ro + j] = w.call(core::conv::u, Sr->proj(rn, ro + j), w.lit_i64(0));
    auto [wc_mem, write_coords] = affine_map(acc_out, Ro, n, Sr, So, w.tuple(wb_idx), wb_mem);

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

    auto after_post             = w.mut_con(Defs{global_ty, Tp})->set("convAfterPost");
    auto [post_mem, elem_post]  = after_post->vars<2>();
    auto final_mem
        = w.call<mem::store>(Defs{post_mem, op_lea_tuple(k_out_dptr, fold_index(So, write_coords)), elem_post});
    after_post->app(true, k_ret, Defs{final_mem, k_shared, k_const, k_local});
    apply_cps(w, write_back, global_post, {pcur, acc_final, w.tuple(post_elems)}, after_post);

    // --- Cooperative weight-filter load: `max_load_iters` rounds of `n_items` threads each loading
    // one element, unconditionally (see `padded_total` above) -- an out-of-range round's flat index
    // still wraps into a valid (if redundant) weight coordinate via `unflatten_index`'s modular
    // arithmetic, and lands in a cache slot >= `filter_total` that is never read back below.
    //
    // Uses `%affine.For`/`counting_for` (unlike a hand-rolled self-recursive mutable, which turned out
    // to send `World::app` into unbounded eager expansion), but with only *one* `%mem.M`-typed
    // component in the accumulator: `%affine.lower_for`'s rewrite (see
    // `affine::phase::LowerFor::rewrite_imm_App`) collapses every phi whose type is *any* `%mem.M`
    // address space onto one shared "bb mem" var, so a `[GlobalM, SharedM]` accumulator silently
    // merges the two (observed as `sync_work_items` receiving `«2; GlobalM»` instead of
    // `[GlobalM, SharedM]`). Side-stepped by never threading a per-iteration GlobalM at all: every
    // round's weight read uses the same fixed `entry_mem` (safe -- unlike the shared-memory *write*
    // below, a read's own "after" token needs no downstream use for the read itself to stay live).
    auto after_load          = w.mut_con(shared_ty)->set("convAfterLoad");
    auto load_init            = k_shared;
    auto [load_body, load_for_call]
        = counting_for(w.lit_i64(max_load_iters), load_init, after_load, w.sym("convWeightLoad"));
    entry->set(true, load_for_call);
    auto [load_iter, load_shared, load_yield] = load_body->vars<3>();

    auto load_flat_i64        = add_i64(mul_i64(load_iter, w.lit_i64(grid.n_items)), item_i64);
    auto [fm, filter_coords]  = unflatten_index(w, load_flat_i64, filter_extents, entry_mem);
    auto weight_iters         = build_full_iters(filter_coords);
    auto [wg, wcoords]        = affine_map(ins.accs[shape.weight_idx], ins.rs[shape.weight_idx], n, Sr,
                                           ins.ss[shape.weight_idx], weight_iters, fm);
    auto [rd_mem_, rd_val]    = w.call<mem::load>(
                                Defs{wg, op_lea_tuple(k_dptrs[shape.weight_idx], fold_index(ins.ss[shape.weight_idx], wcoords))})
                                ->projs<2>();
    auto load_flat_idx = w.call(core::conv::u, w.lit_nat(padded_total), load_flat_i64);
    auto stored        = w.call<mem::store>(Defs{load_shared, mem::op_lea_unsafe(shared_ptr, load_flat_idx), rd_val});
    load_body->app(true, load_yield, stored);

    auto reduction_entry        = w.mut_con(w.sigma(Defs{}))->set("convReductionEntry");
    auto [bar_mem, bar_shared] = w.app(w.annex<gpu::sync_work_items>(), Defs{entry_mem, after_load->var()})->projs<2>();
    after_load->set(true, w.app(reduction_entry, w.tuple()));

    // --- Per-thread reduction over the `rr` window dims: data comes from global memory (generic,
    // coalesced by construction), weight from the shared cache filled above. ---
    const Def* acc    = w.tuple({bar_mem, init});
    const Def* cont    = write_back;
    Lam* current_mut  = reduction_entry;
    DefVec red_idx, red_i64;
    red_idx.reserve(rr);
    red_i64.reserve(rr);
    for (nat_t j = 0; j != rr; ++j) {
        auto dim                    = Sr->proj(rn, ro + j);
        auto bound                  = w.call<core::bitcast>(w.type_i64(), dim);
        auto [rbody, for_call]      = counting_for(bound, acc, cont, w.sym("convRed_" + std::to_string(j)));
        auto [iter, new_acc, yield] = rbody->vars<3>();
        cont                        = yield;
        red_idx.push_back(w.call(core::conv::u, dim, iter));
        red_i64.push_back(iter);
        acc         = new_acc;
        current_mut->set(true, for_call);
        current_mut = rbody;
    }
    auto [red_mem, elem_acc] = acc->projs<2>();

    auto data_iters     = build_full_iters(red_idx);
    auto [dg, dcoords]  = affine_map(ins.accs[shape.data_idx], ins.rs[shape.data_idx], n, Sr, ins.ss[shape.data_idx],
                                     data_iters, red_mem);
    auto [dm, dval]     = w.call<mem::load>(
                            Defs{dg, op_lea_tuple(k_dptrs[shape.data_idx], fold_index(ins.ss[shape.data_idx], dcoords))})
                            ->projs<2>();

    const Def* wflat = w.lit_i64(0);
    for (nat_t j = 0; j != rr; ++j)
        wflat = add_i64(wflat, mul_i64(red_i64[j], w.lit_i64(filter_strides[j])));
    auto wflat_idx      = w.call(core::conv::u, w.lit_nat(filter_total), wflat);
    auto [wm, wval]     = w.call<mem::load>(Defs{bar_shared, mem::op_lea_unsafe(shared_ptr, wflat_idx)})->projs<2>();

    DefVec input_elems(2);
    input_elems[shape.data_idx]   = dval;
    input_elems[shape.weight_idx] = wval;
    apply_cps(w, current_mut, global_comb, {dm, elem_acc, w.tuple(input_elems)}, cont);

    return {kernel, shared_pack_ty};
}

bool is_pow2(nat_t n) { return n != 0 && (n & (n - 1)) == 0; }

/// A shared-memory tree-reduction kernel for a pure reduction (`ro == 0`, no epilogue inputs): up to
/// 1024 threads each fold one element via `comb`, then combine pairwise in shared memory, instead of a
/// single thread sequentially folding the whole range. Requires the flattened reduction size to be an
/// exact power of two and at most 1024 -- the caller falls back to `build_kernel` otherwise; scaling
/// beyond one block (a cross-block combine) is a natural follow-up, not attempted here.
std::pair<Lam*, const Def*> build_kernel_reduction(World& w,
                            const Def* Sr,
                            nat_t rr,
                            const Mapped& ins,
                            const Def* To,
                            Lam* global_comb,
                            const Def* init,
                            const Def* out_dptr,
                            nat_t total_k) {
    auto nis = ins.n();
    auto n   = w.lit_nat(rr);

    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();

    auto shared_pack_ty = w.arr(w.lit_nat(total_k), To);
    auto shared_ptr_ty  = w.call<gpu::SharedPtr>(shared_pack_ty);

    DefVec arg_tys(nis + 1);
    for (size_t i = 0; i != nis; ++i)
        arg_tys[i] = ins.dptrs[i]->type();
    arg_tys[nis] = out_dptr->type();

    Grid grid{1, total_k, total_k};
    auto kernel
        = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(grid.n_groups), w.type_idx(grid.n_items),
                         w.sigma({shared_ptr_ty}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
              ->set("reduceKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_shared_ptrs, k_args, k_ret] = kernel->vars<9>();

    DefVec k_dptrs(nis);
    for (size_t i = 0; i != nis; ++i)
        k_dptrs[i] = k_args->proj(nis + 1, i);
    auto k_out_dptr = k_args->proj(nis + 1, nis);
    auto shared_ptr = k_shared_ptrs->proj(1, 0);

    auto tid_i64 = total_k == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), item_id);

    // --- Each thread folds exactly one k-index (`rr`-dim coordinates unflattened from `tid`). ---
    Vector<nat_t> red_dims(rr);
    for (nat_t d = 0; d != rr; ++d)
        red_dims[d] = *Lit::isa<nat_t>(Sr->proj(rr, d));
    auto [mem1, coords] = unflatten_index(w, tid_i64, red_dims, k_global);
    auto iters           = w.tuple(coords);

    auto cur = mem1;
    DefVec input_elems(nis);
    for (size_t i = 0; i != nis; ++i) {
        auto [mc_mem, ecoords] = affine_map(ins.accs[i], ins.rs[i], n, Sr, ins.ss[i], iters, cur);
        cur                    = mc_mem;
        auto [rd_mem, rd_val]
            = w.call<mem::load>(Defs{cur, op_lea_tuple(k_dptrs[i], fold_index(ins.ss[i], ecoords))})->projs<2>();
        cur            = rd_mem;
        input_elems[i] = rd_val;
    }

    auto after_fold       = w.mut_con(Defs{global_ty, To})->set("reduceAfterFold");
    auto [af_mem, af_val] = after_fold->vars<2>();
    apply_cps(w, kernel, global_comb, {cur, init, w.tuple(input_elems)}, after_fold);

    auto stored0                 = w.call<mem::store>(Defs{k_shared, mem::op_lea_unsafe(shared_ptr, tid_i64), af_val});
    auto [bar0_mem, bar0_shared] = w.app(w.annex<gpu::sync_work_items>(), Defs{af_mem, stored0})->projs<2>();
    auto tree_entry               = w.mut_con(w.sigma(Defs{}))->set("reduceTree");
    after_fold->set(true, w.app(tree_entry, w.tuple()));

    // --- Tree reduction: each round halves the active range, guarded by a real branch (only threads
    // with `tid < stride` have an in-bounds partner) so every thread still reaches the barrier. ---
    Lam* current          = tree_entry;
    const Def* shared_tok = bar0_shared;
    for (nat_t stride = total_k / 2; stride >= 1; stride /= 2) {
        auto is_active = w.call(core::icmp::ul, Defs{tid_i64, w.lit_i64(stride)});
        auto do_update = w.mut_con(w.sigma(Defs{}))->set("reduceActive");
        auto skip      = w.mut_con(w.sigma(Defs{}))->set("reduceInactive");
        auto join      = w.mut_con(shared_ty)->set("reduceJoin");
        current->set(true, w.app(w.extract(w.tuple({skip, do_update}), is_active), w.tuple()));

        auto partner          = w.call(core::wrap::add, core::Mode::none, Defs{tid_i64, w.lit_i64(stride)});
        auto [m1, lo]         = w.call<mem::load>(Defs{shared_tok, mem::op_lea_unsafe(shared_ptr, tid_i64)})->projs<2>();
        auto [m2, hi]         = w.call<mem::load>(Defs{m1, mem::op_lea_unsafe(shared_ptr, partner)})->projs<2>();
        auto write_cont       = w.mut_con(Defs{global_ty, To})->set("reduceCombined");
        auto [wc_mem, wc_val] = write_cont->vars<2>();
        auto m3               = w.call<mem::store>(Defs{m2, mem::op_lea_unsafe(shared_ptr, tid_i64), wc_val});
        write_cont->app(true, join, Defs{m3});
        apply_cps(w, do_update, global_comb, {af_mem, lo, w.tuple({lo, hi})}, write_cont);
        skip->app(true, join, Defs{shared_tok});

        auto after_barrier         = w.mut_con(shared_ty)->set("reduceBarrier");
        auto [bar_mem, bar_shared] = w.app(w.annex<gpu::sync_work_items>(), Defs{af_mem, join->var()})->projs<2>();
        join->set(true, w.app(after_barrier, bar_shared));

        current    = after_barrier;
        shared_tok = after_barrier->var();
        if (stride == 1) break; // unsigned `stride /= 2` never reaches below 1
    }

    // --- Only thread 0 writes the fully-reduced result (no epilogue inputs: `nps == 0` is required by the caller). ---
    auto is_zero     = w.call(core::icmp::e, Defs{tid_i64, w.lit_i64(0)});
    auto do_write    = w.mut_con(w.sigma(Defs{}))->set("reduceWrite");
    auto skip_write  = w.mut_con(w.sigma(Defs{}))->set("reduceSkipWrite");
    auto after_write = w.mut_con(Defs{global_ty, shared_ty})->set("reduceAfterWrite");
    current->set(true, w.app(w.extract(w.tuple({skip_write, do_write}), is_zero), w.tuple()));

    auto [rd_mem, total_val] = w.call<mem::load>(Defs{shared_tok, mem::op_lea_unsafe(shared_ptr, u64{0})})->projs<2>();
    auto write_mem            = w.call<mem::store>(Defs{af_mem, k_out_dptr, total_val});
    do_write->app(true, after_write, Defs{write_mem, rd_mem});
    skip_write->app(true, after_write, Defs{af_mem, shared_tok});

    auto [aw_mem, aw_shared] = after_write->vars<2>();
    after_write->app(true, k_ret, Defs{aw_mem, aw_shared, k_const, k_local});

    return {kernel, shared_pack_ty};
}

Lam* build_teardown(World& w,
                    const Def* Ro,
                    const Def* So,
                    const Def* Tp,
                    Defs dptrs,
                    Defs post_dptrs,
                    const Def* out_dptr,
                    const Def* cont) {
    auto global_ty = w.annex<gpu::GlobalM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto mem_ty    = w.call<mem::M>(0);

    auto after_launch                        = w.mut_con(Defs{mem_ty, global_ty, const_ty})->set("afterLaunch");
    auto [post_mem, post_global, post_const] = after_launch->vars<3>();

    auto [alloc_mem, host_buf] = buffer::op_alloc(Ro, So, Tp, post_mem)->projs<2>();
    auto copy_back
        = w.app(w.app(w.annex<gpu::buf_copy_to_host>(), {Ro, So, Tp}), {alloc_mem, post_global, out_dptr, host_buf});
    auto [cb_mem, cb_global] = copy_back->projs<2>();

    auto cur_global = cb_global;
    for (auto dptr : dptrs)
        cur_global = w.call(gpu::free::block, Defs{cur_global, dptr});
    for (auto dptr : post_dptrs)
        cur_global = w.call(gpu::free::block, Defs{cur_global, dptr});
    cur_global = w.call(gpu::free::block, Defs{cur_global, out_dptr});

    auto final_mem = w.app(w.annex<gpu::auto_deinit>(), Defs{cb_mem, cur_global, post_const});
    after_launch->app(true, cont, Defs{final_mem, host_buf});
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
    Super::start();
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
    auto result_ty                      = rewrite(app->type());

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

    auto mem_ty                                    = w.call<mem::M>(0);
    auto rewritten_arg                             = rewrite(app->arg());
    auto [_, rewritten_inputs, rewritten_post_ins] = rewritten_arg->projs<3>();
    auto fun  = w.mut_fun(w.sigma({mem_ty, rewritten_inputs->type(), rewritten_post_ins->type()}), result_ty)
                    ->set("mapReduceAffGpu");
    auto call = w.app(cps::op_cps2ds_dep(fun), rewritten_arg);
    auto [fun_mem, new_inputs, new_post_ins] = fun->var(0_n)->projs<3>();
    auto cont                                = fun->var(1);

    auto [h_mem, h_global, h_const] = w.app(w.annex<gpu::auto_init>(), fun_mem)->projs<3>();

    auto in_desc = extract_input_desc(nis_n, Ris, Sis, Tis, accs);
    auto inputs  = alloc_copy_inputs(w, h_mem, h_global, in_desc.rs, in_desc.ss, in_desc.ts, new_inputs);

    auto post_desc = extract_input_desc(nps_n, Rps, Sps, Tps, post_accs);
    auto post_inputs
        = alloc_copy_inputs(w, inputs.mem, inputs.global, post_desc.rs, post_desc.ss, post_desc.ts, new_post_ins);

    auto [out_global, out_dptr] = alloc_output(w, post_inputs.global, Tp, So, ro);

    auto global_comb = rebuild_lam_global_mem(comb_lam, To, w.sym("combGlobal"));
    auto global_post = rebuild_lam_global_mem(post_lam, Tp, w.sym("postGlobal"));

    Mapped mapped_ins{in_desc.rs, in_desc.ss, inputs.dptrs, in_desc.accs};
    Mapped mapped_post{post_desc.rs, post_desc.ss, post_inputs.dptrs, post_desc.accs};

    // `sched` names the schedule `%tensor.dot_product_impl` already picked for this op (see
    // `probe_schedule`): `vdim` in the reduction range means it chose the reduction-vectorized nest,
    // i.e. a genuine contraction along that axis -- exactly the structure a specialized kernel below
    // wants, read off rather than rediscovered from the op's operands.
    auto sched_vu  = probe_schedule(sched);
    bool is_redvec = sched_vu && sched_vu->first >= ro && sched_vu->first < ro + rr;

    Lam* kernel = nullptr;
    const Def* shared_pack_ty = nullptr;
    nat_t launch_groups = 0, launch_items = 0;

    if (is_redvec && ro == 0 && nps_n == 0 && rr >= 1) {
        nat_t total_k = 1;
        for (nat_t d = 0; d != rr; ++d)
            total_k *= *Lit::isa<nat_t>(Sr->proj(ro + rr, d));
        if (is_pow2(total_k) && total_k <= 1024) {
            std::tie(kernel, shared_pack_ty)
                = build_kernel_reduction(w, Sr, rr, mapped_ins, To, global_comb, init, out_dptr, total_k);
            launch_groups = 1;
            launch_items  = total_k;
        }
    }
    // `build_kernel_gemm_tiled` flattens the whole reduction group into one GEMM "K" and tiles it;
    // for `rr > 1` (convolution's shape) that flattening measured slower than caching the whole
    // filter once (`detect_conv`/`build_kernel_conv_tiled` below), so restrict this path to true
    // single-axis GEMM/BMM shapes.
    if (!kernel && rr == 1) {
        if (auto shape = detect_gemm(mapped_ins, ro, rr, ro + rr)) {
            Vector<nat_t> k_extents;
            for (nat_t j = 0; j != rr; ++j)
                k_extents.push_back(*Lit::isa<nat_t>(Sr->proj(ro + rr, ro + j)));
            auto dimk = std::accumulate(k_extents.begin(), k_extents.end(), nat_t{1}, std::multiplies<>{});
            Vector<nat_t> m_extents, n_extents;
            for (auto p : shape->m_positions)
                m_extents.push_back(out_dims[p]);
            for (auto p : shape->n_positions)
                n_extents.push_back(out_dims[p]);
            auto m_total = std::accumulate(m_extents.begin(), m_extents.end(), nat_t{1}, std::multiplies<>{});
            auto n_total = std::accumulate(n_extents.begin(), n_extents.end(), nat_t{1}, std::multiplies<>{});
            if (m_total % Gemm_Tile == 0 && n_total % Gemm_Tile == 0 && dimk % Gemm_Tile == 0) {
                std::tie(kernel, shared_pack_ty) = build_kernel_gemm_tiled(
                    w, Ro, ro, rr, Sr, So, mapped_ins, *shape, m_extents, n_extents, To, acc_out, init, global_comb,
                    mapped_post, global_post, Tp, out_dptr);
                launch_groups = (m_total / Gemm_Tile) * (n_total / Gemm_Tile);
                launch_items  = Gemm_Tile * Gemm_Tile;
            }
        }
    }
    // Falls back further to a conv-specific kernel (weight cached whole, no K-tiling) for shapes the
    // tiled-GEMM path above declines -- chiefly a reduction extent that isn't a multiple of
    // `Gemm_Tile` (e.g. VGG16's first layer, `cin = 3`, filter 27).
    if (!kernel) {
        if (auto shape = detect_conv(mapped_ins, ro, rr, ro + rr)) {
            auto ow_extent   = out_dims[shape->data_positions.back()];
            nat_t filter_total = 1;
            for (nat_t j = 0; j != rr; ++j)
                filter_total *= *Lit::isa<nat_t>(Sr->proj(ro + rr, ro + j));
            auto max_load_iters = ow_extent == 0 ? 0 : (filter_total + ow_extent - 1) / ow_extent;
            if (ow_extent <= Conv_Max_Ow && max_load_iters <= Conv_Max_Load_Iters) {
                std::tie(kernel, shared_pack_ty) = build_kernel_conv_tiled(
                    w, Ro, ro, rr, Sr, So, mapped_ins, *shape, out_dims, To, acc_out, init, global_comb, mapped_post,
                    global_post, Tp, out_dptr);
                Vector<nat_t> outer_positions{shape->weight_position};
                for (auto p : shape->data_positions)
                    if (p != shape->data_positions.back()) outer_positions.push_back(p);
                nat_t n_groups = 1;
                for (auto p : outer_positions)
                    n_groups *= out_dims[p];
                launch_groups = n_groups;
                launch_items  = ow_extent;
            }
        }
    }
    if (!kernel) {
        auto grid = grid_layout(out_dims);
        kernel = build_kernel(w, Ro, rr, out_dims, Sr, So, mapped_ins, To, acc_out, init, global_comb, mapped_post,
                              global_post, Tp, out_dptr, grid);
        launch_groups = grid.n_groups;
        launch_items  = grid.n_items;
    }

    DefVec kernel_arg_tys(nis_n + nps_n + 1);
    for (nat_t i = 0; i != nis_n; ++i)
        kernel_arg_tys[i] = inputs.dptrs[i]->type();
    for (nat_t j = 0; j != nps_n; ++j)
        kernel_arg_tys[nis_n + j] = post_inputs.dptrs[j]->type();
    kernel_arg_tys[nis_n + nps_n] = out_dptr->type();

    auto launch = w.app(w.annex<gpu::launch>(), Defs{w.lit_nat(nis_n + nps_n + 1), w.tuple(kernel_arg_tys)});
    launch      = w.app(launch, Defs{w.lit_nat(launch_groups), w.lit_nat(launch_items), w.annex<gpu::default_stream>(),
                                     shared_pack_ty ? w.lit_tt() : w.lit_ff(),
                                     shared_pack_ty ? w.tuple({shared_pack_ty}) : w.tuple()});
    launch      = w.app(launch, kernel);

    DefVec kernel_args = inputs.dptrs;
    kernel_args.insert(kernel_args.end(), post_inputs.dptrs.begin(), post_inputs.dptrs.end());
    kernel_args.push_back(out_dptr);
    launch = w.app(launch, kernel_args);

    auto after_launch = build_teardown(w, Ro, So, Tp, inputs.dptrs, post_inputs.dptrs, out_dptr, cont);
    auto launch_call  = w.app(launch, Defs{w.tuple({post_inputs.mem, out_global, h_const}), after_launch});
    fun->set(true, launch_call);

    return call;
}

} // namespace mim::plug::gpu::phase
