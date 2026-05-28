#include "mim/plug/tensor/phase/lower.h"

#include "mim/def.h"
#include "mim/lam.h"

#include "mim/util/types.h"

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

const Def* Lower::lower_via_impl(const App* app, const Def* impl_annex) {
    auto& w = new_world();

    // Walk the curry chain (innermost App outermost in syntax) to collect the args
    // in the order they were applied.
    DefVec args;
    const Def* head = app;
    while (auto h = head->isa<App>()) {
        args.push_back(rewrite(h->arg()));
        head = h->callee();
    }
    std::reverse(args.begin(), args.end());

    auto impl = impl_annex;
    for (auto a : args)
        impl = w.app(impl, a);

    // The `_impl` is a `lam`, so applying it triggers beta-reduction. Each `_impl`
    // body references the `_impl` variants of its dependencies directly, so the
    // chain bottoms out at the low-level axioms (`map_reduce`, …) in one go.
    return impl;
}

const Def* Lower::lower_broadcast_in_dim(const App* app) {
    auto& w  = new_world();
    auto c   = rewrite(app->callee());
    auto arg = rewrite(app->arg());

    auto [s_in, s_out, input, index] = arg->projs<4>();
    auto callee                      = c->as<App>();
    auto [T, r_in, r_out]            = callee->args<3>();
    w.DLOG("lower_broadcast_in_dim");
    w.DLOG("    s_out = {} : {}", s_out, s_out->type());
    w.DLOG("    input = {} : {}", input, input->type());
    w.DLOG("    index = {} : {}", index, index->type());
    w.DLOG("    T = {} : {}", T, T->type());
    w.DLOG("    r_in = {} : {}", r_in, r_in->type());
    w.DLOG("    r_out = {} : {}", r_out, r_out->type());
    w.DLOG("    s_in = {} : {}", s_in, s_in->type());

    auto r_in_lit = r_in->isa<Lit>();
    if (!r_in_lit) return nullptr;
    auto r_in_nat  = r_in_lit->get<u64>();
    auto r_out_lit = r_out->isa<Lit>();
    if (!r_out_lit) return nullptr;
    auto r_out_nat = r_out_lit->get<u64>();

    auto s_tr_vec = DefVec(r_out_nat, [&](size_t i) {
        if (i < r_in_nat) return s_in->proj(r_in_nat, i);
        return w.lit_nat_1()->as<Def>();
    });
    auto s_tr     = w.tuple(s_tr_vec);

    std::set<u64> set_perm;
    std::map<u64, u64> map_perm;
    for (u64 i = 0; i < r_out_nat; ++i)
        set_perm.insert(i);
    for (u64 i = 0; i < r_in_nat; ++i) {
        auto idx     = index->proj(r_in_nat, i);
        auto idx_lit = Lit::isa(idx);
        if (!idx_lit) return nullptr;
        u64 idx_nat = *idx_lit;

        map_perm[idx_nat] = i;

        set_perm.erase(idx_nat);
    }
    u64 j = r_in_nat;
    for (auto i = set_perm.begin(); i != set_perm.end(); i++) {
        map_perm[*i] = j;
        j++;
    }
    auto permutation_vec = DefVec(r_out_nat, [&](size_t i) { return w.lit_idx(r_out_nat, map_perm[i]); });
    auto permutation     = w.tuple(permutation_vec);

    // Apply `transpose_impl` directly so the lam expands to `%tensor.map_reduce`
    // immediately — no further high-level lowering is needed for the transpose.
    auto tr = w.annex<tensor::transpose_impl>();
    tr      = w.app(tr, {T, r_out, s_tr});
    tr      = w.app(tr, {input, permutation});

    auto s_bc_vec = DefVec(r_out_nat, [&](size_t i) { return s_tr->proj(r_out_nat, map_perm.at(i)); });
    auto s_bc     = w.tuple(s_bc_vec);

    auto bc = w.annex<tensor::broadcast>();
    bc      = w.app(bc, {T, r_out});
    bc      = w.app(bc, {s_bc, s_out, tr});

    // The resulting `%tensor.broadcast` is low-level and is handled by the
    // `LowerMapReduce` phase.
    return bc;
}

const Def* Lower::rewrite_imm_App(const App* app) {
    auto& w = new_world();

    if (auto bid = Axm::isa<tensor::broadcast_in_dim>(app)) {
        if (auto res = lower_broadcast_in_dim(bid)) return res;
    } else if (Axm::isa<tensor::product_2d>(app)) {
        return lower_via_impl(app, w.annex<tensor::product_2d_impl>());
    } else if (Axm::isa<tensor::dot_product>(app)) {
        return lower_via_impl(app, w.annex<tensor::dot_product_impl>());
    } else if (Axm::isa<tensor::transpose>(app)) {
        return lower_via_impl(app, w.annex<tensor::transpose_impl>());
    } else if (Axm::isa<tensor::transpose_2d>(app)) {
        return lower_via_impl(app, w.annex<tensor::transpose_2d_impl>());
    } else if (Axm::isa<tensor::map>(app)) {
        return lower_via_impl(app, w.annex<tensor::map_impl>());
    } else if (Axm::isa<tensor::map_reduce_pure>(app)) {
        return lower_via_impl(app, w.annex<tensor::map_reduce_pure_impl>());
    } else if (Axm::isa<tensor::unary>(app)) {
        return lower_via_impl(app, w.annex<tensor::unary_impl>());
    } else if (Axm::isa<tensor::binary>(app)) {
        return lower_via_impl(app, w.annex<tensor::binary_impl>());
    } else if (Axm::isa<tensor::select>(app)) {
        return lower_via_impl(app, w.annex<tensor::select_impl>());
    }
    return Rewriter::rewrite_imm_App(app);
}

} // namespace mim::plug::tensor::phase
