#include "mim/plug/tensor/phase/fuse.h"

#include "mim/def.h"
#include "mim/lam.h"

#include "mim/util/types.h"

#include "mim/plug/core/core.h"
#include "mim/plug/direct/direct.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

// Fuses an outer `tensor.map_reduce` with any number of its inputs — and, recursively, any
// fusible inputs of those inputs — whenever each such input is itself a `tensor.map_reduce`
// whose subs only reference output indices (i.e. the inner reduction is empty, so reading the
// inner tensor at a position is just a single call to the inner combination function).
//
// Outer:    map_reduce nis_o (To, Ro) So (Tis_o, Ris_o, Sis_o) (f_o, init_o) subs_o is_o
// Inner:    map_reduce nis_k (To_k, Ro_k) So_k (Tis_k, Ris_k, Sis_k) (f_k, init_k) subs_k is_k
//           for every fusible input — possibly nested inside another fusible input
//
// Result:   map_reduce nis_new (To, Ro) So (Tis_new, Ris_new, Sis_new) (f_new, init_o) subs_new
//           is_new
//
// The collection phase walks the tree of fusible inner map_reduces below `app` once, producing
// a flat list of *leaves* (the surviving tensor inputs of the fused mr) and *inner nodes* (the
// inner combiners that must run before `f_o`). Each fusible input is replaced by its inner's
// inputs, with subs remapped through the outer subs at that position; the remapping composes
// across nested levels. The new combination function `f_new` invokes every inner combiner in
// post-order — each starting from its own init — and finally invokes `f_o`, threading inner
// results into the corresponding outer input slots.
const Def* Fuse::fuse_map_reduce(const App* app) {
    auto outer_callee = app->callee()->as<App>();

    auto subs            = outer_callee->arg();
    auto [comb, init]    = outer_callee->decurry()->args<2>();
    auto [Tis, Ris, Sis] = outer_callee->decurry()->decurry()->args<3>();
    auto So              = outer_callee->decurry()->decurry()->decurry()->arg();
    auto [To, Ro]        = outer_callee->decurry()->decurry()->decurry()->decurry()->args<2>();
    auto nis             = outer_callee->decurry()->decurry()->decurry()->decurry()->decurry()->arg();
    auto is              = app->arg();

    auto nis_lit = nis->isa<Lit>();
    if (!nis_lit) return nullptr;
    auto nis_nat = nis_lit->get<u64>();

    struct InnerInfo {
        bool fusible    = false;
        const Def* subs = nullptr;
        const Def* comb = nullptr;
        const Def* init = nullptr;
        const Def* Tis  = nullptr;
        const Def* Ris  = nullptr;
        const Def* Sis  = nullptr;
        const Def* To   = nullptr;
        u64 nis         = 0;
        const Def* is   = nullptr;
        Vector<u64> Ris_nats;
        u64 outer_Ris_nat     = 0;
        const Def* outer_subs = nullptr;
    };

    Vector<InnerInfo> infos(nis_nat);
    bool any_fusible = false;

    for (u64 k = 0; k < nis_nat; ++k) {
        auto input_k = rewrite(is->proj(nis_nat, k));
        auto inner   = Axm::isa<tensor::map_reduce>(input_k);
        if (!inner) continue;

        auto inner_callee                      = inner->callee()->as<App>();
        auto inner_subs                        = inner_callee->arg();
        auto [inner_comb, inner_init]          = inner_callee->decurry()->args<2>();
        auto [inner_Tis, inner_Ris, inner_Sis] = inner_callee->decurry()->decurry()->args<3>();
        auto [inner_To, inner_Ro]              = inner_callee->decurry()->decurry()->decurry()->decurry()->args<2>();
        auto inner_nis = inner_callee->decurry()->decurry()->decurry()->decurry()->decurry()->arg();
        auto inner_is  = inner->arg();

        auto inner_nis_lit = inner_nis->isa<Lit>();
        auto inner_Ro_lit  = inner_Ro->isa<Lit>();
        if (!inner_nis_lit || !inner_Ro_lit) continue;
        auto inner_nis_nat = inner_nis_lit->get<u64>();
        auto inner_Ro_nat  = inner_Ro_lit->get<u64>();

        // We can only fuse when the inner has no reduction dimensions, i.e. all subs are < Ro_i.
        // In that case the inner tensor at any position is just a single call of `inner_comb`.
        Vector<u64> inner_Ris_nats(inner_nis_nat);
        bool fusible = true;
        for (u64 l = 0; l < inner_nis_nat && fusible; ++l) {
            auto Ris_l_lit = Lit::isa(inner_Ris->proj(inner_nis_nat, l));
            if (!Ris_l_lit) {
                fusible = false;
                break;
            }
            inner_Ris_nats[l] = *Ris_l_lit;
            auto inner_subs_l = inner_subs->proj(inner_nis_nat, l);
            for (u64 j = 0; j < inner_Ris_nats[l]; ++j) {
                auto idx_lit = Lit::isa(inner_subs_l->proj(inner_Ris_nats[l], j));
                if (!idx_lit || *idx_lit >= inner_Ro_nat) {
                    fusible = false;
                    break;
                }
            }
        }
        if (!fusible) continue;

        // We need the outer subs for input `k` to be indexable by literal positions.
        auto Ris_k_lit = Lit::isa(Ris->proj(nis_nat, k));
        if (!Ris_k_lit) continue;

        auto& info         = infos[k];
        info.fusible       = true;
        info.subs          = inner_subs;
        info.comb          = inner_comb;
        info.init          = inner_init;
        info.Tis           = inner_Tis;
        info.Ris           = inner_Ris;
        info.Sis           = inner_Sis;
        info.To            = inner_To;
        info.nis           = inner_nis_nat;
        info.is            = inner_is;
        info.Ris_nats      = std::move(inner_Ris_nats);
        info.outer_Ris_nat = *Ris_k_lit;
        info.outer_subs    = subs->proj(nis_nat, k);
        any_fusible        = true;
    }

    if (!any_fusible) return nullptr;

    auto& w = new_world();

    // Each fusible outer input k is replaced by `infos[k].nis` slots in the fused input list;
    // every non-fusible input retains exactly one slot. `new_pos[i]` is the start of input i's
    // slot range in the fused list.
    Vector<u64> new_pos(nis_nat);
    u64 new_nis_nat = 0;
    for (u64 i = 0; i < nis_nat; ++i) {
        new_pos[i] = new_nis_nat;
        new_nis_nat += infos[i].fusible ? infos[i].nis : 1;
    }

    DefVec new_Tis_vec(new_nis_nat);
    DefVec new_Ris_vec(new_nis_nat);
    DefVec new_Sis_vec(new_nis_nat);
    DefVec new_subs_vec(new_nis_nat);
    DefVec new_is_vec(new_nis_nat);

    for (u64 i = 0; i < nis_nat; ++i) {
        if (infos[i].fusible) {
            const auto& info = infos[i];
            for (u64 l = 0; l < info.nis; ++l) {
                auto pos         = new_pos[i] + l;
                new_Tis_vec[pos] = rewrite(info.Tis->proj(info.nis, l));
                new_Ris_vec[pos] = rewrite(info.Ris->proj(info.nis, l));
                new_Sis_vec[pos] = rewrite(info.Sis->proj(info.nis, l));
                new_is_vec[pos]  = rewrite(info.is->proj(info.nis, l));

                auto inner_subs_l = info.subs->proj(info.nis, l);
                DefVec subs_l_vec(info.Ris_nats[l]);
                for (u64 j = 0; j < info.Ris_nats[l]; ++j) {
                    // The inner refers to one of its own output indices; remap that into the
                    // outer index space via the outer subs at position `i`.
                    auto inner_idx = *Lit::isa(inner_subs_l->proj(info.Ris_nats[l], j));
                    subs_l_vec[j]  = rewrite(info.outer_subs->proj(info.outer_Ris_nat, inner_idx));
                }
                new_subs_vec[pos] = w.tuple(subs_l_vec);
            }
        } else {
            auto pos          = new_pos[i];
            new_Tis_vec[pos]  = rewrite(Tis->proj(nis_nat, i));
            new_Ris_vec[pos]  = rewrite(Ris->proj(nis_nat, i));
            new_Sis_vec[pos]  = rewrite(Sis->proj(nis_nat, i));
            new_subs_vec[pos] = rewrite(subs->proj(nis_nat, i));
            new_is_vec[pos]   = rewrite(is->proj(nis_nat, i));
        }
    }

    auto new_Tis  = w.tuple(new_Tis_vec);
    auto new_Ris  = w.tuple(new_Ris_vec);
    auto new_Sis  = w.tuple(new_Sis_vec);
    auto new_subs = w.tuple(new_subs_vec);
    auto new_is   = w.tuple(new_is_vec);

    auto new_nis_def = w.lit_nat(new_nis_nat);
    auto new_To      = rewrite(To);
    auto new_Ro      = rewrite(Ro);
    auto new_So      = rewrite(So);
    auto new_init    = rewrite(init);

    auto new_outer_comb = rewrite(comb);

    // Build the fused combination function:
    //
    //   cn f_new(data: [To, [new_Tis ...]], ret: cn To) =
    //       cn inner_ret_<r>(value_<r>: inner_To_<r>) = ...
    //       f_<fused[0]>((init_<fused[0]>, inner_inputs_<fused[0]>), inner_ret_0)
    //
    //   inner_ret_<r>(value_<r>):
    //       if r is not the last fused input:
    //           f_<fused[r+1]>((init_<fused[r+1]>, inner_inputs_<fused[r+1]>), inner_ret_<r+1>)
    //       else:
    //           f_o((acc, outer_inputs), ret)
    //
    // `outer_inputs[i]` is `value_<r>` when input i is the r-th fused input, and the
    // corresponding `new_in` slot otherwise. Each `inner_ret_<r>` closes over the prior
    // `value_<j>`s as free variables — those are bound by the dynamic call chain.
    auto inputs_sigma = w.sigma(new_Tis_vec);
    auto data_sigma   = w.sigma({new_To, inputs_sigma});
    auto ret_cn_type  = w.cn(new_To);
    auto new_comb     = w.mut_con({data_sigma, ret_cn_type})->set("fused_comb");
    auto new_data     = new_comb->var(0);
    auto new_ret      = new_comb->var(1);
    auto new_acc      = new_data->proj(2, 0);
    auto new_in       = new_data->proj(2, 1);

    Vector<u64> fused_indices;
    for (u64 i = 0; i < nis_nat; ++i)
        if (infos[i].fusible) fused_indices.emplace_back(i);

    Vector<Lam*> inner_rets(fused_indices.size());
    Vector<const Def*> inner_values(fused_indices.size());
    for (size_t r = 0; r < fused_indices.size(); ++r) {
        auto new_inner_To = rewrite(infos[fused_indices[r]].To);
        inner_rets[r]     = w.mut_con(new_inner_To)->set("inner_ret");
        inner_values[r]   = inner_rets[r]->var(0);
    }

    // Map each outer input position to its value at the f_o call site.
    DefVec outer_inputs_vec(nis_nat);
    {
        size_t r = 0;
        for (u64 i = 0; i < nis_nat; ++i)
            if (infos[i].fusible)
                outer_inputs_vec[i] = inner_values[r++];
            else
                outer_inputs_vec[i] = new_in->proj(new_nis_nat, new_pos[i]);
    }

    // Chain: caller for fused step r is new_comb (r==0) or inner_rets[r-1] (otherwise).
    for (size_t r = 0; r < fused_indices.size(); ++r) {
        auto k              = fused_indices[r];
        auto new_inner_comb = rewrite(infos[k].comb);
        auto new_inner_init = rewrite(infos[k].init);

        DefVec inner_inputs_vec(infos[k].nis);
        for (u64 l = 0; l < infos[k].nis; ++l)
            inner_inputs_vec[l] = new_in->proj(new_nis_nat, new_pos[k] + l);

        Lam* caller = (r == 0) ? new_comb : inner_rets[r - 1];
        caller->app(true, new_inner_comb, {w.tuple({new_inner_init, w.tuple(inner_inputs_vec)}), inner_rets[r]});
    }

    // After every inner combiner has produced its value, call the outer combiner.
    inner_rets.back()->app(true, new_outer_comb, {w.tuple({new_acc, w.tuple(outer_inputs_vec)}), new_ret});

    // Construct the fused map_reduce.
    auto mr = w.annex<tensor::map_reduce>();
    mr      = w.app(mr, new_nis_def);
    mr      = w.app(mr, {new_To, new_Ro});
    mr      = w.app(mr, new_So);
    mr      = w.app(mr, {new_Tis, new_Ris, new_Sis});
    mr      = w.app(mr, {new_comb, new_init});
    mr      = w.app(mr, new_subs);
    mr      = w.app(mr, new_is);

    return mr;
}

// A def already in the new world is the result of our own previous rewriting (e.g. a fused
// map_reduce produced during an iterative re-fuse). Rewriting it again would create an
// independent duplicate — in particular, walking through one of its Vars would re-rewrite the
// var's binding mut and create a second copy of an enclosing extern, leaving the original copy
// with a free var. Short-circuit so iterative fusion can re-enter `fuse_map_reduce` safely.
const Def* Fuse::rewrite(const Def* d) {
    if (&d->world() == &new_world()) return d;
    return Rewriter::rewrite(d);
}

const Def* Fuse::rewrite_imm_App(const App* app) {
    if (auto mr = Axm::isa<tensor::map_reduce>(app)) {
        if (auto res = fuse_map_reduce(mr)) return res;
    }
    return Rewriter::rewrite_imm_App(app);
}

} // namespace mim::plug::tensor::phase
