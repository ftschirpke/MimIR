#include <mim/plug/affine/affine.h>
#include <mim/plug/core/core.h>
#include <mim/plug/direct/direct.h>
#include <mim/plug/tuple/tuple.h>
#include <mim/plug/vec/vec.h>

#include "mim/def.h"
#include "mim/plugin.h"
#include "mim/world.h"

#include "mim/util/sets.h"

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

// There's no good reason keeping 1s around for get/set indices.
// So this just skips relevant dimensions in the index and shape, and reduces the rank accordingly.
std::tuple<u64, const Def*, const Def*> fold_shape_and_index(const Def* shape, const Def* index) {
    auto& w = shape->world();

    DefVec dims;
    DefVec index_dims;
    auto r = shape->num_projs();
    for (size_t i = 0, e = r; i != e; ++i) {
        auto dim = shape->proj(r, i);
        if (auto dim_lit = dim->isa<Lit>())
            if (dim_lit->get<u64>() == 1) continue;

        dims.push_back(dim);
        index_dims.push_back(index->proj(r, i));
    }

    assert(dims.size() == index_dims.size());
    return std::make_tuple(dims.size(), w.tuple(dims), w.tuple(index_dims));
}

const Def* normalize_get(const Def*, const Def* c, const Def* arg) {
    auto& w = c->world();

    auto [arr, index] = arg->projs<2>();
    auto callee       = c->as<App>();
    auto [T, r, s]    = callee->args<3>();

    w.DLOG("normalize_get");
    w.DLOG("    arr = {} : {}", arr, arr->type());
    w.DLOG("    index = {} : {}", index, index->type());
    w.DLOG("    T = {} : {}", T, T->type());
    w.DLOG("    r = {} : {}", r, r->type());
    w.DLOG("    s = {} : {}", s, s->type());

    if (r->isa<Lit>()) {
        auto [new_r, new_s, new_index] = fold_shape_and_index(s, index);
        w.DLOG("    new_index = {} : {}", new_index, new_index->type());
        w.DLOG("    new_s = {} : {}", new_s, new_s->type());
        w.DLOG("    new_r = {} : {}", w.lit_nat(new_r), w.lit_nat(new_r)->type());
        if (new_r == 0) return arr;
        if (new_s != s || new_index != index) return op_get(T, w.lit_nat(new_r), new_s, arr, new_index);
    }

    if (Axm::isa<tensor::set>(arr)) {
        w.DLOG("get after set, try to bypass");
        auto set                  = arr->as<App>();
        auto [_, target_index, x] = set->args<3>();
        if (target_index == index) {
            w.DLOG("bypass successful");
            return x;
        }
    }
    if (Axm::isa<tensor::get>(arr)) {
        w.DLOG("get after get, try to bypass");
        auto get                      = arr->as<App>();
        auto [outer_arr, outer_index] = get->args<2>();
        auto [o_T, o_r, o_s]          = get->callee()->as<App>()->args<3>();
        w.DLOG("    outer_arr = {} : {}", outer_arr, outer_arr->type());
        w.DLOG("    outer_index = {} : {}", outer_index, outer_index->type());
        w.DLOG("    o_T = {} : {}", o_T, o_T->type());
        w.DLOG("    o_r = {} : {}", o_r, o_r->type());
        w.DLOG("    o_s = {} : {}", o_s, o_s->type());

        auto new_r     = w.call(core::nat::add, DefVec{r, o_r});
        auto new_s     = w.call<tuple::cat>(DefVec{o_s, s});
        auto new_index = w.call<tuple::cat>(DefVec{outer_index, index});

        return op_get(T, new_r, new_s, outer_arr, new_index);
    }

    return nullptr;
}

const Def* normalize_set(const Def*, const Def* c, const Def* arg) {
    auto& w = c->world();

    auto [arr, index, x] = arg->projs<3>();
    w.DLOG("normalize_set");
    w.DLOG("    arr = {} : {}", arr, arr->type());
    w.DLOG("    index = {} : {}", index, index->type());
    w.DLOG("    x = {} : {}", x, x->type());

    auto callee    = c->as<App>();
    auto [T, r, s] = callee->args<3>();

    if (r->isa<Lit>()) {
        auto [new_r, new_s, new_index] = fold_shape_and_index(s, index);
        w.DLOG("    new_index = {} : {}", new_index, new_index->type());
        w.DLOG("    new_s = {} : {}", new_s, new_s->type());
        w.DLOG("    new_r = {} : {}", w.lit_nat(new_r), w.lit_nat(new_r)->type());
        if (new_r == 0) return x;
        if (new_s != s || new_index != index) return op_set(T, w.lit_nat(new_r), new_s, arr, new_index, x);
    }

    if (Axm::isa<tensor::get>(x)) {
        w.DLOG("set after get, try to bypass");
        auto get                      = x->as<App>();
        auto [inner_arr, inner_index] = get->args<2>();
        if (inner_arr == arr && inner_index == index) {
            w.DLOG("bypass successful");
            return inner_arr;
        }
    }

    if (Axm::isa<tensor::set>(x)) {
        w.DLOG("set after set, try to bypass");
        auto inner_set                         = x->as<App>();
        auto [inner_arr, inner_index, inner_x] = inner_set->args<3>();
        auto [i_T, i_r, i_s]                   = inner_set->callee()->as<App>()->args<3>();

        w.DLOG("    inner_arr = {} : {}", inner_arr, inner_arr->type());
        w.DLOG("    inner_index = {} : {}", inner_index, inner_index->type());
        w.DLOG("    inner_x = {} : {}", inner_x, inner_x->type());
        w.DLOG("    i_T = {} : {}", i_T, i_T->type());
        w.DLOG("    i_r = {} : {}", i_r, i_r->type());
        w.DLOG("    i_s = {} : {}", i_s, i_s->type());

        if (auto inner_get = Axm::isa<tensor::get>(inner_arr)) {
            auto [g_arr, g_index] = inner_get->args<2>();
            if (g_arr == arr && g_index == index) {
                auto new_r     = w.call(core::nat::add, DefVec{r, i_r});
                auto new_s     = w.call<tuple::cat>(DefVec{s, i_s});
                auto new_index = w.call<tuple::cat>(DefVec{index, inner_index});

                return op_set(i_T, new_r, new_s, arr, new_index, inner_x);
            }
        }
        w.DLOG("set after set bypass not applicable: inner_arr is not get(arr, index)");
    }
    w.DLOG("no normalization applicable");
    return nullptr;
}

const Def* normalize_broadcast(const Def*, const Def* c, const Def* arg) {
    auto& w = c->world();

    auto [s_in, s_out, input] = arg->projs<3>();
    auto callee               = c->as<App>();
    auto [T, r]               = callee->args<2>();
    w.DLOG("normalize_broadcast");
    w.DLOG("    s_out = {} : {}", s_out, s_out->type());
    w.DLOG("    input = {} : {}", input, input->type());
    w.DLOG("    T = {} : {}", T, T->type());
    w.DLOG("    r = {} : {}", r, r->type());
    w.DLOG("    s_in = {} : {}", s_in, s_in->type());

    if (s_in == s_out) return input;

    auto r_lit = r->isa<Lit>();
    if (!r_lit) return nullptr;
    auto r_nat = r_lit->get<u64>();
    if(r_nat == 0) return input;
    
    return nullptr;
}

const Def* normalize_broadcast_in_dim(const Def*, const Def* c, const Def* arg) {
    auto& w = c->world();

    auto [s_in, s_out, input, index] = arg->projs<4>();
    auto callee                      = c->as<App>();
    auto [T, r_in, r_out]            = callee->args<3>();
    w.DLOG("normalize_broadcast_in_dim");
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

    auto tr = w.annex<tensor::transpose>();
    tr      = w.app(tr, {T, r_out, s_tr});
    tr      = w.app(tr, {input, permutation});

    auto s_bc_vec = DefVec(r_out_nat, [&](size_t i) { return s_tr->proj(r_out_nat, map_perm.at(i)); });
    auto s_bc     = w.tuple(s_bc_vec);

    auto bc = w.annex<tensor::broadcast>();
    bc      = w.app(bc, {T, r_out});
    bc      = w.app(bc, {s_bc, s_out, tr});

    return bc;
}

const Def* normalize_map_reduce(const Def*, const Def*, const Def*) {
    // TODO: is there anything we can normalize here?
    return nullptr;
}

MIM_tensor_NORMALIZER_IMPL

} // namespace mim::plug::tensor
