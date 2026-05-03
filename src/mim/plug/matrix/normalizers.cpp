#include <mim/normalize.h>

#include "mim/world.h"

#include "mim/plug/core/core.h"
#include "mim/plug/matrix/matrix.h"

namespace mim {

/*
 * This file implements many of the FlatLayout algebra operations found in CuTe and is inspired by the Python
 * implementation see https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html see
 * https://github.com/NVIDIA/cutlass/blob/main/python/pycute/layout.py
 *
 * Most of the implementations in this file work on the layouts assuming they are static i.e. all numbers are literals.
 * TODO: Is there a way to perform the operations on the MimIR's `Nat` nodes directly and is there a use case for it?
 */

namespace {

struct DynLayout {
    nat_t n;
    Vector<nat_t> ms;
    const Def* dim_defs;
    const Def* stride_defs;
};

static std::optional<DynLayout> extract_layout_dynamic(const Def* layout_tup) {
    auto [n_def, ms_def, dims_def, strides_def] = layout_tup->projs<4>();

    auto n_lit = Lit::isa(n_def);
    if (!n_lit.has_value()) return std::nullopt;
    auto n = n_lit.value();

    Vector<nat_t> ms;
    for (size_t i = 0; i < n; ++i) {
        auto m_def = ms_def->proj(i);
        auto m_lit = Lit::isa(m_def);
        if (!m_lit.has_value()) return std::nullopt;
        auto m = m_lit.value();
        ms.push_back(m);
    }

    return DynLayout{.n = n, .ms = ms, .dim_defs = dims_def, .stride_defs = strides_def};
}

struct StaticLayout {
    nat_t n;
    Vector<nat_t> ms;
    Vector<Vector<nat_t>> dims;
    Vector<Vector<nat_t>> strides;
};

static nat_t layout_size(const StaticLayout& layout) {
    nat_t product = 0;
    for (const auto& dim_vec : layout.dims)
        for (auto dim : dim_vec)
            product *= dim;
    return product;
}

static std::optional<StaticLayout> extract_layout_static(const Def* dims_tup) {
    auto dyn_layout = extract_layout_dynamic(dims_tup);
    if (!dyn_layout) return std::nullopt;
    auto out = StaticLayout{.n = dyn_layout->n, .ms = dyn_layout->ms, .dims = {}, .strides = {}};
    for (size_t i = 0; i < out.n; ++i) {
        auto& dims    = out.dims.emplace_back();
        auto& strides = out.strides.emplace_back();
        for (size_t j = 0; j < out.ms[i]; ++j) {
            auto dim_def = dyn_layout->dim_defs->proj(i)->proj(j);
            auto dim_lit = Lit::isa(dim_def);
            if (!dim_lit.has_value()) return std::nullopt;
            auto dim_val = dim_lit.value();
            dims.push_back(dim_val);

            auto stride_def = dyn_layout->stride_defs->proj(i)->proj(j);
            auto stride_lit = Lit::isa(stride_def);
            if (!stride_lit.has_value()) return std::nullopt;
            auto stride_val = stride_lit.value();
            strides.push_back(stride_val);
        }
    }
    return out;
}

static const Def* make_dims(World& world, Vector<nat_t>&& dims_vec) {
    nat_t n = dims_vec.size();
    DefVec result;
    result.emplace_back(world.lit_nat(n));
    DefVec ms, dims;

    for (size_t i = 0; i < n; ++i) {
        DefVec dim;
        nat_t m = 1;
        ms.emplace_back(world.lit_nat(m));
        nat_t val = dims_vec[i];
        dim.emplace_back(world.lit_nat(val));
        dims.emplace_back(world.tuple(dim));
    }

    result.emplace_back(world.tuple(ms));
    result.emplace_back(world.tuple(dims));
    return world.tuple(result);
}

static const Def* make_dims(World& world, Vector<Vector<nat_t>>&& dims_vec) {
    nat_t n = dims_vec.size();
    DefVec result;
    result.emplace_back(world.lit_nat(n));
    DefVec ms, dims;

    for (size_t i = 0; i < n; ++i) {
        DefVec dim;
        nat_t m = dims_vec[i].size();
        ms.emplace_back(world.lit_nat(m));
        for (size_t j = 0; j < m; ++j) {
            nat_t val = dims_vec[i][j];
            dim.emplace_back(world.lit_nat(val));
        }
        dims.emplace_back(world.tuple(dim));
    }

    result.emplace_back(world.tuple(ms));
    result.emplace_back(world.tuple(dims));
    return world.tuple(result);
}

static const Def* make_layout(World& world, StaticLayout&& layout) {
    DefVec result;
    result.emplace_back(make_dims(world, std::move(layout.dims)));
    result.emplace_back(make_dims(world, std::move(layout.strides)));
    return world.tuple(result);
}

struct FlatLayout {
    Vector<nat_t> dims;
    Vector<nat_t> strides;
};

nat_t layout_size(const FlatLayout& layout) {
    nat_t product = 0;
    for (auto dim : layout.dims)
        product *= dim;
    return product;
}

static FlatLayout flatten(const StaticLayout& layout) {
    FlatLayout result;

    for (auto dim_vec : layout.dims)
        for (auto dim : dim_vec)
            result.dims.push_back(dim);

    for (auto stride_vec : layout.strides)
        for (auto stride : stride_vec)
            result.strides.push_back(stride);

    return result;
}

static const Def* make_layout(World& world, FlatLayout&& layout) {
    DefVec result;
    result.emplace_back(make_dims(world, std::move(layout.dims)));
    result.emplace_back(make_dims(world, std::move(layout.strides)));
    return world.tuple(result);
}

static FlatLayout coalesce(const FlatLayout& layout) {
    FlatLayout result;

    assert(layout.dims.size() == layout.strides.size());

    result.dims.push_back(1);
    result.strides.push_back(0);

    for (size_t i = 0; i < layout.dims.size(); ++i) {
        auto dim    = layout.dims[i];
        auto stride = layout.strides[i];
        if (dim == 1) {
            continue;
        } else if (result.dims.back() == 1) {
            result.dims.back()    = dim;
            result.strides.back() = stride;
        } else if (result.dims.back() * result.strides.back() == stride) {
            result.dims.back() = result.dims.back() * dim;
        } else {
            result.dims.push_back(dim);
            result.strides.push_back(stride);
        }
    }

    return result;
}

static FlatLayout composition(const FlatLayout& layout, nat_t compose_dim, nat_t compose_stride) {
    FlatLayout result;
    nat_t rest_dim    = compose_dim;
    nat_t rest_stride = compose_stride;

    print(std::cerr, "FRIEDRICH: composition with {} : {}\n", compose_dim, compose_stride);

    if (compose_stride == 0) {
        result.dims.push_back(compose_dim);
        result.strides.push_back(0);
        return result;
    }

    FlatLayout coalesced = coalesce(layout);
    assert(coalesced.dims.size() == coalesced.strides.size());
    for (size_t i = 0; i < coalesced.dims.size() - 1; ++i) {
        auto current_dim    = coalesced.dims[i];
        auto current_stride = coalesced.strides[i];
        assert(current_dim % rest_stride == 0 || rest_stride % current_dim == 0);
        auto new_dim = std::min(std::max(1ul, current_dim / rest_stride), rest_dim);
        if (new_dim != 1) {
            result.dims.push_back(new_dim);
            result.strides.push_back(rest_stride * current_stride);
        }

        rest_dim /= new_dim;

        auto stride_div = rest_stride / current_dim;
        auto stride_rem = rest_stride % current_dim;
        if (stride_rem > 0)
            rest_stride = stride_div + 1;
        else
            rest_stride = stride_div;
    }

    if (rest_dim != 1 || result.dims.size() == 0) {
        result.dims.push_back(rest_dim);
        result.strides.push_back(rest_stride * coalesced.strides.back());
    }

    return result;
}

static StaticLayout composition_with_layout(const FlatLayout& layout1, const FlatLayout& layout2) {
    StaticLayout result;

    assert(layout2.dims.size() == layout2.strides.size());
    print(std::cerr, "FRIEDRICH: layout composition\n");
    for (size_t i = 0; i < layout2.dims.size(); ++i) {
        print(std::cerr, "FRIEDRICH: i = {}\n", i);
        FlatLayout intermediate = composition(layout1, layout2.dims[i], layout2.strides[i]);
        assert(intermediate.dims.size() == intermediate.strides.size());
        result.dims.emplace_back(std::move(intermediate.dims));
        result.strides.emplace_back(std::move(intermediate.strides));
    }
    print(std::cerr, "FRIEDRICH: computed layout composition\n");
    return result;
}

static FlatLayout composition_with_tile(FlatLayout layout, Vector<std::pair<nat_t, nat_t>> tile) {
    FlatLayout result;

    assert(layout.dims.size() == layout.strides.size());
    assert(tile.size() < layout.dims.size());

    for (size_t i = 0; i < layout.dims.size(); ++i) {
        if (i < tile.size()) {
            FlatLayout sublayout{.dims = {layout.dims[i]}, .strides = {layout.strides[i]}};
            FlatLayout intermediate = composition(sublayout, tile[i].first, tile[i].second);
            assert(intermediate.dims.size() == intermediate.strides.size());
            for (size_t j = 0; i < intermediate.dims.size(); ++j) {
                result.dims.push_back(intermediate.dims[j]);
                result.strides.push_back(intermediate.strides[j]);
            }
        } else {
            result.dims.push_back(layout.dims[i]);
            result.strides.push_back(layout.strides[i]);
        }
    }
    return result;
}

static FlatLayout complement(const StaticLayout& layout, nat_t size) {
    using IdxPair = std::pair<size_t, size_t>;
    Vector<IdxPair> index_order;
    for (size_t i = 0; i < layout.n; ++i)
        for (size_t j = 0; j < layout.ms[i]; ++j)
            index_order.emplace_back(i, j);

    std::sort(index_order.begin(), index_order.end(), [&layout](IdxPair a, IdxPair b) {
        auto [ai, aj] = a;
        auto [bi, bj] = b;
        auto a_dim    = layout.dims[ai][aj];
        auto b_dim    = layout.dims[bi][bj];
        if (a_dim == b_dim) {
            auto a_stride = layout.strides[ai][aj];
            auto b_stride = layout.strides[bi][bj];
            return a_stride < b_stride;
        }
        return a_dim < b_dim;
    });

    Vector<nat_t> result_dims, result_strides;
    size_t current_idx = 1;
    for (const auto& [i, j] : index_order) {
        auto dim    = layout.dims[i][j];
        auto stride = layout.strides[i][j];
        if (dim == 1 || stride == 0) continue;

        bool is_inbound = current_idx <= dim * stride;
        assert(is_inbound);

        result_dims.push_back(stride / current_idx);
        result_strides.push_back(current_idx);
        current_idx = dim * stride;
    }
    result_dims.push_back((size + current_idx - 1) / current_idx);
    result_strides.push_back(current_idx);

    FlatLayout coalesced = coalesce(FlatLayout{.dims = result_dims, .strides = result_strides});
    return coalesced;
}

} // namespace

namespace plug::matrix {

const Def* normalize_idx_NDto1D(const Def* type, const Def* callee, const Def* arg) {
    return {};
    // auto& world = type->world();

    // auto strides_opt = extract_dims_dynamic(arg);
    // if (!strides_opt) return {};
    // auto strides_tup = strides_opt.value();

    // auto callee_app = callee->isa<App>();
    // assert(callee_app);
    // auto [dims, nd_idx] = callee_app->uncurry_args<2>();

    // auto dims_opt = extract_dims_dynamic(dims);
    // if (!dims_opt) return {};
    // auto dims_tup = dims_opt.value();

    // if (dims_tup.n != strides_tup.n || dims_tup.ms != strides_tup.ms)
    //     error("Dimensions and strides must align in size.");

    // auto s = Idx::isa_lit(type);
    // if (!s) return {};
    // auto size = s.value();

    // auto add               = world.call(core::wrap::add, world.lit_nat_0());
    // auto nat_bitcast       = world.call<core::bitcast>(world.type_idx(size));
    // const Def* dot_product = static_cast<const Def*>(world.lit_idx(size, 0));
    // for (size_t i = 0; i < strides_tup.n; ++i) {
    //     for (size_t j = 0; j < strides_tup.ms[i]; ++j) {
    //         auto stride      = world.app(nat_bitcast, strides_tup.def->proj(i)->proj(j));
    //         auto idx_bitcast = world.call<core::bitcast>(world.type_idx(size));
    //         auto idx         = world.app(idx_bitcast, nd_idx->proj(i)->proj(j));
    //         auto mul         = world.call(core::wrap::mul, world.lit_nat_0());
    //         auto product     = world.app(mul, {stride, idx});
    //         dot_product      = world.app(add, {dot_product, product});
    //     }
    // }

    // world.ELOG("FRIEDRICH: successfully normalized idx_NDto1D!");

    // return dot_product;
}

const Def* normalize_layout_complement(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto size_lit = Lit::isa(arg);
    if (!size_lit) return {};
    auto size = size_lit.value();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout = callee_app->arg();

    auto layout_opt = extract_layout_static(layout);
    if (!layout_opt) return {};
    auto layout_tup = layout_opt.value();

    auto complement_layout = complement(layout_tup, size);
    return make_layout(world, std::move(complement_layout));
}

const Def* normalize_layout_composition(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout1 = callee_app->arg();

    auto layout1_opt = extract_layout_static(layout1);
    if (!layout1_opt) return {};
    auto layout1_tup = layout1_opt.value();

    auto layout2_opt = extract_layout_static(arg);
    if (!layout2_opt) return {};
    auto layout2_tup = layout2_opt.value();

    FlatLayout layout1_flat = flatten(layout1_tup);
    FlatLayout layout2_flat = flatten(layout2_tup);

    StaticLayout comp = composition_with_layout(layout1_flat, layout2_flat);
    return make_layout(world, std::move(comp));
}

const Def* normalize_layout_zip_divide(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout_def = callee_app->arg();

    auto layout_opt = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    auto layout_tup = layout_opt.value();

    auto tiler_layout_opt = extract_layout_static(arg);
    if (!tiler_layout_opt) return {};
    auto tiler_layout_tup = tiler_layout_opt.value();

    FlatLayout layout = flatten(layout_tup);

    assert(layout_tup.n >= tiler_layout_tup.n);
    for (size_t i = 0; i < tiler_layout_tup.n; ++i) {
        StaticLayout tiler;
        tiler.n  = tiler_layout_tup.ms[i];
        tiler.ms = Vector<nat_t>(tiler.n, 1);
        for (size_t j = 0; j < tiler.n; ++j) {
            tiler.dims.push_back({tiler_layout_tup.dims[i][j]});
            tiler.strides.push_back({tiler_layout_tup.strides[i][j]});
        }

        // TODO: fix implementation - this is wrong
        FlatLayout tiler_complement = complement(tiler, layout_size(layout_tup));

        // world.ELOG("FRIEDRICH: still here 4");

        // // TODO: do all this for each mode
        // FlatLayout layout2_complement = complement(layout2_tup, layout_size(layout1_tup));
        // world.ELOG("FRIEDRICH: still here 5");
        // FlatLayout layout1 = flatten(layout1_tup);
        // world.ELOG("FRIEDRICH: still here 6");
        // FlatLayout layout2 = flatten(layout2_tup);
        // world.ELOG("FRIEDRICH: still here 7");

        // StaticLayout layout_tile = composition_with_layout(layout1, layout2);
        // world.ELOG("FRIEDRICH: still here 8");
        // StaticLayout layout_outer = composition_with_layout(layout1, layout2_complement);
        // world.ELOG("FRIEDRICH: still here 9");

        // DefVec result_defs;
        // result_defs.push_back(make_layout(world, std::move(layout_outer)));
        // result_defs.push_back(make_layout(world, std::move(layout_tile)));
        // return world.tuple(result_defs);
    }

    return {};
}

const Def* normalize_tile(const Def* type, const Def* callee, const Def* arg) { return {}; }

MIM_matrix_NORMALIZER_IMPL

} // namespace plug::matrix

} // namespace mim
