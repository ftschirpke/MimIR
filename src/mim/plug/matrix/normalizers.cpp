#include <mim/normalize.h>

#include "mim/world.h"

#include "mim/plug/core/core.h"
#include "mim/plug/matrix/matrix.h"

namespace mim {

//
// This file implements many of the FlatLayout algebra operations found in CuTe and is inspired by the Python
// implementation see https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html see
// https://github.com/NVIDIA/cutlass/blob/main/python/pycute/layout.py
//
// Most of the implementations in this file work on the layouts assuming they are static i.e. all numbers are literals.
// TODO: Is there a way to perform the operations on the MimIR's `Nat` nodes directly and is there a use case for it?
//

namespace {

//
// Layout Algebra
//

using NatPair = std::pair<nat_t, nat_t>;

struct NatTuple {
    std::vector<std::variant<nat_t, NatTuple>> items;
};

using FlatNatTuple = Vector<nat_t>;

template<class Visitor>
static void visit(const NatTuple& tuple, Visitor&& visitor) {
    for (const auto& item : tuple.items) {
        switch (item.index()) {
            case 0: {
                nat_t value = std::get<0>(item);
                visitor(value);
                break;
            }
            case 1: {
                const NatTuple& nested_tuple = std::get<1>(item);
                visit(nested_tuple, visitor);
                break;
            }
        }
    }
}

struct RecLayout {
    NatTuple dims;
    NatTuple strides;
};

struct FlatLayout {
    FlatNatTuple dims;
    FlatNatTuple strides;
};

static RecLayout layout_elevate(FlatLayout&& flat_layout) {
    RecLayout result;
    for (auto dim : flat_layout.dims)
        result.dims.items.emplace_back(dim);
    for (auto stride : flat_layout.strides)
        result.strides.items.emplace_back(stride);
    return result;
}

static bool is_valid_dims_strides_pair(const NatTuple& dims, const NatTuple& strides) {
    if (dims.items.size() != strides.items.size()) return false;
    for (size_t i = 0; i < dims.items.size(); ++i) {
        if (dims.items[i].index() != strides.items[i].index()) return false;
        if (auto inner_dims = std::get_if<NatTuple>(&dims.items[i])) {
            auto inner_strides = std::get_if<NatTuple>(&strides.items[i]);
            if (!is_valid_dims_strides_pair(*inner_dims, *inner_strides)) return false;
        }
    }
    return true;
}

static bool is_valid_layout(const RecLayout& layout) { return is_valid_dims_strides_pair(layout.dims, layout.strides); }

static nat_t tuple_size(const NatTuple& tuple) {
    auto total_size = 1;
    visit(tuple, [&total_size](nat_t value) { total_size *= value; });
    return total_size;
}

static FlatNatTuple tuple_flatten(const NatTuple& tuple) {
    FlatNatTuple result;
    visit(tuple, [&result](nat_t value) { result.push_back(value); });
    return result;
}

static FlatLayout layout_flatten(const RecLayout& layout) {
    FlatLayout result{.dims = tuple_flatten(layout.dims), .strides = tuple_flatten(layout.strides)};
    assert(result.dims.size() == result.strides.size());
    return result;
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

static RecLayout layout_ith_sublayout(const RecLayout& layout, size_t i) {
    assert(i < layout.dims.items.size());
    assert(i < layout.strides.items.size());

    RecLayout result;
    switch (layout.dims.items[i].index()) {
        case 0: // nat_t
        {
            result.dims.items    = {std::get<0>(layout.dims.items[i])};
            result.strides.items = {std::get<0>(layout.strides.items[i])};
            break;
        }
        case 1: // NatTuple
        {
            result.dims    = std::get<1>(layout.dims.items[i]);
            result.strides = std::get<1>(layout.strides.items[i]);
            break;
        }
    }
    return result;
}

static FlatLayout layout_composition_integral(const RecLayout& layout, nat_t compose_dim, nat_t compose_stride) {
    FlatLayout result;
    nat_t rest_dim    = compose_dim;
    nat_t rest_stride = compose_stride;

    if (compose_stride == 0) {
        result.dims.push_back(compose_dim);
        result.strides.push_back(0);
        return result;
    }

    FlatLayout flattened = layout_flatten(layout);
    FlatLayout coalesced = coalesce(flattened);
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

static RecLayout impl_layout_composition(const RecLayout& layout1, const RecLayout& layout2) {
    assert(is_valid_layout(layout1));
    assert(is_valid_layout(layout2));

    auto is_integral = layout2.dims.items.size() == 1 && std::get_if<nat_t>(&layout2.dims.items[0]);
    if (is_integral) {
        auto dim    = std::get<nat_t>(layout2.dims.items[0]);
        auto stride = std::get<nat_t>(layout2.strides.items[0]);
        return layout_elevate(layout_composition_integral(layout1, dim, stride));
    }
    RecLayout result;
    for (size_t i = 0; i < layout2.dims.items.size(); ++i) {
        RecLayout sublayout2 = layout_ith_sublayout(layout2, i);
        RecLayout subresult  = impl_layout_composition(layout1, sublayout2);
        result.dims.items.emplace_back(std::move(subresult.dims));
        result.strides.items.emplace_back(std::move(subresult.strides));
    }
    return result;
}

static FlatLayout complement(const RecLayout& layout, nat_t size) {
    Vector<nat_t> dims;
    visit(layout.dims, [&dims](nat_t dim) { dims.push_back(dim); });

    Vector<NatPair> pairs;
    visit(layout.strides, [&dims, &pairs](nat_t stride) {
        size_t i = pairs.size();
        pairs.emplace_back(dims[i], stride);
    });

    std::sort(pairs.begin(), pairs.end(), [](const NatPair& a, const NatPair& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first < b.first;
    });

    Vector<nat_t> result_dims, result_strides;
    size_t current_idx = 1;
    for (const auto& [dim, stride] : pairs) {
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

static RecLayout layout_logical_divide(const RecLayout& layout, const RecLayout& tiler) {
    assert(is_valid_layout(layout));
    assert(is_valid_layout(tiler));
    auto is_integral_tiler = tiler.dims.items.size() == 1 && std::get_if<nat_t>(&tiler.dims.items[0]);
    if (is_integral_tiler) {
        FlatLayout tiler_flat_complement = complement(tiler, tuple_size(layout.dims));
        RecLayout tiler_complement       = layout_elevate(std::move(tiler_flat_complement));
        RecLayout tile                   = impl_layout_composition(layout, tiler);
        RecLayout outer                  = impl_layout_composition(layout, tiler_complement);
        return RecLayout{.dims    = {.items = {tile.dims, outer.dims}},
                         .strides = {.items = {tile.strides, outer.strides}}};
    }

    assert(layout.dims.items.size() >= tiler.dims.items.size());
    RecLayout result;
    for (size_t i = 0; i < tiler.dims.items.size(); ++i) {
        RecLayout sublayout = layout_ith_sublayout(layout, i);
        RecLayout subtiler  = layout_ith_sublayout(tiler, i);
        RecLayout subresult = layout_logical_divide(sublayout, subtiler);
        result.dims.items.emplace_back(std::move(subresult.dims));
        result.strides.items.emplace_back(std::move(subresult.strides));
    }
    for (size_t i = tiler.dims.items.size(); i < layout.dims.items.size(); ++i) {
        result.dims.items.push_back(layout.dims.items[i]);
        result.strides.items.push_back(layout.strides.items[i]);
    }
    return result;
}

static NatTuple prefix_product(const NatTuple& tup, nat_t init_val) {
    NatTuple result;
    for (const auto& element : tup.items) {
        switch (element.index()) {
            case 0: // nat_t
            {
                auto val = std::get<0>(element);
                result.items.push_back(init_val);
                init_val *= val;
                break;
            }
            case 1: // NatTuple
            {
                auto& inner_tup = std::get<1>(element);
                nat_t product   = 1;
                visit(inner_tup, [&product](nat_t val) { product *= val; });
                result.items.emplace_back(std::move(inner_tup));
                init_val *= product;
                break;
            }
        }
    }
    return result;
}

using DefPair = std::pair<const Def*, const Def*>;

static DefPair impl_idx_1DtoND(World& world, const Def* mem, const Def* idx, const RecLayout& layout) {
    auto is_integral = layout.dims.items.size() == 1 && std::get_if<nat_t>(&layout.dims.items[0]);
    if (is_integral) {
        auto dim    = std::get<nat_t>(layout.dims.items[0]);
        auto stride = std::get<nat_t>(layout.strides.items[0]);

        auto bitcast_calc   = world.call<plug::core::bitcast>(idx->type());
        auto bitcast_result = world.call<plug::core::bitcast>(world.type_idx(dim));

        auto dim_lit_nat    = world.lit_nat(dim);
        auto stride_lit_nat = world.lit_nat(stride);
        auto dim_lit        = world.app(bitcast_calc, dim_lit_nat);
        auto stride_lit     = world.app(bitcast_calc, stride_lit_nat);

        auto idx_div = world.call(plug::core::div::udiv, world.tuple({mem, world.tuple({idx, stride_lit})}));
        mem          = world.extract(idx_div, 2, 0);
        idx          = world.extract(idx_div, 2, 1);
        auto idx_rem = world.call(plug::core::div::urem, world.tuple({mem, world.tuple({idx, dim_lit})}));
        mem          = world.extract(idx_rem, 2, 0);
        idx          = world.extract(idx_rem, 2, 1);
        auto result  = world.app(bitcast_result, idx);
        return std::make_pair(mem, result);
    }

    DefVec tuple_entries;
    for (size_t i = 0; i < layout.dims.items.size(); ++i) {
        RecLayout sublayout = layout_ith_sublayout(layout, i);
        DefPair pair        = impl_idx_1DtoND(world, mem, idx, sublayout);
        mem                 = pair.first;
        tuple_entries.push_back(pair.second);
    }
    return std::make_pair(mem, world.tuple(tuple_entries));
}

//
// Layout Parsing
//
// TODO: rewrite parsing when MimIR supports recursive Nat tuples

static std::optional<Vector<nat_t>> extract_ms(nat_t n, const Def* ms_def) {
    Vector<nat_t> ms;
    if (ms_def->num_projs() != n) error("Failed to parse ms for n = {} given ms = {}", n, ms_def);
    for (size_t i = 0; i < n; ++i) {
        auto m_def = ms_def->proj(i);
        auto m_lit = Lit::isa(m_def);
        if (!m_lit.has_value()) return std::nullopt;
        auto m = m_lit.value();
        ms.push_back(m);
    }
    return ms;
}

static std::optional<NatTuple> tuple_from_def(const Def* tup) {
    NatTuple result;
    for (auto op : tup->projs())
        if (auto lit = Lit::isa(op))
            result.items.push_back(lit.value());
        else if (auto inner_tup = tuple_from_def(op))
            result.items.push_back(inner_tup.value());
        else
            return std::nullopt;
    return result;
}

static std::optional<RecLayout> extract_layout_static(const Def* layout_tup) {
    RecLayout result;

    auto [n_def, ms_def, dims_def, strides_def] = layout_tup->projs<4>();

    auto n_lit = Lit::isa(n_def);
    if (!n_lit) return std::nullopt;
    auto n = n_lit.value();

    auto ms_opt = extract_ms(n, ms_def);
    if (!ms_opt) return std::nullopt;
    auto& ms = ms_opt.value();

    if (auto dims_tup = tuple_from_def(dims_def))
        result.dims = dims_tup.value();
    else
        return std::nullopt;
    if (auto strides_tup = tuple_from_def(strides_def))
        result.strides = strides_tup.value();
    else
        return std::nullopt;

    assert(is_valid_layout(result));
    assert(n == result.dims.items.size());
    for (size_t i = 0; i < n; ++i)
        if (auto inner_dims = std::get_if<NatTuple>(&result.dims.items[i]))
            assert(ms[i] == inner_dims->items.size());
        else
            assert(ms[i] == 1);

    return result;
}

//
// Layout Generation
//
//
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

static const Def* make_tup(World& world, NatTuple&& tup) {
    DefVec outer_tup;
    for (const auto& outer_var : tup.items) {
        if (auto val = std::get_if<nat_t>(&outer_var)) {
            outer_tup.emplace_back(world.lit_nat(*val));
            continue;
        }
        DefVec inner_tup;
        auto& inner_vec = std::get<NatTuple>(outer_var).items;
        for (auto inner_var : inner_vec)
            if (auto val = std::get_if<nat_t>(&inner_var)) {
                inner_tup.emplace_back(world.lit_nat(*val));
                continue;
            } else
                error("MimIR currently only supports layout tuples to be nested two layers deep");
        outer_tup.emplace_back(world.tuple(inner_tup));
    }
    return world.tuple(outer_tup);
}

static const Def* make_layout(World& world, RecLayout&& layout) {
    assert(is_valid_layout(layout));

    nat_t n = layout.dims.items.size();
    DefVec ms;
    for (auto& element : layout.dims.items) {
        switch (element.index()) {
            case 0: ms.push_back(world.lit_nat_1()); break;
            case 1:
                const auto& inner = std::get<1>(element);
                auto m            = inner.items.size();
                ms.push_back(world.lit_nat(m));
                break;
        }
    }

    DefVec result;
    result.emplace_back(world.lit_nat(n));
    result.emplace_back(world.tuple(ms));
    result.emplace_back(make_tup(world, std::move(layout.dims)));
    result.emplace_back(make_tup(world, std::move(layout.strides)));
    return world.tuple(result);
}

} // namespace

namespace plug::matrix {

const Def* normalize_idx_1DtoND(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    world.WLOG("FRIEDRICH: still here 1");

    auto [mem, idx] = arg->projs<2>();
    world.WLOG("FRIEDRICH: still here 2");

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout_def = callee_app->arg();
    world.WLOG("FRIEDRICH: still here 4");

    auto layout_opt = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    const RecLayout& layout = layout_opt.value();
    world.WLOG("FRIEDRICH: still here 6");

    auto [result_mem, result_idx] = impl_idx_1DtoND(world, mem, idx, layout);
    world.WLOG("FRIEDRICH: still here 7");
    return world.tuple({result_mem, result_idx});
}

const Def* normalize_idx_NDto1D(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto [strides_n_def, strides_ms_def, strides_tup] = arg->projs<3>();

    auto strides_n_lit = Lit::isa(strides_n_def);
    if (!strides_n_lit) return {};
    auto strides_n = strides_n_lit.value();

    auto strides_ms_opt = extract_ms(strides_n, strides_ms_def);
    if (!strides_ms_opt) return {};
    auto strides_ms = strides_ms_opt.value();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto [dims, nd_idx]                      = callee_app->uncurry_args<2>();
    auto [dims_n_def, dims_ms_def, dims_tup] = dims->projs<3>();

    auto dims_n_lit = Lit::isa(dims_n_def);
    if (!dims_n_lit) return {};
    auto dims_n = dims_n_lit.value();

    auto dims_ms_opt = extract_ms(dims_n, dims_ms_def);
    if (!dims_ms_opt) return {};
    auto& dims_ms = dims_ms_opt.value();

    if (dims_n != strides_n || dims_ms != strides_ms) error("Dimensions and strides must align in size.");

    auto s = Idx::isa_lit(type);
    if (!s) return {};
    auto size = s.value();

    auto add               = world.call(core::wrap::add, world.lit_nat_0());
    auto nat_bitcast       = world.call<core::bitcast>(world.type_idx(size));
    const Def* dot_product = static_cast<const Def*>(world.lit_idx(size, 0));
    for (size_t i = 0; i < strides_n; ++i) {
        for (size_t j = 0; j < strides_ms[i]; ++j) {
            auto stride      = world.app(nat_bitcast, strides_tup->proj(i)->proj(j));
            auto idx_bitcast = world.call<core::bitcast>(world.type_idx(size));
            auto idx         = world.app(idx_bitcast, nd_idx->proj(i)->proj(j));
            auto mul         = world.call(core::wrap::mul, world.lit_nat_0());
            auto product     = world.app(mul, {stride, idx});
            dot_product      = world.app(add, {dot_product, product});
        }
    }

    world.ELOG("FRIEDRICH: successfully normalized idx_NDto1D!");

    return dot_product;
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
    auto& layout_tup = layout_opt.value();

    auto complement_layout = layout_elevate(complement(layout_tup, size));
    return make_layout(world, std::move(complement_layout));
}

const Def* normalize_layout_composition(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout1 = callee_app->arg();

    auto layout1_opt = extract_layout_static(layout1);
    if (!layout1_opt) return {};
    auto& layout1_tup = layout1_opt.value();

    auto layout2_opt = extract_layout_static(arg);
    if (!layout2_opt) return {};
    auto& layout2_tup = layout2_opt.value();

    RecLayout comp = impl_layout_composition(layout1_tup, layout2_tup);
    return make_layout(world, std::move(comp));
}

const Def* normalize_layout_zip_divide(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout_def = callee_app->arg();

    auto layout_opt = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    auto& layout_tup = layout_opt.value();

    auto tiler_layout_opt = extract_layout_static(arg);
    if (!tiler_layout_opt) return {};
    auto& tiler_layout_tup = tiler_layout_opt.value();

    RecLayout logical_div = layout_logical_divide(layout_tup, tiler_layout_tup);

    assert(logical_div.dims.items.size() == 2);
    RecLayout tile_layout  = layout_ith_sublayout(logical_div, 0);
    RecLayout outer_layout = layout_ith_sublayout(logical_div, 1);

    DefVec result;
    result.emplace_back(make_layout(world, std::move(tile_layout)));
    result.emplace_back(make_layout(world, std::move(outer_layout)));
    return world.tuple(result);
}

const Def* normalize_tile(const Def* type, const Def* callee, const Def* arg) { return {}; }

MIM_matrix_NORMALIZER_IMPL

} // namespace plug::matrix

} // namespace mim
