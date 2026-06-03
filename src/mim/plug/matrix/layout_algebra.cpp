#include "mim/plug/matrix/layout_algebra.h"

#include <mim/world.h>

#include <mim/plug/core/core.h>

namespace mim::plug::matrix::layalg {

//
// This file implements many of the Layout algebra operations found in CuTe and is inspired by the Python implementation
// * see https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html
// * see https://github.com/NVIDIA/cutlass/blob/main/python/pycute/layout.py
//
// The implementations in this file assume all layouts to be static, i.e. compile-time-known literals,
// and all indices to be dynamic, i.e. calculations are done in the IR.
// TODO: Is there a way to perform the operations on the MimIR's `Nat` nodes directly and is there a use case for it?

// TODO: remove
static void print_tuple(const NatTuple& tup) {
    print(std::cerr, "(");
    bool first = true;
    for (const auto& element : tup.items) {
        if (first)
            first = false;
        else
            print(std::cerr, ", ");
        switch (element.index()) {
            case 0: {
                auto value = std::get<0>(element);
                print(std::cerr, "{}", value);
                break;
            }
            case 1: {
                const auto& inner_tup = std::get<1>(element);
                print_tuple(inner_tup);
                break;
            }
        }
    }
    print(std::cerr, ")");
}

// TODO: remove
static void print_layout(const Layout& layout, const char* end) {
    print_tuple(layout.dims);
    print(std::cerr, " : ");
    print_tuple(layout.strides);
    print(std::cerr, end);
}

// TODO: remove
static void print_layout(const Layout& layout) { print_layout(layout, "\n"); }

using FlatNatTuple = Vector<nat_t>;

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

bool is_valid_layout(const Layout& layout) { return is_valid_dims_strides_pair(layout.dims, layout.strides); }

static NatTuple tuple_simplify(NatTuple&& tuple) {
    NatTuple result;
    for (auto&& element : tuple.items) {
        switch (element.index()) {
            case 0: {
                auto value = std::get<0>(element);
                result.items.emplace_back(value);
                break;
            }
            case 1: {
                auto& inner_tup     = std::get<1>(element);
                auto simplified_tup = tuple_simplify(std::move(inner_tup));
                if (simplified_tup.items.size() == 1) {
                    assert(std::get_if<nat_t>(&simplified_tup.items[0]));
                    result.items.emplace_back(simplified_tup.items[0]);
                } else {
                    result.items.emplace_back(simplified_tup);
                }
                break;
            }
        }
    }
    return result;
}

Layout simplify(Layout&& layout) {
    Layout result;
    result.dims    = tuple_simplify(std::move(layout.dims));
    result.strides = tuple_simplify(std::move(layout.strides));
    assert(is_valid_layout(result));
    return result;
}

static FlatNatTuple tuple_flatten(const NatTuple& tuple) {
    FlatNatTuple result;
    tuple_visit(tuple, [&result](nat_t value) { result.push_back(value); });
    return result;
}

FlatLayout flatten(const Layout& layout) {
    FlatLayout result{.dims = tuple_flatten(layout.dims), .strides = tuple_flatten(layout.strides)};
    assert(result.dims.size() == result.strides.size());
    return result;
}

Layout elevate(FlatLayout&& flat_layout) {
    Layout result;
    for (auto dim : flat_layout.dims)
        result.dims.items.emplace_back(dim);
    for (auto stride : flat_layout.strides)
        result.strides.items.emplace_back(stride);
    return result;
}

Layout sublayout(const Layout& layout, size_t i) {
    assert(i < layout.dims.items.size());
    assert(i < layout.strides.items.size());

    Layout result;
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

FlatLayout coalesce(const FlatLayout& layout) {
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

Layout concat(Vector<Layout>&& layouts, bool simplify) {
    Layout result;
    for (auto&& layout : layouts) {
        if (simplify && layout.dims.items.size() == 1) {
            assert(layout.strides.items.size() == 1);
            result.dims.items.emplace_back(std::move(layout.dims.items[0]));
            result.strides.items.emplace_back(std::move(layout.strides.items[0]));
        } else {
            result.dims.items.emplace_back(std::move(layout.dims));
            result.strides.items.emplace_back(std::move(layout.strides));
        }
    }
    return result;
}

Layout concat(Vector<Layout>&& layouts) { return concat(std::move(layouts), true); }

static FlatLayout integral_composition(const Layout& layout, nat_t compose_dim, nat_t compose_stride) {
    FlatLayout result;
    nat_t rest_dim    = compose_dim;
    nat_t rest_stride = compose_stride;

    if (compose_stride == 0) {
        result.dims.push_back(compose_dim);
        result.strides.push_back(0);
        return result;
    }

    FlatLayout flattened = flatten(layout);
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

Layout composition(const Layout& layout1, const Layout& layout2) {
    assert(is_valid_layout(layout1));
    assert(is_valid_layout(layout2));

    auto is_integral = layout2.dims.items.size() == 1 && std::get_if<nat_t>(&layout2.dims.items[0]);
    if (is_integral) {
        auto dim    = std::get<nat_t>(layout2.dims.items[0]);
        auto stride = std::get<nat_t>(layout2.strides.items[0]);
        return elevate(integral_composition(layout1, dim, stride));
    }
    Layout result;
    for (size_t i = 0; i < layout2.dims.items.size(); ++i) {
        Layout sublayout2 = sublayout(layout2, i);
        Layout subresult  = composition(layout1, sublayout2);
        result.dims.items.emplace_back(std::move(subresult.dims));
        result.strides.items.emplace_back(std::move(subresult.strides));
    }
    return result;
}

FlatLayout complement(const Layout& layout, nat_t size) {
    Vector<nat_t> dims;
    tuple_visit(layout.dims, [&dims](nat_t dim) { dims.push_back(dim); });

    using NatPair = std::pair<nat_t, nat_t>;
    Vector<NatPair> pairs;
    tuple_visit(layout.strides, [&dims, &pairs](nat_t stride) {
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

Layout logical_divide(const Layout& layout, const Layout& tiler) {
    assert(is_valid_layout(layout));
    assert(is_valid_layout(tiler));

    FlatLayout tiler_flat_complement = complement(tiler, tuple_size(layout.dims));
    Layout tiler_complement          = elevate(std::move(tiler_flat_complement));
    Layout merged                    = concat({tiler, tiler_complement});
    Layout result                    = composition(layout, merged);
    return result;
}

std::pair<Layout, Layout> zipped_divide(const Layout& layout, const Vector<Layout>& tiler_layouts) {
    Layout tile_layout;
    Layout outer_layout;
    if (tiler_layouts.size() == 1) {
        auto& tiler   = tiler_layouts[0];
        Layout result = logical_divide(layout, tiler);
        tile_layout   = sublayout(result, 0);
        outer_layout  = sublayout(result, 1);
    } else {
        Vector<Layout> tile_layouts;
        Vector<Layout> outer_layouts;
        for (size_t i = 0; i < tiler_layouts.size(); ++i) {
            auto& subtiler = tiler_layouts[i];

            Layout sublay    = sublayout(layout, i);
            Layout subresult = logical_divide(sublay, subtiler);
            tile_layouts.emplace_back(sublayout(subresult, 0));
            outer_layouts.emplace_back(sublayout(subresult, 1));
        }
        for (size_t i = tiler_layouts.size(); i < layout.dims.items.size(); ++i)
            outer_layouts.emplace_back(sublayout(layout, i));
        tile_layout  = concat(std::move(tile_layouts), false);
        outer_layout = concat(std::move(outer_layouts), false);
    }
    tile_layout  = simplify(std::move(tile_layout));
    outer_layout = simplify(std::move(outer_layout));
    return std::make_pair(tile_layout, outer_layout);
}

static NatTuple impl_tuple_prefix_product(nat_t start, const NatTuple& tuple) {
    NatTuple result;
    for (const auto& element : tuple.items) {
        switch (element.index()) {
            case 0: // nat_t
            {
                auto value = std::get<0>(element);
                result.items.push_back(start);
                start *= value;
                break;
            }
            case 1: // NatTuple
            {
                const auto& inner_tuple = std::get<1>(element);
                nat_t product           = 1;
                tuple_visit(inner_tuple, [&product](nat_t value) { product *= value; });
                result.items.emplace_back(impl_tuple_prefix_product(start, inner_tuple));
                start *= product;
                break;
            }
        }
    }
    return result;
}

NatTuple tuple_prefix_product(const NatTuple& tuple) { return impl_tuple_prefix_product(1, tuple); }

std::pair<const Def*, const Def*> idx_1DtoND(World& world, const Def* mem, const Def* idx, const Layout& layout) {
    auto is_integral = layout.dims.items.size() == 1 && std::get_if<nat_t>(&layout.dims.items[0]);
    if (is_integral) {
        auto dim    = std::get<nat_t>(layout.dims.items[0]);
        auto stride = std::get<nat_t>(layout.strides.items[0]);

        if (stride != 1 && 2 * stride <= dim) {
            auto stride_lit_nat = world.lit_nat(stride);
            auto bitcast_idx    = world.call<plug::core::bitcast>(idx->type());
            auto stride_lit     = world.app(bitcast_idx, stride_lit_nat);
            auto idx_div        = world.call(plug::core::div::udiv, world.tuple({mem, world.tuple({idx, stride_lit})}));
            mem                 = world.extract(idx_div, 2, 0);
            idx                 = world.extract(idx_div, 2, 1);
            auto new_idx_size   = dim / stride + (dim % stride > 0);
            auto bitcast_div    = world.call<plug::core::bitcast>(world.type_idx(new_idx_size));
            idx                 = world.app(bitcast_div, idx);
        }

        auto idx_n = Idx::isa_lit(idx->type());
        if (!idx_n || idx_n.value() > dim) {
            auto dim_lit_nat = world.lit_nat(dim);
            auto bitcast_idx = world.call<plug::core::bitcast>(idx->type());
            auto dim_lit     = world.app(bitcast_idx, dim_lit_nat);
            auto idx_rem     = world.call(plug::core::div::urem, world.tuple({mem, world.tuple({idx, dim_lit})}));
            mem              = world.extract(idx_rem, 2, 0);
            idx              = world.extract(idx_rem, 2, 1);
        }

        auto bitcast_result = world.call<plug::core::bitcast>(world.type_idx(dim));
        auto result         = world.app(bitcast_result, idx);
        return std::make_pair(mem, result);
    }

    DefVec tuple_entries;
    for (size_t i = 0; i < layout.dims.items.size(); ++i) {
        Layout sublay = sublayout(layout, i);
        auto pair     = idx_1DtoND(world, mem, idx, sublay);
        mem           = pair.first;
        tuple_entries.push_back(pair.second);
    }
    return std::make_pair(mem, world.tuple(tuple_entries));
}

const Def* idx_NDto1D(World& world, nat_t idx_size, const NatTuple& strides, const Def* nd_idx) {
    assert(strides.items.size() == nd_idx->num_projs());

    auto add               = world.call(core::wrap::add, world.lit_nat_0());
    auto mul               = world.call(core::wrap::mul, world.lit_nat_0());
    const Def* dot_product = static_cast<const Def*>(world.lit_idx(idx_size, 0));
    for (size_t i = 0; i < strides.items.size(); ++i) {
        auto bitcast = world.call<core::bitcast>(world.type_idx(idx_size));
        auto& stride = strides.items[i];
        const Def* product;
        switch (stride.index()) {
            case 0: // nat_t
            {
                auto lit_stride = std::get<0>(stride);
                auto idx_stride = world.lit_idx(idx_size, lit_stride);
                auto idx        = world.app(bitcast, nd_idx->proj(i));
                product         = world.app(mul, {idx_stride, idx});
                break;
            }
            case 1: // NatTuple
            {
                const auto& tup_stride = std::get<1>(stride);
                product                = idx_NDto1D(world, idx_size, tup_stride, nd_idx->proj(i));
                break;
            }
        }
        dot_product = world.app(add, {dot_product, product});
    }
    return dot_product;
}

} // namespace mim::plug::matrix::layalg
