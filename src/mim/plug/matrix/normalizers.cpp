#include <mim/normalize.h>

#include <mim/plug/matrix/layout_algebra.h>

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

using namespace plug::matrix::layalg;

//
// Layout Algebra
//

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

static void print_layout(const Layout& layout, const char* end) {
    print_tuple(layout.dims);
    print(std::cerr, " : ");
    print_tuple(layout.strides);
    print(std::cerr, end);
}

static void print_layout(const Layout& layout) { print_layout(layout, "\n"); }

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

static std::optional<NatTuple> tuple_from_def(const Def* def) {
    NatTuple result;
    if (def->type()->isa<Nat>()) {
        if (auto lit = Lit::isa(def))
            result.items = {lit.value()};
        else
            return std::nullopt;
    } else if (auto pack = def->isa<Pack>()) {
        auto arity_lit = Lit::isa(pack->arity());
        auto value_lit = Lit::isa(pack->body());
        if (!arity_lit || !value_lit) return std::nullopt;
        for (size_t i = 0; i < arity_lit.value(); ++i)
            result.items.push_back(value_lit.value());
    } else if (auto tup = def->isa<Tuple>()) {
        for (auto op : tup->projs())
            if (auto lit = Lit::isa(op))
                result.items.push_back(lit.value());
            else if (auto inner_tup = tuple_from_def(op))
                result.items.push_back(inner_tup.value());
            else
                return std::nullopt;
    } else {
        return std::nullopt;
    }
    return result;
}

static std::optional<Layout> extract_layout_static(const Def* layout_tup) {
    Layout result;

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

    print(std::cerr, "FRIEDRICH: extracted ");
    print_layout(result);

    assert(is_valid_layout(result));
    assert(n == result.dims.items.size());
    for (size_t i = 0; i < n; ++i)
        if (auto inner_dims = std::get_if<NatTuple>(&result.dims.items[i]))
            assert(ms[i] == inner_dims->items.size());
        else
            assert(ms[i] == 1);

    // if (result.dims.items.empty() || result.strides.items.empty()) return std::nullopt; // TODO: is this needed

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

static const Def* make_layout(World& world, Layout&& layout) {
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

const Def* normalize_size(const Def* type, const Def* _, const Def* arg) {
    auto& world = type->world();

    auto [dims_n, dims_ms, dims_def, __] = arg->projs<4>();

    auto dims_tup = tuple_from_def(dims_def);
    if (!dims_tup) return {};
    auto dims = dims_tup.value();

    nat_t dims_size = tuple_size(dims);
    return world.lit_nat(dims_size);
}

const Def* normalize_idx_1DtoND(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto [mem, idx] = arg->projs<2>();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto [implicits, layout_def] = callee_app->uncurry_args<2>();
    auto [n_def, _]              = implicits->projs<2>();

    auto n_lit = Lit::isa(n_def);
    if (!n_lit) return {};
    auto n = n_lit.value();

    auto layout_opt = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    const auto& layout = layout_opt.value();

    nat_t layout_size = tuple_size(layout.dims);
    if (n != layout_size)
        error("size of 1d-index must align with matrix dimensions ({} != {}) layout = {}", n, layout_size, layout_def);

    auto [result_mem, result_idx] = layalg::idx_1DtoND(world, mem, idx, layout);
    return world.tuple({result_mem, result_idx});
}

const Def* normalize_idx_NDto1D(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto nd_idx = arg;

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto [n_def, ms_def, dims_def, strides_def] = callee_app->arg()->projs<4>();

    auto n_lit = Lit::isa(n_def);
    if (!n_lit) return {};
    auto n = n_lit.value();

    auto ms_opt = extract_ms(n, ms_def);
    if (!ms_opt) return {};
    auto& ms = ms_opt.value();

    auto s = Idx::isa_lit(type);
    if (!s) return {};
    auto size = s.value();

    auto add               = world.call(core::wrap::add, world.lit_nat_0());
    auto nat_bitcast       = world.call<core::bitcast>(world.type_idx(size));
    const Def* dot_product = static_cast<const Def*>(world.lit_idx(size, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < ms[i]; ++j) {
            auto stride      = world.app(nat_bitcast, strides_def->proj(i)->proj(j));
            auto idx_bitcast = world.call<core::bitcast>(world.type_idx(size));
            auto idx         = world.app(idx_bitcast, nd_idx->proj(i)->proj(j));
            auto mul         = world.call(core::wrap::mul, world.lit_nat_0());
            auto product     = world.app(mul, {stride, idx});
            dot_product      = world.app(add, {dot_product, product});
        }
    }
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

    auto complement_layout = elevate(complement(layout_tup, size));
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

    layalg::Layout comp = composition(layout1_tup, layout2_tup);
    return make_layout(world, std::move(comp));
}

const Def* normalize_layout_zipped_divide(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto layout_def = callee_app->arg();

    auto layout_opt = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    auto& layout = layout_opt.value();

    auto [num_tilers, tiler_def] = arg->projs<2>();

    auto tiler_n_lit = Lit::isa(num_tilers);
    if (!tiler_n_lit) return {};
    auto tiler_n = tiler_n_lit.value();

    Vector<layalg::Layout> tiler_layouts;
    if (tiler_n == 1) {
        auto tiler_opt = extract_layout_static(tiler_def);
        if (!tiler_opt) return {};
        tiler_layouts.emplace_back(std::move(tiler_opt.value()));
    } else {
        for (size_t i = 0; i < tiler_n; ++i) {
            auto subtiler_def = tiler_def->proj(i);
            auto subtiler_opt = extract_layout_static(subtiler_def);
            if (!subtiler_opt) return {};
            tiler_layouts.emplace_back(std::move(subtiler_opt.value()));
        }
    }

    auto pair          = zipped_divide(layout, tiler_layouts);
    auto& tile_layout  = pair.first;
    auto& outer_layout = pair.second;

    DefVec result;
    result.emplace_back(make_layout(world, std::move(tile_layout)));
    result.emplace_back(make_layout(world, std::move(outer_layout)));
    return world.tuple(result);
}

const Def* normalize_tile(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto [num_tilers, tiler_defs] = arg->projs<2>();

    auto tiler_n_lit = Lit::isa(num_tilers);
    if (!tiler_n_lit) return {};
    auto tiler_n = tiler_n_lit.value();

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto [implicits, mat] = callee_app->uncurry_args<2>();

    auto [layout_def, _] = implicits->projs<2>();
    auto layout_opt      = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    auto& layout = layout_opt.value();

    Vector<layalg::Layout> tilers, tiler_complements;
    for (size_t i = 0; i < tiler_n; ++i) {
        auto tiler_def = tiler_defs->proj(i);
        auto tiler_opt = extract_layout_static(tiler_def);
        if (!tiler_opt) return {};
        auto& tiler = tiler_opt.value();

        auto sublay           = layalg::sublayout(layout, i);
        auto sublay_size      = layalg::size(sublay);
        auto tiler_flat_compl = layalg::complement(tiler, sublay_size);
        auto tiler_complement = layalg::elevate(std::move(tiler_flat_compl));

        tilers.emplace_back(std::move(tiler));
        tiler_complements.emplace_back(std::move(tiler_complement));
    }

    auto pair          = zipped_divide(layout, tilers);
    auto& tile_layout  = pair.first;
    auto& outer_layout = pair.second;

    return {};
}

MIM_matrix_NORMALIZER_IMPL

} // namespace plug::matrix

} // namespace mim
