#include <mim/normalize.h>

#include <mim/plug/matrix/layout_algebra.h>

#include "mim/world.h"

#include "mim/plug/core/core.h"
#include "mim/plug/matrix/matrix.h"
#include "mim/plug/mem/mem.h"

namespace mim {

namespace {

using namespace plug::matrix::layalg;

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

//
// Layout Parsing
//

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
    auto layout_def = callee_app->arg();

    auto layout_opt = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    const auto& layout = layout_opt.value();

    auto s = Idx::isa_lit(type);
    if (!s) return {};
    auto size = s.value();

    return layalg::idx_NDto1D(world, size, layout.strides, nd_idx);
}

const Def* normalize_tiling_layouts(const Def* type, const Def* callee, const Def* arg) {
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

    auto tiler_def = arg;

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto [implicits, mat]         = callee_app->uncurry_args<2>();
    auto [num_tilers, tiler_defs] = tiler_def->projs<2>();

    auto tiler_n_lit = Lit::isa(num_tilers);
    if (!tiler_n_lit) return {};
    auto tiler_n = tiler_n_lit.value();

    auto [layout_def, T] = implicits->projs<2>();
    auto layout_opt      = extract_layout_static(layout_def);
    if (!layout_opt) return {};
    auto& layout = layout_opt.value();

    auto layout_size = layalg::size(layout);

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

    auto tile_lay_def  = make_layout(world, layalg::Layout(tile_layout));
    auto outer_lay_def = make_layout(world, layalg::Layout(outer_layout));

    auto inner_pi_impl  = world.mut_pi(world.type(), true)->set_dom(world.type_nat());
    auto inner_pi_mem_t = world.call<mem::M>(inner_pi_impl->var());
    auto inner_idx_t    = world.call<matrix::MatIdx>(tile_lay_def);
    inner_pi_impl->set_codom(world.pi({inner_pi_mem_t, inner_idx_t}, {inner_pi_mem_t, T}));

    auto inner_lam_impl         = world.mut_lam(inner_pi_impl);
    auto inner_lam_mem_t        = world.call<mem::M>(inner_lam_impl->var());
    auto inner_pi               = world.pi({inner_lam_mem_t, inner_idx_t}, {inner_lam_mem_t, T});
    auto inner_lam              = world.mut_lam(inner_pi);
    auto [inner_mem, inner_idx] = inner_lam->vars<2>();

    auto outer_pi_impl  = world.mut_pi(world.type(), true)->set_dom(world.type_nat());
    auto outer_pi_mem_t = world.call<mem::M>(outer_pi_impl->var());
    auto outer_idx_t    = world.call<matrix::MatIdx>(outer_lay_def);
    outer_pi_impl->set_codom(world.pi({outer_pi_mem_t, outer_idx_t}, {outer_pi_mem_t, inner_pi_impl}));

    auto outer_lam_impl         = world.mut_lam(outer_pi_impl);
    auto outer_lam_mem_t        = world.call<mem::M>(outer_lam_impl->var());
    auto outer_pi               = world.pi({outer_lam_mem_t, outer_idx_t}, {outer_lam_mem_t, inner_pi_impl});
    auto outer_lam              = world.mut_lam(outer_pi);
    auto [outer_mem, outer_idx] = outer_lam->vars<2>();

    DefVec layout_idx_defs;
    auto add = world.call(core::wrap::add, world.lit_nat_0());
    for (size_t i = 0; i < tiler_n; ++i) {
        auto sublay          = sublayout(layout, i);
        auto& subtiler       = tilers[i];
        auto& subtiler_compl = tiler_complements[i];

        auto inner_idx_1d = layalg::idx_NDto1D(world, layout_size, subtiler.strides, inner_idx->proj(i));
        auto outer_idx_1d = layalg::idx_NDto1D(world, layout_size, subtiler_compl.strides, outer_idx->proj(i));

        sublay.strides   = layalg::tuple_prefix_product(sublay.dims);
        auto idx_1d      = world.app(add, {inner_idx_1d, outer_idx_1d});
        auto toND_result = layalg::idx_1DtoND(world, inner_mem, idx_1d, sublay);
        inner_mem        = toND_result.first;
        layout_idx_defs.push_back(toND_result.second);
    }

    auto nd_idx   = world.tuple(layout_idx_defs);
    auto mat_call = world.app(mat, inner_lam_impl->var());
    auto call     = world.app(mat_call, world.tuple({inner_mem, nd_idx}));

    inner_lam->set(true, call);
    inner_lam_impl->set(true, inner_lam);
    outer_lam->set(true, world.tuple({outer_mem, inner_lam_impl}));
    outer_lam_impl->set(true, outer_lam);
    return outer_lam_impl;
}

MIM_matrix_NORMALIZER_IMPL

} // namespace plug::matrix

} // namespace mim
