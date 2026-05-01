#include <mim/normalize.h>

#include "mim/world.h"

#include "mim/plug/core/core.h"
#include "mim/plug/matrix/matrix.h"

namespace mim::plug::matrix {

const Def* normalize_idx_NDto1D(const Def* type, const Def* callee, const Def* arg) {
    auto& world = type->world();

    auto [n, ms, strides] = arg->projs<3>();

    world.ELOG("FRIEDRICH: trying to normalize idx_NDto1D...");

    auto callee_app = callee->isa<App>();
    assert(callee_app);
    auto [dims, nd_idx] = callee_app->uncurry_args<2>();

    auto n_lit = Lit::isa(n);
    if (!n_lit.has_value()) return {};
    auto num_ranks = n_lit.value();

    auto s = Idx::isa_lit(type);
    if (!s.has_value()) return {};
    auto size = s.value();

    auto add               = world.call(core::wrap::add, world.lit_nat_0());
    auto nat_bitcast       = world.call<core::bitcast>(world.type_idx(size));
    const Def* dot_product = static_cast<const Def*>(world.lit_idx(size, 0));
    for (size_t i = 0; i < num_ranks; ++i) {
        auto m = Lit::isa(ms->proj(i));
        if (!m.has_value()) return {};
        auto num_strides_for_rank = m.value();
        for (size_t j = 0; j < num_strides_for_rank; ++j) {
            auto stride      = world.app(nat_bitcast, strides->proj(i)->proj(j));
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

const Def* normalize_tiling_dims(const Def* type, const Def* callee, const Def* arg) { return {}; }

const Def* normalize_tile(const Def* type, const Def* callee, const Def* arg) { return {}; }

MIM_matrix_NORMALIZER_IMPL

} // namespace mim::plug::matrix
