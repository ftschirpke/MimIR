#include "mim/world.h"

#include "mim/plug/nvptx/nvptx.h"

namespace mim::plug::nvptx {

const Def* normalize_const(const Def* type, const Def*, const Def* arg) {
    auto& world = type->world();
    return world.lit(world.type_idx(arg), 42);
}

MIM_nvptx_NORMALIZER_IMPL

} // namespace mim::plug::nvptx
