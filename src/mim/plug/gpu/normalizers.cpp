#include "mim/world.h"

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu {

const Def* normalize_const(const Def* type, const Def*, const Def* arg) {
    auto& world = type->world();
    return world.lit(world.type_idx(arg), 42);
}

MIM_gpu_NORMALIZER_IMPL

} // namespace mim::plug::gpu
