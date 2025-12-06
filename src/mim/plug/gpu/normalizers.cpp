#include "mim/world.h"

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu {

const Def* normalize_store(const Def*, const Def*, const Def* arg) {
    auto [mem, ptr, val] = arg->projs<3>();

    if (ptr->isa<Bot>() || val->isa<Bot>()) return mem;
    if (auto pack = val->isa<Pack>(); pack && pack->body()->isa<Bot>()) return mem;
    if (auto tuple = val->isa<Tuple>()) {
        if (std::ranges::all_of(tuple->ops(), [](const Def* op) { return op->isa<Bot>(); })) return mem;
    }

    return {};
}

MIM_gpu_NORMALIZER_IMPL

} // namespace mim::plug::gpu
