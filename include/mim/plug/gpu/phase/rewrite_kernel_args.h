#pragma once

#include "mim/phase.h"

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

class RewriteKernelArgs : public RWPhase {
public:
    RewriteKernelArgs(World& world, flags_t annex)
        : RWPhase(world, annex) {}

private:
    const Def* rewrite_mut_Lam(Lam*) final;
};

} // namespace mim::plug::gpu::phase
