#pragma once

#include "mim/phase.h"

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

class MemChecks : public Analysis {
public:
    using Super = Analysis;

    MemChecks(World& world, flags_t annex)
        : Analysis(world, annex) {}

private:
    const Def* rewrite(const Def*) final;
    const Def* rewrite_mut_Lam(Lam*) final;

    DefSet analyzed_;
    LamSet kernels_;
};

} // namespace mim::plug::gpu::phase
