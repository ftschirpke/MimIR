#pragma once

#include <mim/phase.h>

namespace mim::plug::nvptx::phase {

class CudaSetupCleanup : public RWPhase {
public:
    CudaSetupCleanup(World& world, flags_t annex)
        : RWPhase(world, annex) {}

    const Def* rewrite(const Def*) override;
    void rewrite_external(Def*) override;
};

} // namespace mim::plug::nvptx::phase
