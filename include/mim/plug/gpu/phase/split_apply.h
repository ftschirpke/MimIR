#pragma once

#include <mim/plug/gpu/phase/split_off_kernels.h>

#include "mim/phase.h"

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

class SplitApply : public RWPhase {
public:
    SplitApply(World& world, flags_t annex)
        : RWPhase(world, annex)
        , split_phase(world, "splitoff_kernels_in_setup4backend") {}

    void start() final;

private:
    SplitOffKernels split_phase;
};

} // namespace mim::plug::gpu::phase
