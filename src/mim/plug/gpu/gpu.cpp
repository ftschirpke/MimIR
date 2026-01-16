#include "mim/plug/gpu/gpu.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include <mim/plug/gpu/phase/split_off_kernels.h>

using namespace mim;
using namespace mim::plug;

void reg_stages(Flags2Stages& stages) {
    Stage::hook<gpu::split_off_kernels_phase, gpu::phase::SplitOffKernels>(stages);
}

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"gpu", nullptr, reg_stages, nullptr}; }
