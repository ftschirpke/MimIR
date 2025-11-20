#include "mim/plug/gpu/gpu.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include "mim/plug/gpu/pass/make_kernels_external.h"

using namespace mim;
using namespace mim::plug;

void reg_stages(Flags2Stages& stages) {
    Stage::hook<gpu::make_kernels_external_pass, gpu::pass::MakeKernelsExternal>(stages);
    printf("Hi there, registering GPU pass\n");
}

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"gpu", nullptr, reg_stages, nullptr}; }
