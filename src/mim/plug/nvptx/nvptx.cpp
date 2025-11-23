#include "mim/plug/nvptx/nvptx.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include "mim/plug/nvptx/be/ll_nvptx.h"
#include "mim/plug/nvptx/phase/cuda_setup_cleanup.h"

using namespace mim;
using namespace mim::plug;

void reg_stages(Flags2Stages& stages) {
    Stage::hook<nvptx::cuda_setup_cleanup, nvptx::phase::CudaSetupCleanup>(stages);
}

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"nvptx", nullptr, reg_stages, [](Backends& backends) {
                backends["ll-host-nvptx"] = &ll::nvptx::emit_host;
                backends["ll-dev-nvptx"]  = &ll::nvptx::emit_device;
            }};
}
