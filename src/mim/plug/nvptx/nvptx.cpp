#include "mim/plug/nvptx/nvptx.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include <mim/plug/gpu/gpu.h>

#include "mim/plug/nvptx/be/ll_nvptx.h"

using namespace mim;
using namespace mim::plug;

void reg_stages(Flags2Stages& stages) {
    MIM_REPL(stages, nvptx::stream_impl_repl, {
        auto stream_flags = Annex::base<gpu::Stream>();
        if (def->flags() == stream_flags) return world().annex<nvptx::Stream>();
        return {};
    });
}

extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"nvptx", nullptr, reg_stages, [](Backends& backends) {
                backends["ll-host-nvptx"]           = &ll::nvptx::emit_host;
                backends["ll-dev-nvptx"]            = &ll::nvptx::emit_device;
                backends["ll-host-nvptx-embed-dev"] = &ll::nvptx::emit_host_with_embedded_device;
            }};
}
