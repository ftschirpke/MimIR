#include "mim/plug/nvptx/nvptx.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include "mim/plug/nvptx/be/ll_nvptx.h"

using namespace mim;

extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"nvptx", nullptr, nullptr, [](Backends& backends) {
                backends["ll-host-nvptx"]       = &ll::nvptx::emit_host;
                backends["ll-dev-nvptx"]        = &ll::nvptx::emit_device;
                backends["ll-host-nvptx-embed"] = &ll::nvptx::emit_host_with_embedded_device;
            }};
}
