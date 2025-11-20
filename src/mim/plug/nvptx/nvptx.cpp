#include "mim/plug/nvptx/nvptx.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include "mim/plug/nvptx/be/ll_nvptx.h"

using namespace mim;

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"nvptx", nullptr, nullptr, [](Backends& backends) {
                backends["ll-host-nvptx"] = &ll::nvptx::emit_host;
                backends["ll-dev-nvptx"]  = &ll::nvptx::emit_device;
            }};
}
