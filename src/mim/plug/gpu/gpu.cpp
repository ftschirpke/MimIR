#include "mim/plug/gpu/gpu.h"

#include <mim/pass.h>
#include <mim/plugin.h>

using namespace mim;
using namespace mim::plug;

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"gpu", gpu::register_normalizers, nullptr, nullptr}; }
