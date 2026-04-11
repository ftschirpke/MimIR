#include "mim/plug/matrix/matrix.h"

#include <mim/pass.h>
#include <mim/plugin.h>

using namespace mim;
using namespace mim::plug;

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"matrix", nullptr, nullptr, nullptr}; }
