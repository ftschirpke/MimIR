#include <mim/driver.h>
#include <mim/plugin.h>

// pCUDA plugin - provides additional GPU operations for backend management
// The actual operation definitions are in gpu_pcuda.mim

extern "C" MIM_EXPORT mim::Plugin mim_get_plugin() {
    return {"gpu_pcuda", nullptr, nullptr, nullptr};
}
