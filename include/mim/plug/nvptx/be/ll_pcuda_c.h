#pragma once

#include <string>

namespace mim {
class World;
}

namespace mim::ll::pcuda {

/// C-based pCUDA code generator for AdaptiveCpp compilation
/// Generates portable C code that uses pCUDA runtime and can be
/// compiled by AdaptiveCpp's acpp compiler for multiple GPU backends

void emit_host_c(World&, std::ostream&);
void emit_device_c(World&, std::ostream&);

} // namespace mim::ll::pcuda
