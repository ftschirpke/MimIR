#pragma once

#include "mim/plug/core/be/ll.h"

namespace mim {

class World;

namespace ll::pcuda {

void emit_host(World&, std::ostream&);
void emit_device(World&, std::ostream&);
void emit_host_with_embedded_device(World&, std::ostream&);

} // namespace ll::pcuda
} // namespace mim
