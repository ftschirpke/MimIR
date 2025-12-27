#pragma once

#include "mim/plug/core/be/ll.h"

namespace mim {

class World;

namespace ll::nvptx {

void emit_host(World&, std::ostream&);
void emit_device(World&, std::ostream&);

} // namespace ll::nvptx

} // namespace mim
