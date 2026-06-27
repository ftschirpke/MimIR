#pragma once

#include "mim/plug/ll/ll.h"

namespace mim {

class World;

namespace ll_nvptx {

void emit_host(World&, std::ostream&);
void emit_host_with_embedded_device(World&, std::ostream&);
void emit_device(World&, std::ostream&);

} // namespace ll_nvptx

} // namespace mim
