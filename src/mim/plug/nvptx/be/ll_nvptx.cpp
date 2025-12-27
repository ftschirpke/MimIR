#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <mim/plug/nvptx/nvptx.h>

namespace mim::ll::nvptx {

class HostEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_host_emitter", ostream) {}
};

class DeviceEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_device_emitter", ostream) {}
};

void emit_host(World& world, std::ostream& ostream) {
    HostEmitter emitter(world, ostream);
    emitter.run();
}

void emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();
}

} // namespace mim::ll::nvptx
