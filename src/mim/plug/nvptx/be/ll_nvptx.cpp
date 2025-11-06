#include "mim/plug/nvptx/be/ll_nvptx.h"

using namespace std::string_literals;

namespace mim::ll {

namespace nvptx {

class Emitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    Emitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_emitter", ostream) {}
};

void emit(World& world, std::ostream& ostream) {
    Emitter emitter(world, ostream);
    emitter.run();
}

} // namespace nvptx

} // namespace mim::ll
