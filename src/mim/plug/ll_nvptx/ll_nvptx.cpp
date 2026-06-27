#include "mim/plug/ll_nvptx/phase/ll_nvptx.h"

#include <mim/pass.h>
#include <mim/plugin.h>

#include <mim/plug/gpu/gpu.h>

#include "mim/plug/ll_nvptx/ll_nvptx.h"

using namespace std::string_literals;

using namespace mim;
using namespace mim::plug;

void reg_stages(Flags2Stages& stages) {
    MIM_REPL(stages, nvptx::stream_impl_repl, {
        auto stream_flags = Annex::base<gpu::Stream>();
        if (def->flags() == stream_flags) return world().annex<nvptx::Stream>();
        return {};
    });
}

class Emit : public Phase {
public:
    Emit(World& world, flags_t annex)
        : Phase(world, annex) {}

    void start() override {
        auto name = world().name() ? std::string(world().name().view()) : "a"s;
        auto ofs  = std::ofstream(name + ".ll"s);
        ll_nvptx::emit_host_with_embedded_device(world(), ofs);
    }
};

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"ll_nvptx", MIM_VERSION, nullptr, reg_stages}; }
