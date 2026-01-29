#include "mim/plug/gpu/gpu.h"

#include <mim/config.h>
#include <mim/pass.h>
#include <mim/phase.h>
#include <mim/plugin.h>

#include <mim/plug/mem/mem.h>

using namespace mim;
using namespace mim::plug;

// TODO: decide whether gpu::alloc -> mem::alloc or mem::malloc -> gpu::alloc makes more sense
void reg_stages(Flags2Stages& stages) {
    MIM_REPL(stages, gpu::malloc2gpualloc_repl, {
        auto global_as = Lit::as(world().annex<gpu::addr_space_global>());
        if (auto malloc = Axm::isa<mem::malloc>(def)) {
            auto [type, addr_space] = malloc->decurry()->args<2>();
            if (Lit::as(addr_space) == global_as) {
                auto [mem, _] = malloc->args<2>();
                World& w      = type->world();
                return w.app(w.app(w.annex<gpu::alloc>(gpu::alloc::block), type), mem);
            }
        } else if (auto free = Axm::isa<mem::free>(def)) {
            auto [type, addr_space] = free->decurry()->args<2>();
            if (Lit::as(addr_space) == global_as) {
                auto [mem, ptr] = free->args<2>();
                World& w        = type->world();
                return w.app(w.app(w.annex<gpu::free>(gpu::free::block), type), {mem, ptr});
            }
        }
        return {};
    });
}

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"gpu", nullptr, reg_stages, nullptr}; }
