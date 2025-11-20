#include "mim/plug/gpu/pass/make_kernels_external.h"

namespace mim::plug::gpu::pass {

const Def* MakeKernelsExternal::rewrite(const Def* def) {
    ILOG("Rewriting {}", def);
    printf("Hi there, running GPU pass\n");

    if (auto lam = def->isa<Lam>()) {
        curr_mut()->make_external();
        return curr_mut();
    }

    return def;
}

void MakeKernelsExternal::enter() {
    ILOG("Hi there, entering GPU pass\n");
    printf("Hi there, entering GPU pass\n");
}

} // namespace mim::plug::gpu::pass
