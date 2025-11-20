#pragma once

#include <mim/pass.h>

namespace mim {

namespace plug::gpu::pass {

class MakeKernelsExternal : public RWPass<MakeKernelsExternal, Lam> {
public:
    MakeKernelsExternal(World& world, flags_t annex)
        : RWPass(world, annex) {
        printf("Hi there, constructing GPU pass\n");
    }

    const Def* rewrite(const Def* def) override;

    void enter() override;
};

} // namespace plug::gpu::pass

} // namespace mim
