#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class Lower : public RWPhase {
public:
    Lower(World& world, flags_t annex)
        : RWPhase(world, annex) {}

private:
    const Def* rewrite_imm_App(const App*) final;

    const Def* lower_get(const App*);
    const Def* lower_set(const App*);
    const Def* lower_broadcast(const App*);
    const Def* lower_broadcast_in_dim(const App* app);
    const Def* lower_map_reduce(const App*);

    const Def* rec_broadcast(const Def* s_in, const Def* s_out, const Def* input, u64 r, u64 i);
};

} // namespace mim::plug::tensor::phase
