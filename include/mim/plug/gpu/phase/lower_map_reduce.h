#pragma once

#include <functional>

#include <mim/phase.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

class LowerMapReduce : public RWPhase {
public:
    using Super = RWPhase;

    LowerMapReduce(World& world, flags_t annex)
        : Super(world, annex) {}

private:
    /// What to build once a (sub-)call's kernel launch has produced its device output: given the
    /// post-launch `[mem, GlobalM, ConstM]` and the device pointer to that call's result, returns
    /// the Def the launch's continuation should reduce to.
    using DeviceCont = std::function<const Def*(const Def* mem, const Def* global, const Def* const_tok,
                                                 const Def* out_dptr)>;

    /// Skips the whole phase if the program already contains an explicit `%gpu.init`.
    // TODO: consider different solution to %gpu.init vs %gpu.auto_init problem
    void start() final;
    const Def* rewrite_imm_App(const App*) final;

    const Def* lower_map_reduce_post(const App*);
    std::optional<const Def*> lower_map_reduce_chained(const App*, const Def* mem, const Def* global,
                                                        const Def* const_tok, DefVec& to_free, const DeviceCont&,
                                                        const DefVec* resolved_is       = nullptr,
                                                        const DefVec* resolved_post_is = nullptr);

    /// Use-count of every Def reachable from the program's roots, computed once in start(): tells
    /// `lower_map_reduce_chained` whether an input's producer is consumed only by this one call (so
    /// its result can stay device-resident) or also needed elsewhere (so it must go through the
    /// usual host round-trip instead).
    DefMap<nat_t> use_count_;
};

} // namespace mim::plug::gpu::phase
