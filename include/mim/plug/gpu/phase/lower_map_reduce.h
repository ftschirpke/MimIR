#pragma once

#include <mim/nest.h>
#include <mim/phase.h>
#include <mim/schedule.h>

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

class LowerMapReduce : public RWPhase {
public:
    using Super = RWPhase;

    LowerMapReduce(World& world, flags_t annex)
        : Super(world, annex) {}

private:
    /// How a `%btensor.map_reduce_post` call's result is consumed, as seen from `old_world()`.
    enum class Classification {
        HostVisible, ///< Some real consumer isn't another lowered call here; materialize a host buffer as today.
        DeviceOnly, ///< Its single real consumer is another `map_reduce_post` call lowered here; stays device-resident.
        Dead,       ///< No real consumer at all.
    };

    struct CallInfo {
        Classification cls         = Classification::HostVisible;
        const App* single_consumer = nullptr; ///< Set iff `cls == DeviceOnly`.
        /// A second `HostVisible` call, data-independent of and provably co-resident with another one in
        /// the same straight-line block (same `Scheduler::early` node, no loop/recursion), gets its own
        /// `%gpu.Stream` instead of sharing `%gpu.default_stream` — letting the two launches overlap.
        bool own_stream = false;
    };

    /// Skips the whole phase if the program already contains an explicit `%gpu.init`.
    // TODO: consider different solution to %gpu.init vs %gpu.auto_init problem
    void start() final;
    const Def* rewrite_imm_App(const App*) final;

    const Def* lower_map_reduce_post(const App*);
    void classify_map_reduce_calls();
    const CallInfo& call_info(const App*) const;

    std::unique_ptr<Nest> nest_;
    std::unique_ptr<Scheduler> sched_;
    DefMap<CallInfo> call_info_;
};

} // namespace mim::plug::gpu::phase
