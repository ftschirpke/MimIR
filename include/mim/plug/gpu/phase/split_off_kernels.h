#pragma once

#include "mim/phase.h"

#include "mim/plug/gpu/gpu.h"

namespace mim::plug::gpu::phase {

class SplitOffKernels : public RWPhase {
public:
    SplitOffKernels(World& world, flags_t annex)
        : RWPhase(world, annex) {}
    SplitOffKernels(World& world, std::string name)
        : RWPhase(world, std::move(name)) {}

private:
    void start() final;
    bool analyze() final;
    void analyze(const Def*);

    const Def* rewrite_mut_Lam(Lam*) final;

    DefSet analyzed_;
    LamSet kernels_;
};

inline void SplitOffKernels::start() {
    analyze();

    for (const auto& [f, def] : old_world().flags2annex())
        rewrite_annex(f, def);

    for (auto kernel : kernels_)
        rewrite(kernel);
}

inline bool SplitOffKernels::analyze() {
    for (auto def : old_world().annexes())
        analyze(def);
    for (auto def : old_world().externals().muts())
        analyze(def);

    return false; // no fixed-point neccessary
}

inline void SplitOffKernels::analyze(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel = launch->decurry()->arg();
        if (auto lam = kernel->isa_mut<Lam>()) kernels_.emplace(lam);
    }

    for (auto d : def->deps())
        analyze(d);
}

inline const Def* SplitOffKernels::rewrite_mut_Lam(Lam* old_lam) {
    auto new_def = RWPhase::rewrite_mut_Lam(old_lam);

    if (kernels_.contains(old_lam)) {
        auto new_lam = new_def->as_mut<Lam>();
        new_lam->externalize();
        old_lam->unset();
    }

    return new_def;
}

} // namespace mim::plug::gpu::phase
