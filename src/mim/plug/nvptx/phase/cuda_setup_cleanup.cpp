#include "mim/plug/nvptx/phase/cuda_setup_cleanup.h"

namespace mim::plug::nvptx::phase {

const Def* CudaSetupCleanup::rewrite(const Def* def) { return Rewriter::rewrite(def); }

void CudaSetupCleanup::rewrite_external(Def* def) {
    auto lam     = Lam::isa_cn(def);
    auto new_def = rewrite(def)->as_mut();
    if (def->is_external() && !new_def->is_external()) new_def->externalize();
    if (!lam) return;
    ILOG("FRIEDRICH checking {}", def);
    if (lam->sym().str() != "main") return;
    ILOG("FRIEDRICH rewriting {}", def);

    assert(lam->is_set());
    ILOG("FRIEDRICH body {}", lam->body());
    ILOG("FRIEDRICH filter {}", lam->filter());

    // TODO: use this phase to implement the setup and cleanup phases
}

} // namespace mim::plug::nvptx::phase
