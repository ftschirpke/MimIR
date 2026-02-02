#include "mim/plug/gpu/phase/rewrite_kernel_args.h"

#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

const Def* RewriteKernelArgs::rewrite_mut_Lam(Lam* lam) {
    if (!lam->is_external()) return RWPhase::rewrite_mut_Lam(lam);

    ILOG("FRIEDRICH RewriteKernelArgs thinks '{}' is a kernel", lam);
    return RWPhase::rewrite_mut_Lam(lam);

    // TODO: implement
}

} // namespace mim::plug::gpu::phase
