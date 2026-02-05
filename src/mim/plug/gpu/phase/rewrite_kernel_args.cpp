#include "mim/plug/gpu/phase/rewrite_kernel_args.h"

#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

const Def* RewriteKernelArgs::rewrite_mut_Lam(Lam* lam) {
    if (!lam->is_external()) return RWPhase::rewrite_mut_Lam(lam);

    ILOG("FRIEDRICH RewriteKernelArgs thinks '{}' is a kernel", lam);
    return RWPhase::rewrite_mut_Lam(lam);

    // TODO: implement

    // for (size_t i = 0; i < lam->num_vars(); ++i)
    //     ILOG("FRIEDRICH RewriteKernelArgs var {} - {} : {}", i, lam->var(i), lam->var(i)->type());

    // auto m1 = lam->var(0);
    // auto m3 = lam->var(1);
    // auto m4 = lam->var(2);
    // auto m5 = lam->var(3);
    // assert(Axm::isa<mem::M>(m1->type()));
    // assert(Axm::isa<mem::M>(m3->type()));
    // assert(Axm::isa<mem::M>(m4->type()));
    // assert(Axm::isa<mem::M>(m5->type()));

    // ILOG("FRIEDRICH found all mem::M");

    // auto group_idx = lam->var(4);
    // auto item_idx  = lam->var(5);

    // ILOG("FRIEDRICH indices are {} : {} and {} : {}", group_idx, group_idx->type(), item_idx, item_idx->type());

    // const Def* smem_ptr;
    // if (lam->num_vars() == 8) {
    //     smem_ptr = nullptr;
    //     ILOG("FRIEDRICH smem is null");
    // } else if (lam->num_vars() == 9) {
    //     smem_ptr = lam->var(6);
    //     ILOG("FRIEDRICH smem is {} : {}", smem_ptr, smem_ptr->type());
    // } else
    //     error("Unexpected number of arguments for kernel '{}'", lam);

    // ILOG("FRIEDRICH defined smem");

    // auto var_arg = lam->var(lam->num_vars() - 2);
    // auto cn      = lam->var(lam->num_vars() - 1);

    // ILOG("FRIEDRICH final args are {} : {} and {} : {}", var_arg, var_arg->type(), cn, cn->type());

    // World& w = lam->world();

    // ILOG("FRIEDRICH got world");

    // Defs new_dom = {m1->type(), m3->type(), m4->type(), m5->type(), var_arg->type(), cn->type()};
    // ILOG("FRIEDRICH created new dom {}", new_dom);

    // auto new_con = w.mut_con(new_dom);
    // ILOG("FRIEDRICH created new con {} : {}", new_con, new_con->type());

    // auto ret = rewrite_stub(lam, new_con);
    // ILOG("FRIEDRICH created new stub {}", ret);

    // return ret;
}

} // namespace mim::plug::gpu::phase
