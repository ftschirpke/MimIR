#pragma once

#include <mim/axm.h>
#include <mim/lam.h>
#include <mim/world.h>

#include <mim/plug/core/core.h>

#include "mim/plug/gpu/autogen.h"

namespace mim::plug::gpu {

/// Removes recusively all occurences of mem from a type (sigma).
inline const Def* strip_mem_ty(const Def* def) {
    auto& world = def->world();

    if (auto sigma = def->isa<Sigma>()) {
        DefVec new_ops;
        for (auto op : sigma->ops())
            if (auto new_op = strip_mem_ty(op); new_op != world.sigma()) new_ops.push_back(new_op);

        return world.sigma(new_ops);
    } else if (Axm::isa<gpu::M>(def)) {
        return world.sigma();
    }

    return def;
}

} // namespace mim::plug::gpu
