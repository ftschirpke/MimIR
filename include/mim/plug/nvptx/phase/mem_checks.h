#pragma once

#include "mim/phase.h"

#include "mim/plug/nvptx/nvptx.h"

namespace mim::plug::nvptx::phase {

class MemChecks : public Analysis {
public:
    using Super = Analysis;

    MemChecks(World& world, flags_t annex)
        : Analysis(world, annex) {}

private:
    const Def* rewrite_imm_App(const App*) final;
};

} // namespace mim::plug::nvptx::phase
