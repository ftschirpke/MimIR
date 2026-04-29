#include <mim/plug/eqsat/eqsat.h>
#include <mim/plug/eqsat/phase/egg_rewrite.h>
#include <mim/plug/eqsat/phase/eqsat_phase.h>
#include <mim/plug/eqsat/phase/slotted_rewrite.h>

namespace mim::plug::eqsat {

void EqsatPhase::start() {
    bool slotted = false;

    // Infers whether to use 'egg' or 'slotted-egraphs' based on a
    // config function with the signature '[] -> %eqsat.Impl'
    // Each rewrite phase will further infer config values from
    // config functions and internalize all of them, including this one.
    for (auto def : world().externals().mutate()) {
        if (auto lam = def->isa<Lam>()) {
            if (Axm::isa<eqsat::Impl>(lam->ret_dom())) {
                auto body = lam->as<Lam>()->body();
                if (auto body_app = body->isa<App>()) {
                    if (Axm::isa<eqsat::slotted>(body_app->arg()))
                        slotted = true;
                    else if (Axm::isa<eqsat::AstDepth>(body_app->arg()))
                        slotted = false;
                }
            }
        }
    }

    if (slotted) {
        SlottedRewrite slotted_rewrite(world(), "slotted_rewrite");
        slotted_rewrite.start();
    } else {
        EggRewrite egg_rewrite(world(), "egg_rewrite");
        egg_rewrite.start();
    }
}

}; // namespace mim::plug::eqsat
