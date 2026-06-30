#include "mim/plug/nvptx/phase/mem_checks.h"

#include <mim/axm.h>

#include <mim/plug/mem/mem.h>

namespace mim::plug::nvptx::phase {

class MemFinder {
public:
    MemFinder(const Def* to_search)
        : to_search({to_search}) {}
    MemFinder(DefVec&& to_search)
        : to_search(std::move(to_search)) {}

    using IsaMem = Axm::IsA<mem::M, mim::App>;

    IsaMem next_mem();

private:
    DefVec to_search;
};

MemFinder::IsaMem MemFinder::next_mem() {
    while (!to_search.empty()) {
        auto cur = to_search.back();
        to_search.pop_back();
        if (cur->isa<Sigma>())
            for (auto op : cur->ops())
                to_search.push_back(op);
        else if (auto mem = Axm::isa<mem::M>(cur))
            return mem;
    }
    return IsaMem();
}

const Def* MemChecks::rewrite_imm_App(const App* app) {
    if (auto symbols = Axm::isa<nvptx::symbols>(app)) {
        auto [_, __, ___, global_syms, const_syms] = symbols->args<5>();

        MemFinder global_mem_finder(global_syms);
        if (global_mem_finder.next_mem()) {
            error("You may not pass any %mem.M across device boundaries: creating symbol(s) {} in global address space "
                  "with {}",
                  global_syms, app);
        }
        MemFinder const_mem_finder(const_syms);
        if (const_mem_finder.next_mem()) {
            error("You may not pass any %mem.M across device boundaries: creating symbol(s) {} in constant address "
                  "space with {}",
                  const_syms, app);
        }
    }
    return Super::rewrite_imm_App(app);
}

} // namespace mim::plug::nvptx::phase
