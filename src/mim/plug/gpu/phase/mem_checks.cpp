#include "mim/plug/gpu/phase/mem_checks.h"

#include <mim/axm.h>

#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

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

const Def* MemChecks::rewrite(const Def* def) {
    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel        = launch->decurry()->arg();
        auto kernel_args   = launch->arg();
        auto kernel_args_t = kernel_args->type();

        MemFinder mem_finder(kernel_args_t);
        if (mem_finder.next_mem()) {
            error("You may not pass any %mem.M across device boundaries: passing {} : {} from host to kernel '{}'",
                  kernel_args, kernel_args_t, kernel);
        }
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        auto [_, global_syms, const_syms] = init->args<3>();

        MemFinder global_mem_finder(global_syms);
        if (global_mem_finder.next_mem()) {
            error("You may not pass any %mem.M across device boundaries: creating symbol(s) {} in global address space "
                  "with {}",
                  global_syms, def);
        }
        MemFinder const_mem_finder(const_syms);
        if (const_mem_finder.next_mem()) {
            error("You may not pass any %mem.M across device boundaries: creating symbol(s) {} in constant address "
                  "space with {}",
                  const_syms, def);
        }
    }
    return Super::rewrite(def);
}

const Def* MemChecks::rewrite_mut_Lam(Lam* lam) {
    auto name = lam->sym().str();
    if (name == "main") {
        MemFinder intype_mem_finder(lam->type()->dom());
        while (auto mem = intype_mem_finder.next_mem()) {
            auto addr_space = mem->arg();
            if (Lit::as(addr_space) != 0) error("The main function may not take %mem.M n with n != 0 as an argument");
        }

        MemFinder outtype_mem_finder(lam->type()->ret_dom());
        while (auto mem = outtype_mem_finder.next_mem()) {
            auto addr_space = mem->arg();
            if (Lit::as(addr_space) != 0) error("The main function may not return any %mem.M n with n != 0");
        }
    }
    return Super::rewrite_mut_Lam(lam);
}

} // namespace mim::plug::gpu::phase
