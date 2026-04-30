#pragma once

#include <cstdint>

#include <unordered_map>

#include <mim/phase.h>

#include "mim/def.h"
#include "mim/rewrite.h"

#include "rust/mimir_eqsat.h"

namespace mim::plug::eqsat {

class SlottedRewrite : public Phase, public Rewriter {
public:
    SlottedRewrite(World& world, std::string name)
        : Phase(world, std::move(name))
        , Rewriter(world.inherit()) {
        register_symbols();
    }
    SlottedRewrite(World& world, flags_t annex)
        : Phase(world, annex)
        , Rewriter(world.inherit()) {
        register_symbols();
    }

    void start() override;

    using Phase::world;
    using Rewriter::world;
    World& world() = delete;
    World& old_world() { return Phase::world(); }
    World& new_world() { return Rewriter::world(); }

private:
    void register_symbols() {
        for (auto [flags, annex] : old_world().flags2annex()) {
            auto new_annex                = new_world().register_annex(flags, rewrite(annex));
            axms_[new_annex->sym().str()] = new_annex;
        }

        aliases_["Univ"] = new_world().univ();
        aliases_["Bool"] = new_world().type_bool();
        aliases_["Nat"]  = new_world().type_nat();
        aliases_["I8"]   = new_world().type_i8();
        aliases_["I16"]  = new_world().type_i16();
        aliases_["I32"]  = new_world().type_i32();
        aliases_["I64"]  = new_world().type_i64();
        aliases_["ff"]   = new_world().lit_ff();
        aliases_["tt"]   = new_world().lit_tt();
        aliases_["i8"]   = new_world().lit_nat(0x100);
        aliases_["i16"]  = new_world().lit_nat(0x10000);
        aliases_["i32"]  = new_world().lit_nat(0x100000000);
    }

    std::pair<rust::Vec<RuleSet>, CostFn> import_config();

    // Performs a top-down traverse of each RecExprFFI
    // and creates and stores all bindings with their definitions.
    // Lambdas are created without their bodies in this phase.
    enum class InitStage {
        Declarations,
        Bindings,
    };
    void init(rust::Vec<RecExprFFI> rewrites, InitStage stage);
    const Def* init(uint32_t id, InitStage stage, bool recurse = false);
    const Def* init_axm(uint32_t id, NodeFFI node);
    const Def* init_root(uint32_t id, NodeFFI node);
    const Def* init_let(uint32_t id, NodeFFI node);
    const Def* init_con(uint32_t id, NodeFFI node);

    // Performs a bottom-up traverse of each RecExprFFI and
    // creates a Def in the new_world() for every node.
    // At this point, the bodies of the lambdas created
    // in the init phase will be set.
    void convert(rust::Vec<RecExprFFI> rewrites);
    const Def* convert(uint32_t id, bool recurse = false);
    const Def* convert_root(uint32_t id, NodeFFI node);
    const Def* convert_let(uint32_t id, NodeFFI node);
    const Def* convert_con(uint32_t id, NodeFFI node);
    const Def* convert_app(uint32_t id, NodeFFI node);
    const Def* convert_var(uint32_t id, NodeFFI node);
    const Def* convert_lit(uint32_t id, NodeFFI node);
    const Def* convert_pack(uint32_t id, NodeFFI node);
    const Def* convert_tuple(uint32_t id, NodeFFI node);
    const Def* convert_extract(uint32_t id, NodeFFI node);
    const Def* convert_insert(uint32_t id, NodeFFI node);
    const Def* convert_inj(uint32_t id, NodeFFI node);
    const Def* convert_merge(uint32_t id, NodeFFI node);
    const Def* convert_match(uint32_t id, NodeFFI node);
    const Def* convert_proxy(uint32_t id, NodeFFI node);
    const Def* convert_join(uint32_t id, NodeFFI node);
    const Def* convert_meet(uint32_t id, NodeFFI node);
    const Def* convert_bot(uint32_t id, NodeFFI node);
    const Def* convert_top(uint32_t id, NodeFFI node);
    const Def* convert_arr(uint32_t id, NodeFFI node);
    const Def* convert_sigma(uint32_t id, NodeFFI node);
    const Def* convert_cn(uint32_t id, NodeFFI node);
    const Def* convert_pi(uint32_t id, NodeFFI node);
    const Def* convert_idx(uint32_t id, NodeFFI node);
    const Def* convert_hole(uint32_t id, NodeFFI node);
    const Def* convert_type(uint32_t id, NodeFFI node);
    const Def* convert_num(uint32_t id, NodeFFI node);
    const Def* convert_symbol(uint32_t id, NodeFFI node);

    // A node that is associated with a Def can be:
    // 1) A node representing an arbitrary term
    // 2) A symbol node representing an annex
    // 3) A symbol node representing a type or term alias
    // 4) A symbol node representing a variable
    const Def* get_def(uint32_t id) {
        auto def = added_[id];
        if (def == nullptr) {
            auto sym = get_symbol(id);
            sym.empty() ? sym = get_slot(id) : sym;

            if (aliases_.contains(sym))
                def = aliases_[sym];
            else if (axms_.contains(sym))
                def = get_axm(sym);
            else if (vars_.contains(sym))
                def = get_var(sym);
        }
        return def;
    }

    void register_var(std::string name, const Def* converted) {
        if (vars_.contains(name)) {
            std::cerr << "register_var: can't define the same var: " << name << " twice\n"
                      << "existing def: " << vars_[name] << "\n";
            assert(false);
        }
        std::cout << "registering " << name << " in scope: (" << curr_loc_.first << curr_loc_.second << ")\n";
        vars_[name] = converted;
    }
    void register_axm(std::string name, const Axm* converted) {
        if (axms_.contains(name)) {
            std::cerr << "register_axm: can't define the same axiom: " << name << " twice\n"
                      << "existing def: " << axms_[name] << "\n";
            assert(false);
        }
        axms_[name] = converted;
    }

    const Def* get_var(std::string name) { return vars_[name]; }
    const Def* get_axm(std::string name) { return axms_[name]; }

    NodeFFI get_node(MimKind expected, uint32_t id) {
        assert(res_[id].kind == expected && "get_node: mismatch between expected and actual node kind");
        return res_[id];
    }
    NodeFFI get_node_unsafe(uint32_t id) { return res_[id]; }
    std::string get_symbol(uint32_t id) { return res_[id].symbol.c_str(); }
    uint64_t get_num(uint32_t id) { return res_[id].num; }
    std::string get_slot(uint32_t id) { return res_[id].slot.c_str(); }
    std::vector<uint32_t> get_cons_flat(uint32_t id) {
        std::vector<uint32_t> flattened;
        auto curr_cons = get_node_unsafe(id);
        while (curr_cons.kind != MimKind::Nil) {
            flattened.push_back(curr_cons.children[0]);
            curr_cons = get_node_unsafe(curr_cons.children[1]);
        }
        return flattened;
    }

    // Tracks the current location in the scope tree.
    // This is done by maintaining a depth and an offset while
    // traversing the RecExprFFI that indicate the exact scope we
    // are currently in. To visualize this:
    //                s1
    //              /    \
    //             s2    s3
    //            /     /  \
    //           s4    s5  s6
    // The location of scope s5 would be at (2, 1) because it is at
    // at a tree depth of 2 and at an offset of 1 at that depth.
    typedef std::pair<size_t, size_t> Loc;
    Loc curr_loc_;

    struct Scope {
        Scope* parent;
        std::string var_name;
        const Def* def;
    };

    // The current scope which we mostly use to construct the scope map during init
    Scope* curr_scope_;

    // For every scope-location we store a Scope struct that stores a pointer to its
    // parent scope, the name of the var it introduces, and the Def associated with this var.
    std::unordered_map<Loc, Scope> scopes_;

    // There is a special root scope that we access with curr_scope_ = 0
    // which is a registry of all top-level/closed Defs that exist beyond
    // the current RecExprFFI.
    std::set<std::pair<std::string, const Def*>> root_scope_;

    rust::Vec<NodeFFI> res_;
    std::unordered_map<uint32_t, const Def*> added_;
    std::unordered_map<std::string, const Def*> vars_;
    std::unordered_map<std::string, const Def*> axms_;
    std::unordered_map<std::string, const Def*> aliases_;
};

}; // namespace mim::plug::eqsat
