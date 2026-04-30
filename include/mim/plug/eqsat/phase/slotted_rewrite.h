#pragma once

#include <cstdint>

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
    // 5) A symbol node representing a lambda
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
            else if (lams_.contains(sym))
                def = get_lam(sym);
        }
        return def;
    }

    void register_var(std::string name, const Def* converted) {
        if (vars_.contains(name)) {
            std::cerr << "register_var: can't define the same var: " << name << " twice\n"
                      << "existing def: " << vars_[name] << "\n";
            assert(false);
        }
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
    void register_lam(std::string name, const Lam* converted) {
        if (lams_.contains(name)) {
            std::cerr << "register_lam: can't define the same lambda: " << name << " twice\n"
                      << "existing def: " << lams_[name] << "\n";
            assert(false);
        }
        lams_[name] = converted;
    }

    const Def* get_var(std::string name) { return vars_[name]; }
    const Def* get_axm(std::string name) { return axms_[name]; }
    const Lam* get_lam(std::string name) { return lams_[name]; }

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

    std::string remove_uid(std::string name) {
        if (auto pos = name.rfind("_"); pos != std::string::npos) {
            auto maybe_uid = name.substr(pos + 1);
            if (!maybe_uid.empty() && std::all_of(maybe_uid.begin(), maybe_uid.end(), ::isdigit))
                return name.substr(0, pos);
        }
        return name;
    }

    rust::Vec<NodeFFI> res_;
    std::unordered_map<uint32_t, const Def*> added_;
    std::unordered_map<std::string, const Def*> vars_;
    std::unordered_map<std::string, const Lam*> lams_;
    std::unordered_map<std::string, const Def*> axms_;
    std::unordered_map<std::string, const Def*> aliases_;
};

}; // namespace mim::plug::eqsat
