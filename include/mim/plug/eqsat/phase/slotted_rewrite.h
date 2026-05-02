#pragma once

#include <cstdint>

#include <unordered_map>

#include <mim/phase.h>

#include "mim/def.h"
#include "mim/rewrite.h"

#include "rust/mimir_eqsat.h"

namespace mim::plug::eqsat {

const bool DEBUG        = false;
const bool DEBUG_SCOPES = false;

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
    void init(rust::Vec<RecExprFFI> rec_exprs);
    const Def* init(uint32_t id);
    const Def* init_axm(uint32_t id, NodeFFI node);
    const Def* init_root(uint32_t id, NodeFFI node);
    const Def* init_let(uint32_t id, NodeFFI node);
    const Def* init_con(uint32_t id, NodeFFI node);

    // Performs a bottom-up traverse of each RecExprFFI and
    // creates a Def in the new_world() for every node.
    // At this point, the bodies of the lambdas created
    // in the init phase will be set.
    void convert(rust::Vec<RecExprFFI> rec_exprs);
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

    // The nodes of the RecExprFFI we are currently processing
    void set_nodes(rust::Vec<NodeFFI> nodes) { nodes_ = nodes; }
    rust::Vec<NodeFFI> nodes() const { return nodes_; }

    // Stores Defs that were already created for a node via the nodes' id
    const Def* cache_get(uint32_t id) { return added_[id]; }
    const Def* cache_set(uint32_t id, const Def* def) { return added_[id] = def; }
    void reset_cache() { added_ = {}; }

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
            else if (auto var = get_var(sym); var)
                def = var;
        }
        return def;
    }

    void register_var(std::string name, const Def* def) {
        if (curr_loc_.depth == ROOT_SCOPE_DEPTH) {
            root_scope_.insert({name, def});
            if (DEBUG_SCOPES) std::cout << "Registering: " << name << "-" << def << " in root scope\n";
        } else {
            curr_scope_->var_name = name;
            curr_scope_->def      = def;
            if (DEBUG_SCOPES) std::cout << "Registering: " << curr_scope_->to_str() << "\n";
        }
    }

    const Def* get_var(std::string name) {
        auto curr_scope = get_scope(curr_loc_);

        while (name != curr_scope->var_name) {
            if (curr_scope->parent_loc.depth == ROOT_SCOPE_DEPTH) {
                for (auto [var_name, def] : root_scope_)
                    if (var_name == name) return def;
                break;
            }
            curr_scope = get_scope(curr_scope->parent_loc);
        }

        if (name == curr_scope->var_name) return curr_scope->def;

        return nullptr;
    }

    void register_axm(std::string name, const Axm* converted) {
        if (axms_.contains(name)) {
            std::cerr << "register_axm: can't define the same axiom: " << name << " twice\n"
                      << "existing def: " << axms_[name] << "\n";
            assert(false);
        }
        axms_[name] = converted;
    }
    const Def* get_axm(std::string name) { return axms_[name]; }

    NodeFFI get_node(MimKind expected, uint32_t id) {
        assert(nodes_[id].kind == expected && "get_node: mismatch between expected and actual node kind");
        return nodes_[id];
    }
    NodeFFI get_node_unsafe(uint32_t id) { return nodes_[id]; }

    std::string get_symbol(uint32_t id) { return nodes_[id].symbol.c_str(); }
    uint64_t get_num(uint32_t id) { return nodes_[id].num; }
    std::string get_slot(uint32_t id) { return nodes_[id].slot.c_str(); }

    // Returns a flattened vector of node id's for a cons list
    // i.e.: (cons 23 (cons 12 nil)) => [23, 12]
    std::vector<uint32_t> get_cons_flat(uint32_t id) {
        std::vector<uint32_t> flattened;
        auto curr_cons = get_node_unsafe(id);
        while (curr_cons.kind != MimKind::Nil) {
            flattened.push_back(curr_cons.children[0]);
            curr_cons = get_node_unsafe(curr_cons.children[1]);
        }
        return flattened;
    }

    // Loc tracks the current location in the scope tree.
    // This is done by maintaining a depth and an offset while
    // traversing the RecExprFFI that indicate the exact scope we
    // are currently in. To visualize this:
    //
    //                s1
    //              /    \
    //             s2    s3
    //            /     /  \
    //           s4    s5  s6
    //
    // The location of scope s5 would be at (2, 1) because it is at
    // at a tree-depth of 2 and at an offset of 1 at that depth.
    //
    // The reason we do this is because we convert the RecExprFFI's
    // into the new_world() in multiple traverses. First, we perform
    // a top-down traverse to create all bindings in their proper scopes
    // and then we perform a bottom-up traverse to create all other Defs.
    //
    // To be able to access the variables we bound in the first top-down
    // traverse, in the second bottom-up traverse, we need only to provide
    // our current location and the name of the variable whose definition
    // we need and we can simply look it up in the scopes_ map.
    struct Loc {
        int32_t depth;
        size_t offset;

        bool operator==(const Loc& other) const noexcept { return depth == other.depth && offset == other.offset; }

        std::string to_str() const {
            std::ostringstream os;
            os << "Loc{ depth=" << depth << ", offset=" << offset << " }";
            return os.str();
        }
    };

    struct LocHash {
        std::size_t operator()(const Loc& loc) const noexcept {
            return std::hash<size_t>()(loc.depth) ^ (std::hash<size_t>()(loc.offset) << 1);
        }
    };

    const int32_t ROOT_SCOPE_DEPTH = -1;

    void reset_loc() {
        depth_visits_ = {};
        // We start at Loc {depth: -1, offset: 0} because
        // enter_scope() increments the depth and we want
        // the first scope to begin at (0,0) rather than (1,0)
        curr_loc_ = {ROOT_SCOPE_DEPTH, 0};
    }

    // Keeps track of how often we have visited each scope-depth
    // so we can keep track of the current locations' offset at each depth.
    // maps: Depth -> #Visits
    std::unordered_map<size_t, size_t> depth_visits_;
    Loc curr_loc_;

    struct Scope {
        Loc loc;
        Loc parent_loc;
        std::string var_name;
        const Def* def;

        std::string to_str() const {
            std::ostringstream os;
            os << "Scope{ loc=" << loc.to_str() << ", parent_loc=" << parent_loc.to_str() << ", var=\"" << var_name
               << "\", def=";
            if (def)
                os << def;
            else
                os << "null";
            os << " }";
            return os.str();
        }
    };

    void set_scope() {
        get_scope(curr_loc_)->loc = curr_loc_;
        curr_scope_               = get_scope(curr_loc_);
    }

    void enter_scope(NodeFFI node, bool ignore_last_visit = false) {
        if (node.kind == MimKind::Scope) {
            auto parent_loc = curr_loc_;

            curr_loc_.depth++;
            curr_loc_.offset = depth_visits_[curr_loc_.depth];

            // We act like we didn't already visit at that depth before,
            // which we need to be able to revisit a scope we just exited
            // in the convert() bottom-up traverse.
            if (ignore_last_visit) curr_loc_.offset--;

            set_scope();
            curr_scope_->parent_loc = parent_loc;
            if (DEBUG_SCOPES) std::cout << "Entering: " << curr_scope_->to_str() << "\n";
        }
    }

    void exit_scope(NodeFFI node, bool ignore_visit = false) {
        if (node.kind == MimKind::Scope) {
            if (DEBUG_SCOPES) std::cout << "Exiting: " << curr_scope_->to_str() << "\n";

            if (!ignore_visit) depth_visits_[curr_loc_.depth]++;
            curr_loc_.offset = depth_visits_[--curr_loc_.depth];
            set_scope();
        }
    }

    // The current scope which we mostly use to construct the scope map during init
    Scope* curr_scope_;

    // For every scope-location we store a Scope struct that stores a pointer to its
    // parent scope, the name of the var it introduces, and the Def associated with this var.
    typedef std::unordered_map<Loc, Scope, LocHash> Scopes;
    Scopes* scopes_;

    Scope* get_scope(Loc loc) { return &(*scopes_)[loc]; }

    void set_scopes(size_t rec_expr_idx) {
        scopes_     = &scopes_map_[rec_expr_idx];
        curr_scope_ = get_scope(curr_loc_);
    }

    // For every RecExprFFI keyed by its idx, we store a structure representing its scopes.
    typedef std::unordered_map<size_t, Scopes> ScopesMap;
    ScopesMap scopes_map_;

    // There is a special root scope which is a registry of all top-level/closed Defs
    // that exist beyond the current RecExprFFI.
    std::set<std::pair<std::string, const Def*>> root_scope_;

    rust::Vec<NodeFFI> nodes_;
    std::unordered_map<uint32_t, const Def*> added_;
    std::unordered_map<std::string, const Def*> vars_;
    std::unordered_map<std::string, const Def*> axms_;
    std::unordered_map<std::string, const Def*> aliases_;
};

}; // namespace mim::plug::eqsat
