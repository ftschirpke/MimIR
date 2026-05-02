#include <cstdint>

#include <mim/plug/eqsat/eqsat.h>
#include <mim/plug/eqsat/phase/slotted_rewrite.h>

#include "mim/def.h"
#include "mim/driver.h"

#include "mim/plug/eqsat/autogen.h"

namespace mim::plug::eqsat {

void SlottedRewrite::start() {
    auto [rulesets, cost_fn] = import_config();

    // We are assuming that the core plugin and its backends have been loaded at this point
    // because the 'eqsat' plugin declared it as a dependency via 'plugin core;'
    std::ostringstream sexpr;
    driver().backend("sexpr-slotted")(old_world(), sexpr);

    if (DEBUG) std::cout << sexpr.str() << "\n";

    auto rec_exprs = eqsat_slotted(sexpr.str(), rulesets, cost_fn);

    if (DEBUG) std::cout << pretty_ffi(rec_exprs, 80).c_str() << "\n";

    init(rec_exprs);
    convert(rec_exprs);

    swap(old_world(), new_world());
}

std::pair<rust::Vec<RuleSet>, CostFn> SlottedRewrite::import_config() {
    // Internalize config lambdas (with signature [] -> %eqsat.Ruleset | %eqsat.CostFun | %eqsat.Impl)
    DefVec lams;
    for (auto def : old_world().externals().mutate()) {
        if (auto lam = def->isa<Lam>()) {
            if (Axm::isa<eqsat::Ruleset>(lam->ret_dom()) || Axm::isa<eqsat::CostFun>(lam->ret_dom())
                || Axm::isa<eqsat::Impl>(lam->ret_dom())) {
                lams.push_back(lam);
                def->internalize();
            }
        }
    }

    // Import rulesets and cost function from config lambdas
    rust::Vec<RuleSet> rulesets;
    CostFn cost_fn = CostFn::AstSize;
    for (auto lam : lams) {
        auto body = lam->as<Lam>()->body();
        if (auto body_app = body->isa<App>()) {
            if (auto ruleset_config = Axm::isa<eqsat::rulesets>(body_app->arg())) {
                for (auto ruleset : ruleset_config->args())
                    if (Axm::isa<eqsat::standard>(ruleset))
                        rulesets.push_back(RuleSet::Standard);
                    else
                        assert(false && "Provided ruleset does not exist for slotted");

            } else if (Axm::isa<eqsat::AstSize>(body_app->arg())) {
                cost_fn = CostFn::AstSize;
            } else if (Axm::isa<eqsat::slotted>(body_app->arg()) || Axm::isa<eqsat::egg>(body_app->arg())) {
                continue;
            } else {
                assert(false && "Invalid config value provided for slotted");
            }
        }
    }

    return {rulesets, cost_fn};
}

void SlottedRewrite::init(rust::Vec<RecExprFFI> rec_exprs) {
    size_t rec_expr_idx = 0;
    for (auto rec_expr : rec_exprs) {
        reset_loc();
        reset_cache();

        set_nodes(rec_expr.nodes);
        set_scopes(rec_expr_idx);

        auto root_id = nodes().size() - 1;
        init(root_id);

        rec_expr_idx++;
    }
}

const Def* SlottedRewrite::init(uint32_t id) {
    auto node = get_node_unsafe(id);
    enter_scope(node);

    const Def* res = nullptr;
    switch (node.kind) {
        case MimKind::Axm: res = init_axm(id, node); break;
        case MimKind::Root: res = init_root(id, node); break;
        case MimKind::Let: res = init_let(id, node); break;
        default: break;
    }

    for (uint32_t child : node.children)
        init(child);

    exit_scope(node);
    return res;
}

// (axm <name> <type>)
const Def* SlottedRewrite::init_axm(uint32_t id, NodeFFI node) {
    if (DEBUG) std::cout << "init - current node(" << id << "): " << node_ffi_str(node).c_str() << " - ";
    auto name = get_symbol(node.children[0]);
    if (DEBUG) std::cout << "\n";
    auto type = convert(node.children[1], true);

    auto new_axm = new_world().axm(type);
    new_axm->set(name);
    register_axm(name, new_axm);

    if (DEBUG) std::cout << new_axm << "\n";
    return new_axm;
}

// (root <extern> <name> <definition>)
const Def* SlottedRewrite::init_root(uint32_t id, NodeFFI node) {
    if (DEBUG) std::cout << "init - current node(" << id << "): " << node_ffi_str(node).c_str() << " - \n";

    auto name = get_symbol(node.children[1]);

    const Def* def = nullptr;
    auto def_node  = get_node_unsafe(node.children[2]);
    if (def_node.kind == MimKind::Con) {
        def = init_con(node.children[2], def_node);
        def->set(name);
        register_var(name, def);
    }

    if (DEBUG) std::cout << def << "\n";
    return def;
}

// (let $name (scope <definition> <expression>))
const Def* SlottedRewrite::init_let(uint32_t id, NodeFFI node) {
    if (DEBUG) std::cout << "init - current node(" << id << "): " << node_ffi_str(node).c_str() << " - \n";

    auto name       = get_slot(id);
    auto name_scope = get_node(MimKind::Scope, node.children[0]);

    const Def* def = nullptr;
    auto def_node  = get_node_unsafe(name_scope.children[0]);

    enter_scope(name_scope);
    if (def_node.kind == MimKind::Con) {
        def = init_con(name_scope.children[0], def_node);
        def->set(name);
        register_var(name, def);
    } else {
        def = convert(name_scope.children[0], true);
        def->set(name);
        register_var(name, def);
    }
    exit_scope(name_scope, true);

    if (DEBUG) std::cout << def << "\n";
    return nullptr;
}

// (con <domain-type> $var-name (scope <filter> <body>))
const Def* SlottedRewrite::init_con(uint32_t id, NodeFFI node) {
    if (DEBUG) std::cout << "init - current node(" << id << "): " << node_ffi_str(node).c_str() << " - \n";

    // TODO: Polymorphic lambdas
    auto domain_type = convert(node.children[0], true);
    auto new_con     = new_world().mut_con(domain_type);

    auto var_name = get_slot(id);
    auto var      = new_con->var();
    var->set(var_name);

    // When we are calling on init_con() via init_let(), we have to manually adjust our location for
    // the additional scopes introduced by the let binder and the continuations' variable.
    // This is because location changes are made via enter_/exit_scope() in init() where init() calls
    // on init_let() which calls on init_con() so the location changes from init() are not being applied.
    auto var_scope = get_node(MimKind::Scope, node.children[1]);
    enter_scope(var_scope);
    register_var(var_name, var);
    // We set ignore_visit=true to ensure that this visit won't be
    // counted twice in depth_visits_ (init() will already count it once)
    exit_scope(var_scope, true);

    return new_con;
}

void SlottedRewrite::convert(rust::Vec<RecExprFFI> rec_exprs) {
    size_t rec_expr_idx = 0;
    for (auto rec_expr : rec_exprs) {
        reset_loc();
        reset_cache();

        set_nodes(rec_expr.nodes);
        set_scopes(rec_expr_idx);

        auto root_id = nodes().size() - 1;
        convert(root_id, true);

        rec_expr_idx++;
    }
}

const Def* SlottedRewrite::convert(uint32_t id, bool recurse) {
    auto node = get_node_unsafe(id);

    enter_scope(node);

    if (recurse)
        for (uint32_t child : node.children)
            convert(child, recurse);

    const Def* res = cache_get(id);
    if (res) return res;

    if (DEBUG) std::cout << "convert - current node(" << id << "): " << node_ffi_str(node).c_str() << " - ";
    switch (node.kind) {
        case MimKind::Root: res = convert_root(id, node); break;
        case MimKind::Let: res = convert_let(id, node); break;
        case MimKind::Con: res = convert_con(id, node); break;
        case MimKind::App: res = convert_app(id, node); break;
        case MimKind::Var: res = convert_var(id, node); break;
        case MimKind::Lit: res = convert_lit(id, node); break;
        case MimKind::Pack: res = convert_pack(id, node); break;
        case MimKind::Tuple: res = convert_tuple(id, node); break;
        case MimKind::Extract: res = convert_extract(id, node); break;
        case MimKind::Insert: res = convert_insert(id, node); break;
        case MimKind::Inj: res = convert_inj(id, node); break;
        case MimKind::Merge: res = convert_merge(id, node); break;
        case MimKind::Match: res = convert_match(id, node); break;
        case MimKind::Proxy: res = convert_proxy(id, node); break;
        case MimKind::Join: res = convert_join(id, node); break;
        case MimKind::Meet: res = convert_meet(id, node); break;
        case MimKind::Bot: res = convert_bot(id, node); break;
        case MimKind::Top: res = convert_top(id, node); break;
        case MimKind::Arr: res = convert_arr(id, node); break;
        case MimKind::Sigma: res = convert_sigma(id, node); break;
        case MimKind::Cn: res = convert_cn(id, node); break;
        case MimKind::Pi: res = convert_pi(id, node); break;
        case MimKind::Idx: res = convert_idx(id, node); break;
        case MimKind::Hole: res = convert_hole(id, node); break;
        case MimKind::Type: res = convert_type(id, node); break;
        case MimKind::Num: res = convert_num(id, node); break;
        case MimKind::Symbol: res = convert_symbol(id, node); break;
        default: break;
    }

    if (DEBUG_SCOPES && node.kind == MimKind::Scope) std::cout << "\n";
    exit_scope(node);

    if (DEBUG) std::cout << res << "\n";
    return cache_set(id, res);
}

// (root <extern> <name> <definition>)
const Def* SlottedRewrite::convert_root(uint32_t id, NodeFFI node) {
    auto is_extern = get_symbol(node.children[0]);
    auto def       = get_def(node.children[1]);

    if (def->isa<Lam>()) {
        auto con       = def->as_mut<Lam>();
        auto con_node  = get_node(MimKind::Con, node.children[2]);
        auto var_scope = get_node(MimKind::Scope, con_node.children[1]);
        if (DEBUG_SCOPES) std::cout << "\n";
        enter_scope(var_scope, true);
        auto filter = get_def(var_scope.children[0]);
        auto body   = get_def(var_scope.children[1]);
        if (filter && body) {
            con->set_filter(filter);
            con->set_body(body);
        } else {
            con->set_filter(false);
        }
        if (is_extern == "extern") con->externalize();
        exit_scope(var_scope, true);
    }

    return def;
}

// (let $name (scope <definition> <expression>))
const Def* SlottedRewrite::convert_let(uint32_t id, NodeFFI node) {
    auto name_scope = get_node(MimKind::Scope, node.children[0]);
    if (DEBUG_SCOPES) std::cout << "\n";
    enter_scope(name_scope, true);
    auto def  = get_def(id);
    auto expr = get_def(name_scope.children[1]);

    if (def->isa<Lam>()) {
        auto con       = def->as_mut<Lam>();
        auto con_node  = get_node(MimKind::Con, name_scope.children[0]);
        auto var_scope = get_node(MimKind::Scope, con_node.children[1]);
        enter_scope(var_scope, true);
        auto filter = get_def(var_scope.children[0]);
        auto body   = get_def(var_scope.children[1]);
        if (filter && body) {
            con->set_filter(filter);
            con->set_body(body);
        } else {
            con->set_filter(false);
        }
        exit_scope(var_scope, true);
    }

    exit_scope(name_scope, true);
    return expr;
}

// (con <domain-type> $var-name (scope <filter> <body>))
const Def* SlottedRewrite::convert_con(uint32_t id, NodeFFI node) {
    // Real conversion happens in convert_root and convert_let because we need a name to get to the Def
    auto con = get_def(id);
    return con;
}

// (app <callee> <arg>)
const Def* SlottedRewrite::convert_app(uint32_t id, NodeFFI node) {
    auto callee  = get_def(node.children[0]);
    auto arg     = get_def(node.children[1]);
    auto new_app = new_world().app(callee, arg);
    return new_app;
}

// (var $name)
const Def* SlottedRewrite::convert_var(uint32_t id, NodeFFI node) {
    auto var = get_def(id);
    return var;
}

// (lit <val> <type>)
const Def* SlottedRewrite::convert_lit(uint32_t id, NodeFFI node) {
    auto lit_def = get_def(node.children[0]);
    if (lit_def) return lit_def;

    auto lit_val  = get_num(node.children[0]);
    auto lit_type = get_def(node.children[1]);
    auto new_lit  = new_world().lit(lit_type, lit_val);
    return new_lit;
}

// (pack <arity> <body>)
const Def* SlottedRewrite::convert_pack(uint32_t id, NodeFFI node) {
    auto arity    = get_def(node.children[0]);
    auto body     = get_def(node.children[1]);
    auto new_pack = new_world().pack(arity, body);
    return new_pack;
}

// (tuple <elem-cons>)
const Def* SlottedRewrite::convert_tuple(uint32_t id, NodeFFI node) {
    auto elem_ids = get_cons_flat(node.children[0]);

    DefVec elems;
    for (auto elem_id : elem_ids) {
        auto elem = get_def(elem_id);
        elems.push_back(elem);
    }
    auto new_tuple = new_world().tuple(elems);
    return new_tuple;
}

// (extract <tuple> <index>)
const Def* SlottedRewrite::convert_extract(uint32_t id, NodeFFI node) {
    auto tuple       = get_def(node.children[0]);
    auto index       = get_def(node.children[1]);
    auto new_extract = new_world().extract(tuple, index);
    return new_extract;
}

// (ins <tuple> <index> <value>)
const Def* SlottedRewrite::convert_insert(uint32_t id, NodeFFI node) {
    auto tuple      = get_def(node.children[0]);
    auto index      = get_def(node.children[1]);
    auto value      = get_def(node.children[2]);
    auto new_insert = new_world().insert(tuple, index, value);
    return new_insert;
}

// (inj <type> <value>)
const Def* SlottedRewrite::convert_inj(uint32_t id, NodeFFI node) {
    auto type    = get_def(node.children[0]);
    auto value   = get_def(node.children[1]);
    auto new_inj = new_world().inj(type, value);
    return new_inj;
}

// (merge <type> <value-cons>)
const Def* SlottedRewrite::convert_merge(uint32_t id, NodeFFI node) {
    auto type = get_def(node.children[0]);

    auto value_ids = get_cons_flat(node.children[1]);
    DefVec values;
    for (auto value_id : value_ids) {
        auto value = get_def(value_id);
        values.push_back(value);
    }
    auto new_merge = new_world().merge(type, values);
    return new_merge;
}

// (match <op-cons>)
const Def* SlottedRewrite::convert_match(uint32_t id, NodeFFI node) {
    auto op_ids = get_cons_flat(node.children[0]);

    DefVec ops;
    for (auto op_id : op_ids) {
        auto op = get_def(op_id);
        ops.push_back(op);
    }
    auto new_match = new_world().match(ops);
    return new_match;
}

// (proxy <type> <pass> <tag> <op-cons>)
const Def* SlottedRewrite::convert_proxy(uint32_t id, NodeFFI node) {
    auto type = get_def(node.children[0]);
    auto pass = get_num(node.children[1]);
    auto tag  = get_num(node.children[2]);

    auto op_ids = get_cons_flat(node.children[3]);
    DefVec ops;
    for (auto op_id : op_ids) {
        auto op = get_def(op_id);
        ops.push_back(op);
    }
    auto new_proxy = new_world().proxy(type, ops, pass, tag);
    return new_proxy;
}

// (join <type-cons>)
const Def* SlottedRewrite::convert_join(uint32_t id, NodeFFI node) {
    auto type_ids = get_cons_flat(node.children[0]);

    DefVec types;
    for (auto type_id : type_ids) {
        auto type = get_def(type_id);
        types.push_back(type);
    }
    auto new_join = new_world().join(types);
    return new_join;
}

// (meet <type-cons>)
const Def* SlottedRewrite::convert_meet(uint32_t id, NodeFFI node) {
    auto type_ids = get_cons_flat(node.children[0]);

    DefVec types;
    for (auto type_id : type_ids) {
        auto type = get_def(type_id);
        types.push_back(type);
    }
    auto new_meet = new_world().meet(types);
    return new_meet;
}

// (bot <type>)
const Def* SlottedRewrite::convert_bot(uint32_t id, NodeFFI node) {
    auto type    = get_def(node.children[0]);
    auto new_bot = new_world().bot(type);
    return new_bot;
}

// (top <type>)
const Def* SlottedRewrite::convert_top(uint32_t id, NodeFFI node) {
    auto type    = get_def(node.children[0]);
    auto new_top = new_world().top(type);
    return new_top;
}

// (arr <arity> <body>)
const Def* SlottedRewrite::convert_arr(uint32_t id, NodeFFI node) {
    auto arity   = get_def(node.children[0]);
    auto body    = get_def(node.children[1]);
    auto new_arr = new_world().arr(arity, body);
    return new_arr;
}

// (sigma <type-cons>)
const Def* SlottedRewrite::convert_sigma(uint32_t id, NodeFFI node) {
    auto type_ids = get_cons_flat(node.children[0]);

    DefVec types;
    for (auto type_id : type_ids) {
        auto type = get_def(type_id);
        types.push_back(type);
    }

    auto new_sigma = new_world().sigma(types);
    return new_sigma;
}

// (cn <domain>)
const Def* SlottedRewrite::convert_cn(uint32_t id, NodeFFI node) {
    auto domain = get_def(node.children[0]);
    auto new_cn = new_world().cn(domain);
    return new_cn;
}

// (pi <domain> <codomain>)
const Def* SlottedRewrite::convert_pi(uint32_t id, NodeFFI node) {
    auto domain   = get_def(node.children[0]);
    auto codomain = get_def(node.children[1]);
    auto new_pi   = new_world().pi(domain, codomain);
    return new_pi;
}

// (idx <size>)
const Def* SlottedRewrite::convert_idx(uint32_t id, NodeFFI node) {
    auto size    = get_def(node.children[0]);
    auto new_idx = new_world().type_idx(size);
    return new_idx;
}

// (hole <type>)
const Def* SlottedRewrite::convert_hole(uint32_t id, NodeFFI node) {
    auto type_    = get_def(node.children[0]);
    auto new_hole = new_world().mut_hole(type_);
    return new_hole;
}

// (type <level>)
const Def* SlottedRewrite::convert_type(uint32_t id, NodeFFI node) {
    auto level    = get_def(node.children[0]);
    auto new_type = new_world().type(level);
    return new_type;
}

// <u64>
const Def* SlottedRewrite::convert_num(uint32_t id, NodeFFI node) { return nullptr; }

// <string>
const Def* SlottedRewrite::convert_symbol(uint32_t id, NodeFFI node) {
    auto def = get_def(id);
    return def;
}

} // namespace mim::plug::eqsat
