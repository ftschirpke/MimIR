#include "mim/world.h"

#include "mim/plug/matrix/matrix.h"

namespace mim::plug::matrix {

static std::vector<nat_t> multiply_flatten_tuple(const Tuple* tuple) {
    auto& w = tuple->world();

    std::vector<nat_t> result;

    auto n    = Lit::as<nat_t>(tuple->op(0));
    auto ms   = tuple->op(1);
    auto dims = tuple->op(2);
    for (size_t i = 0; i < n; ++i) {
        w.DLOG("[i={}] n={} ...", i, n);
        auto m_op = (n == 1) ? ms : ms->op(i);
        auto m    = Lit::as<nat_t>(m_op);
        w.DLOG("[i={}] n={} m={} ...", i, n, m);
        auto dim_op       = (n == 1) ? dims : dims->op(i);
        nat_t dim_product = 1;
        for (size_t j = 0; j < m; ++j) {
            auto op = (m == 1) ? dim_op : dim_op->op(j);
            dim_product *= Lit::as<nat_t>(op);
        }
        w.DLOG("[i={}] n={} m={} => dim={}", i, n, m, dim_product);
        result.push_back(dim_product);
    }
    return result;
}

const Def* normalize_shape(const Def* type, const Def* callee, const Def* arg) {
    auto& w = type->world();
    w.DLOG("shape start");
    auto tuple = arg->isa<Tuple>();
    if (!tuple) return nullptr;

    (void)type;
    (void)callee;
    auto dims = multiply_flatten_tuple(tuple);

    DefVec def_vec;
    for (auto dim : dims)
        def_vec.push_back(w.lit_nat(dim));

    auto n        = tuple->op(0);
    auto ms       = w.pack(n, w.lit_nat_1());
    auto new_dims = w.tuple(def_vec);
    w.DLOG("shape end");
    return w.tuple(DefVec{n, ms, new_dims});
}

const Def* normalize_size(const Def* type, const Def* callee, const Def* arg) {
    auto& w    = type->world();
    auto tuple = arg->isa<Tuple>();
    if (!tuple) return nullptr;

    (void)type;
    (void)callee;
    auto dims = multiply_flatten_tuple(tuple);

    nat_t product = 1;
    for (auto dim : dims)
        product *= dim;

    return w.lit_nat(product);
}

const Def* normalize_strides2extract(const Def* type, const Def* callee, const Def* arg) {
    auto& w = type->world();
    w.DLOG("s2e start");

    w.DLOG("s2e: t {}, c {}, a {}", type, callee, arg);

    w.DLOG("s2e end");
    return nullptr;
}

MIM_matrix_NORMALIZER_IMPL

} // namespace mim::plug::matrix
