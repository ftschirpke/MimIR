#pragma once

#include <mim/world.h>

namespace mim::plug::matrix::layalg {

struct NatTuple {
    std::vector<std::variant<nat_t, NatTuple>> items;
};

using FlatNatTuple = Vector<nat_t>;

struct Layout {
    NatTuple dims;
    NatTuple strides;
};

struct FlatLayout {
    FlatNatTuple dims;
    FlatNatTuple strides;
};

bool is_valid_layout(const Layout&);

template<class Visitor>
void tuple_visit(const NatTuple& tuple, Visitor&& visitor) {
    for (const auto& item : tuple.items) {
        switch (item.index()) {
            case 0: {
                nat_t value = std::get<0>(item);
                visitor(value);
                break;
            }
            case 1: {
                const NatTuple& nested_tuple = std::get<1>(item);
                tuple_visit(nested_tuple, visitor);
                break;
            }
        }
    }
}

inline nat_t tuple_size(const NatTuple& tuple) {
    nat_t total_size = 1;
    tuple_visit(tuple, [&total_size](nat_t value) { total_size *= value; });
    return total_size;
}
inline nat_t size(const Layout& layout) { return tuple_size(layout.dims); }

/// Simplify one-element tuples inside the layout to just the entry; effectively, does not change the layout
Layout simplify(Layout&&);
FlatLayout flatten(const Layout&);
Layout elevate(FlatLayout&& flat_layout);

/// Extract the i-th sublayout from the layout
Layout sublayout(const Layout&, size_t);
/// Simplify the layout regarding redundant dimensions or strides; changes the layout but not its function
FlatLayout coalesce(const FlatLayout&);
Layout concat(Vector<Layout>&&);

Layout composition(const Layout&, const Layout&);
FlatLayout complement(const Layout&, nat_t);

Layout logical_divide(const Layout&, const Layout&);
std::pair<Layout, Layout> zipped_divide(const Layout&, const Vector<Layout>&);

std::pair<const Def*, const Def*> idx_1DtoND(World&, const Def* mem, const Def* idx, const Layout&);

} // namespace mim::plug::matrix::layalg
