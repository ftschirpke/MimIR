#include <fe/sym.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mim/def.h>
namespace py = pybind11;

namespace fe {

void init_sym(py::module_& m) {
    py::class_<fe::Sym, std::unique_ptr<fe::Sym, py::nodelete>>(m, "Sym")
        .def(py::init<>())
        .def("empty", &fe::Sym::empty)
        .def("size", &fe::Sym::size)
        .def("view", &fe::Sym::view)
        .def("str", &fe::Sym::str);
}

void init_sym_pool(py::module_& m) {
    py::class_<fe::SymPool>(m, "SymPool")
        .def(py::init<>())
        .def("sym", static_cast<Sym (SymPool::*)(std::string_view)>(&fe::SymPool::sym))
        .def("sym", static_cast<Sym (SymPool::*)(const std::string&)>(&fe::SymPool::sym))
        .def("sym", static_cast<Sym (SymPool::*)(const char*)>(&fe::SymPool::sym));
}

} // namespace fe
