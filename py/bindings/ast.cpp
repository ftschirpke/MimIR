#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mim/ast/ast.h>

namespace py = pybind11;

namespace mim::ast {

void init_ast(py::module_& m) { py::class_<mim::ast::AST>(m, "AST").def(py::init<>()).def(py::init<mim::World&>()); }

} // namespace mim::ast
