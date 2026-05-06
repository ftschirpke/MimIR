#include <pybind11/pybind11.h>

#include <mim/ast/parser.h>
#include <pybind11/stl.h>


namespace py = pybind11;

namespace mim::ast {

void init_parser(py::module_& m) {
    py::class_<mim::ast::Parser, std::unique_ptr<mim::ast::Parser, py::nodelete>>(m, "Parser")
        .def(py::init<mim::ast::AST&>())
        .def("plugin", [](mim::ast::Parser &p, const std::string plug){
            auto& ast = p.ast();
            if (auto mod = p.plugin(plug)) mod->compile(ast);
        });
}

} // namespace mim::ast
