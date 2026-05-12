#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mim/flags.h>

namespace py = pybind11;

namespace mim {

void init_flags(py::module_& m) {
    py::class_<mim::Flags>(m, "Flags")
        .def(py::init<>())
        .def_readwrite("scalarize_threshold", &mim::Flags::scalarize_threshold);
}
} // namespace mim
