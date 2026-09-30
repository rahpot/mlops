#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <string>

namespace py = pybind11;

namespace {
    void require_supported_array(const py::array& value, const char* name) {
        if (!value.dtype().equal(py::dtype::of<double>())) {
            throw py::type_error(std::string(name) + " must be a native-endian float64 array");
        }
        if (value.ndim() != 3) {
            throw py::value_error(std::string(name) + " must have exactly 3 dimensions");
        }
        if ((value.flags() & py::array::c_style) == 0) {
            throw py::value_error(std::string(name) + " must be C-contiguous");
        }
        if (!value.attr("flags").attr("aligned").cast<bool>()) {
            throw py::value_error(std::string(name) + " must be aligned for float64");
        }
    }

    void require_same_shape(const py::array& reference, const py::array& value, const char* name) {
        for (py::ssize_t axis = 0; axis < reference.ndim(); ++axis) {
            if (reference.shape(axis) != value.shape(axis)) {
                throw py::value_error(std::string(name) + " must have the same shape as a");
            }
        }
    }
}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    require_supported_array(a, "a");
    require_supported_array(b, "b");
    require_supported_array(c, "c");
    require_same_shape(a, b, "b");
    require_same_shape(a, c, "c");

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});

    mac_kernel(static_cast<const double*>(a.data()),
               static_cast<const double*>(b.data()),
               static_cast<const double*>(c.data()),
               out.mutable_data(),
               static_cast<std::size_t>(a.size()));
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
