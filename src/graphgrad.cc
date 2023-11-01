#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cstdio>
#include <cstring>
#include <memory>

#include "Tensor.h"
#include "UnaryOp.h"

PYBIND11_MODULE(graphgrad, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("rand", &Tensor::rand)
        .def("__repr__", &Tensor::to_string)
        .def("eval", [](Tensor& t) {
            t.eval();
        });

    m.def("neg", [](std::shared_ptr<Tensor> t) {
        return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::NEG));
    });
}
