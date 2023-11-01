#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cstdio>
#include <cstring>
#include <memory>

#include "Tensor.h"

PYBIND11_MODULE(graphgrad, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor");
}
