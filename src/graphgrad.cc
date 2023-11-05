#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cstdio>
#include <cstring>
#include <memory>

#include "Tensor.h"
#include "UnaryOp.h"
#include "BinaryOp.h"

py::object make_sublist(const std::vector<size_t> &dims, const std::vector<size_t> &strides, const scalar_t *data, size_t dim)
{
    if (dim == dims.size())
    {
        return py::float_(*data);
    }
    else
    {
        py::list list;
        for (size_t i = 0; i < dims[dim]; i++, data += strides[dim])
        {
            list.append(make_sublist(dims, strides, data, dim + 1));
        }
        return list;
    }
}

py::object to_list(Tensor &t)
{
    const std::vector<size_t> &dims = t.dims;
    const size_t num_dims = dims.size();

    std::vector<size_t> strides(num_dims);
    if (num_dims > 0)
    {
        strides[num_dims - 1] = 1;
        for (int i = int(num_dims) - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }

    return make_sublist(dims, strides, t.eval(), 0);
}

PYBIND11_MODULE(graphgrad, m)
{
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("rand", &Tensor::rand)
        .def("to_list", to_list)
        .def("__repr__", [](Tensor &t) { 
            t.eval();  
            return t.to_string();    
        })
        .def("neg", [](std::shared_ptr<Tensor> t) 
            { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::NEG)); })
        .def("reciprocal", [](std::shared_ptr<Tensor> t) 
            { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::RECIP)); })
        .def("relu", [](std::shared_ptr<Tensor> t) 
            { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::RELU)); })
        .def("binilarize", [](std::shared_ptr<Tensor> t) 
            { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::BIN)); })
        .def("exp", [](std::shared_ptr<Tensor> t) 
            { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::EXP)); })
        .def("add", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) 
            { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::ADD)); })
        .def("subtract", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) 
            { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::SUB)); })
        .def("mult", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) 
            { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::MULT)); })
        .def("elementwise_mult", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) 
            { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::ELMULT)); })
        .def("matmul", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) 
            { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::MATMUL)); })
        .def("pow", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) 
            { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::POW)); });

    m.def("neg", [](std::shared_ptr<Tensor> t)
        { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::NEG)); });

    m.def("reciprocal", [](std::shared_ptr<Tensor> t)
        { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::RECIP)); });

    m.def("relu", [](std::shared_ptr<Tensor> t)
        { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::RELU)); });

    m.def("binilarize", [](std::shared_ptr<Tensor> t)
        { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::BIN)); });

    m.def("exp", [](std::shared_ptr<Tensor> t)
        { return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::EXP)); });

    m.def("add", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2)
        { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::ADD)); });

    m.def("subtract", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2)
        { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::SUB)); });

    m.def("mult", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2)
        { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::MULT)); });

    m.def("elementwise_mult", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2)
        { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::ELMULT)); });

    m.def("matmul", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2)
        { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::MATMUL)); });

    m.def("pow", [](std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2)
        { return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::POW)); });
        
}
