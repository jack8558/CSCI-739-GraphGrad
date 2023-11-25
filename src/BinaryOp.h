#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <memory>

#include "Tensor.h"
#include "utils.h"

enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    MATMUL,
    POW,
    DIV,
};

class BinaryOp : public Tensor {
   public:
    BinaryOp(std::shared_ptr<Tensor> arg1, std::shared_ptr<Tensor> arg2, BinaryOpType op_type)
        : Tensor(verify_and_get_dims(*arg1, *arg2, op_type)), leftChild(arg1), rightChild(arg2), op_type(op_type) {}

    const scalar_t* eval() override {
        if (!this->data) {
            // Evaluate the child node and get its data.
            const scalar_t* left_child_data = this->leftChild->eval();
            const scalar_t* right_child_data = this->rightChild->eval();

            // Allocate the data buffer.
            auto& data = this->allocate_data();

            // Get a function to compute each value.
            scalar_t (*scalar_func)(scalar_t, scalar_t);
            switch (this->op_type) {
                case BinaryOpType::ADD:
                    scalar_func = [](scalar_t x, scalar_t y) { return x + y; };
                    break;
                case BinaryOpType::SUB:
                    scalar_func = [](scalar_t x, scalar_t y) { return x - y; };
                    break;
                case BinaryOpType::POW:
                    scalar_func = [](scalar_t x, scalar_t y) { return std::pow(x, y); };
                    break;
                case BinaryOpType::MUL:  // MUL and MATMUL implementation of scalar_func is same
                case BinaryOpType::MATMUL:
                    scalar_func = [](scalar_t x, scalar_t y) { return x * y; };
                    break;
                case BinaryOpType::DIV:
                    scalar_func = [](scalar_t x, scalar_t y) { return x / y; };
                    break;
                default:
                    throw std::domain_error("bad op_type");
            }

            // Fill the buffer with computed values.
            #pragma omp parallel for
            for (size_t i = 0; i < data.size(); i++) {
                switch (this->op_type) {
                    case BinaryOpType::MATMUL: {
                        size_t width;
                        size_t cols;
                        if (this->rightChild->dims.size() == 1) {
                            width = 1;
                            cols = this->rightChild->dims[0];
                        } else {
                            width = this->rightChild->dims[0];
                            cols = this->rightChild->dims[1];
                        }

                        size_t r = i / cols;
                        size_t c = i % cols;
                        for (size_t j = 0; j < width; j++) {
                            data[r * cols + c] += scalar_func(left_child_data[r * width + j], right_child_data[j * cols + c]);
                        }
                        break;
                    }
                    default:
                        if (product(leftChild->dims) == 1) {
                            data[i] = scalar_func(left_child_data[0], right_child_data[i]);
                        } else if (product(rightChild->dims) == 1) {
                            data[i] = scalar_func(left_child_data[i], right_child_data[0]);
                        } else {
                            data[i] = scalar_func(left_child_data[i], right_child_data[i]);
                        }
                }
            }
        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->leftChild.get(), this->rightChild.get()};
    }

    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
    static std::vector<size_t> verify_and_get_dims(const Tensor& left, const Tensor& right, BinaryOpType op_type) {
        switch (op_type) {
            case BinaryOpType::MATMUL:
                if (product(left.dims) == 1 && product(right.dims) == 1) {
                    if (left.dims.size() == 1 && right.dims.size() == 1)
                        return std::vector<size_t>{1};
                    else if (left.dims.size() == 2 && right.dims.size() == 2)
                        return std::vector<size_t>{1, 1};
                    return std::vector<size_t>{1};
                } else if (left.dims.size() <= 1 && right.dims.size() <= 1) {
                    if (left.dims[0] <= right.dims[0]) {
                        return std::vector<size_t>{right.dims[0]};
                    } else {
                        return std::vector<size_t>{left.dims[0]};
                    }
                } else if ((left.dims.size() == 1 && right.dims.size() == 2) && (left.dims[0] == right.dims[0])) {
                    return std::vector<size_t>{right.dims[1]};
                } else if ((left.dims.size() == 2 && right.dims.size() == 1) && (left.dims[1] == right.dims[0])) {
                    return std::vector<size_t>{left.dims[0]};
                } else if ((left.dims.size() == 2 && right.dims.size() == 2) && left.dims[1] == right.dims[0]) {
                    return std::vector<size_t>{left.dims[0], right.dims[1]}; 
                } else{
                    std::string error_message = "invalid matmul dims: left.dims=";
                    error_message += vector_to_string(left.dims);
                    error_message += ", right.dims=";
                    error_message += vector_to_string(right.dims);
                    throw py::value_error(error_message);
                }

            default:
                if (left.dims != right.dims && product(left.dims) != 1 && product(right.dims) != 1) {
                    std::string error_message = "binary op dims mismatch: left.dims=";
                    error_message += vector_to_string(left.dims);
                    error_message += ", right.dims=";
                    error_message += vector_to_string(right.dims);
                    throw py::value_error(error_message);
                } else if (product(left.dims) == 1) {
                    return right.dims;
                } else {
                    return left.dims;
                }
        }
    }

    std::shared_ptr<Tensor> leftChild;
    std::shared_ptr<Tensor> rightChild;
    BinaryOpType op_type;
};

// Operator overloads:

#define IMPL_OPERATOR_OVERLOAD(op, op_type)                                                                     \
    inline static std::shared_ptr<Tensor> operator op(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) { \
        return std::shared_ptr<Tensor>(new BinaryOp(t1, t2, BinaryOpType::op_type));                            \
    }

IMPL_OPERATOR_OVERLOAD(+, ADD)
IMPL_OPERATOR_OVERLOAD(-, SUB)
IMPL_OPERATOR_OVERLOAD(*, MUL)
IMPL_OPERATOR_OVERLOAD(/, DIV)

#undef IMPL_OPERATOR_OVERLOAD

// Functions:

#define IMPL_OP_FUNC(func_name, op_type)                                         \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> t2) { \
        return std::shared_ptr<Tensor>(new BinaryOp(t, t2, BinaryOpType::op_type));    \
    }

IMPL_OP_FUNC(add, ADD)
IMPL_OP_FUNC(subtract, SUB)
IMPL_OP_FUNC(mul, MUL)
IMPL_OP_FUNC(matmul, MATMUL)
IMPL_OP_FUNC(pow, POW)
IMPL_OP_FUNC(div, DIV)


#undef IMPL_OP_FUNC
