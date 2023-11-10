#pragma once

#include <memory>

#include "Tensor.h"

enum class UnaryOpType {
    NEG,
    RECIP,
    RELU,
    BIN,
    EXP,
    TRANSPOSE,
};

class UnaryOp : public Tensor {
   public:
    UnaryOp(std::shared_ptr<Tensor> arg, UnaryOpType op_type)
        : Tensor(verify_and_get_dims(*arg, op_type)), child(arg), op_type(op_type) {}

    const scalar_t* eval() override {
        if (!this->data) {
            // Evaluate the child node and get its data.
            const scalar_t* child_data = this->child->eval();

            // Allocate the data buffer.
            auto& data = this->allocate_data();

            // Get a function to compute each value.
            scalar_t (*scalar_func)(scalar_t);
            switch (this->op_type) {
                case UnaryOpType::NEG:
                    scalar_func = [](scalar_t x) { return -x; };
                    break;
                case UnaryOpType::RECIP:
                    scalar_func = [](scalar_t x) { return 1.0 / x; };
                    break;
                case UnaryOpType::RELU:
                    scalar_func = [](scalar_t x) { return x > 0.0 ? x : 0.0; };
                    break;
                case UnaryOpType::BIN:
                    scalar_func = [](scalar_t x) { return x > 0.0 ? 1.0 : 0.0; };
                    break;
                case UnaryOpType::EXP:
                    scalar_func = [](scalar_t x) { return std::exp(x); };
                    break;
                default:
                    throw std::domain_error("bad op_type");
            }

            switch (this->op_type) {
                case UnaryOpType::TRANSPOSE: {
                    // Currently only support 2D transpose
                    for (size_t i = 0; i < data.size(); i++) {
                        size_t row = i / this->dims[1];
                        size_t col = i % this->dims[1];

                        data[col * (this->dims[0]) + row] = child_data[i];
                    }
                    break;
                }

                default:
                    // Fill the buffer with computed values.
                    for (size_t i = 0; i < data.size(); i++) {
                        data[i] = scalar_func(child_data[i]);
                    }
            }
        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
    static std::vector<size_t> verify_and_get_dims(const Tensor& tensor, UnaryOpType op_type) {
        switch (op_type) {
            case UnaryOpType::TRANSPOSE: {
                std::vector<size_t> new_dims(tensor.dims.size());
                for (size_t i = 0; i < tensor.dims.size(); ++i) {
                    new_dims[i] = tensor.dims[tensor.dims.size() - i - 1];
                }
                return new_dims;
            }

            default:
                return tensor.dims;
        }
    }

    std::shared_ptr<Tensor> child;
    UnaryOpType op_type;
};

// Functions:

#define IMPL_OP_FUNC(func_name, op_type)                                         \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t) { \
        return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::op_type));    \
    }

IMPL_OP_FUNC(neg, NEG)
IMPL_OP_FUNC(reciprocal, RECIP)
IMPL_OP_FUNC(relu, RELU)
IMPL_OP_FUNC(binilarize, BIN)
IMPL_OP_FUNC(exp, EXP)
IMPL_OP_FUNC(transpose, TRANSPOSE)

#undef IMPL_OP_FUNC

// Operator overloads:

inline static std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> t) {
    return neg(t);
}
