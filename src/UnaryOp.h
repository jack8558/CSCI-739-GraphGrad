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
    UnaryOp(std::shared_ptr<Tensor> arg, UnaryOpType op_type) : Tensor(arg->dims), child(arg), op_type(op_type) {}

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
                case UnaryOpType::TRANSPOSE:
                    break;
                default:
                    throw std::domain_error("bad op_type");
            }

            if (this->op_type == UnaryOpType::TRANSPOSE){
                // Currently only support 2D transpose
                for (size_t i = 0; i < data.size(); i++) {
                    size_t row = i / this->dims[0];
                    size_t col = i % this->dims[0];

                    data[col * (this->dims[1]) + row] = child_data[i];
                }
                this->dims = {this->dims[1], this->dims[0]};
            }
            else {
                // Fill the buffer with computed values.
                for (size_t i = 0; i < data.size(); i++) {
                    data[i] = scalar_func(child_data[i]);
                }
            }

            
        }

        return data->data();
    }

   protected:
    std::shared_ptr<Tensor> child;
    UnaryOpType op_type;
};

// Operator overloads:

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> t) {
    return std::shared_ptr<Tensor>(new UnaryOp(t, UnaryOpType::NEG));
}
