#pragma once

#include <memory>

#include "Tensor.h"

enum UnaryOpType {
    NEG,
};

class UnaryOp : public Tensor {
   public:
    UnaryOp(std::shared_ptr<Tensor> arg, UnaryOpType op_type) : Tensor(arg->dims), child(arg), op_type(op_type) {}

    const scalar_t* eval() override {
        if (!this->data) {
            const scalar_t* child_data = this->child->eval();

            // Allocate a data buffer.
            this->data.emplace(product(dims));
            auto& data = *this->data;

            // Get a function to compute each value.
            scalar_t (*scalar_func)(scalar_t);
            switch (this->op_type) {
                case NEG:
                    scalar_func = [](scalar_t x) { return -x; };
                    break;
            }

            // Fill the buffer with computed values.
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = scalar_func(child_data[i]);
            }
        }

        return data->data();
    }

   protected:
    std::shared_ptr<Tensor> child;
    UnaryOpType op_type;
};
