#pragma once

#include <memory>

#include "Tensor.h"
#include "utils.h"

enum class ReductionOpType {
    SUM,
};

class ReductionOp : public Tensor {
   public:
    ReductionOp(std::shared_ptr<Tensor> arg, ReductionOpType op_type)
        : Tensor(std::vector<size_t>{1}), child(arg), op_type(op_type) {}

    const scalar_t* eval() override {
        if (!this->data) {
            // Evaluate the child node and get its data.
            const scalar_t* child_data = this->child->eval();

            // Allocate the data buffer.
            auto& data = this->allocate_data();

            for (size_t i = 0; i < product(this->child->dims); i++) {
                switch (this->op_type)
                {
                case ReductionOpType::SUM:
                    data[0] += child_data[i];
                    break;
                
                default:
                    throw std::runtime_error("Reduction type not supported.");
                }
                
            }
        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    // TODO
    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
    std::shared_ptr<Tensor> child;
    ReductionOpType op_type;
};

// Functions:
