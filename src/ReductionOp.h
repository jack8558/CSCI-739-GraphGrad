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

            double tmp = data[0];

            #pragma omp parallel for reduction(+:tmp)
            for (size_t i = 0; i < product(this->child->dims); i++) {
                switch (this->op_type) {
                    case ReductionOpType::SUM:
                        // data[0] += child_data[i];
                        tmp += child_data[i];
                        break;

                    default:
                        throw std::runtime_error("Reduction type not supported.");
                }
            }
            data[0] = tmp;
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

#define IMPL_OP_FUNC(func_name, op_type)                                              \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t) {      \
        return std::shared_ptr<Tensor>(new ReductionOp(t, ReductionOpType::op_type)); \
    }

IMPL_OP_FUNC(sum, SUM)

#undef IMPL_OP_FUNC
