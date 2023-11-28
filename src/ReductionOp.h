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
        : Tensor(std::vector<size_t>{1}), child(arg), op_type(op_type) {
            this->hashValue = tensor_hash();
        }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("reduction"));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    void compute_data() override {
        // Evaluate the child node and get its data.
        const scalar_t* child_data = this->child->eval();

        // Allocate the data buffer.
        auto& data = this->allocate_data_cpu();

        switch (this->op_type) {
            case ReductionOpType::SUM: {
                size_t child_data_len = product(this->child->dims);
                scalar_t tmp = 0.0;
                #pragma omp parallel for reduction(+:tmp)
                for (size_t i = 0; i < child_data_len; i++) {
                    tmp += child_data[i];
                }
                data[0] = tmp;
                break;
            }

            default:
                throw std::runtime_error("Reduction type not supported.");
        }
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

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
