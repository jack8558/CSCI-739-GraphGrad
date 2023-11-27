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

    // Equality operator for Tensor
    bool operator==(const ReductionOp& other) const {
        return this->dims == other.dims && this->op_type == other.op_type && this->child == other.child;
    }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("reduction"));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    const scalar_t* eval() override {
        if (!this->data) {
            // Allocate the data buffer.
            auto& data = this->allocate_data();

            auto result = Tensor::lruMap.get(this->hashValue);
            if (result.has_value()) {
                // The key was found, and you can access the value using result.value()
                data = result.value();
                return data.data();
            }

            // Evaluate the child node and get its data.
            const scalar_t* child_data = this->child->eval();

            scalar_t tmp = data[0];

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

            // Add it to hashmap
            Tensor::lruMap.insert(this->hashValue, data);
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
