#pragma once

#include <memory>

#include "Tensor.h"
#include "utils.h"


class ReshapeOp : public Tensor {
   public:

    ReshapeOp(std::shared_ptr<Tensor> arg, std::vector<size_t> new_dims)
        : Tensor(new_dims), child(arg)  {
            if (product(arg->dims) != product(new_dims)) {
                throw std::invalid_argument("Mismatched dims in reshape");
            }
            this->hashValue = tensor_hash();
        }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("reshape"));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    const scalar_t* eval() override {
        // The data for ReshapeOp is exactly the same as its child.
        return this->child->eval();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    void backward_step() override; // Implementation in Tensor_backward.cc

   protected:
    std::shared_ptr<Tensor> child;
};

inline static std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor> t, std::vector<size_t> new_dims) {
    return std::shared_ptr<Tensor>(new ReshapeOp(t, new_dims));
}
