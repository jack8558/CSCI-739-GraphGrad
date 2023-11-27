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

    // Equality operator for Tensor
    bool operator==(const ReshapeOp& other) const {
        return this->dims == other.dims && this->child == other.child;
    }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("reshape"));
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


            #pragma omp parallel for
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = child_data[i];
            }

            // Add it to hashmap
            Tensor::lruMap.insert(this->hashValue, data);

        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    // TODO
    void backward_step() override; // Implementation in Tensor_backward.cc

   protected:
    std::shared_ptr<Tensor> child;
};

inline static std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor> t, std::vector<size_t> new_dims) {
    return std::shared_ptr<Tensor>(new ReshapeOp(t, new_dims));                                           
}