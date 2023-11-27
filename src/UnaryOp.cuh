#pragma once

#include <cmath>
#include <memory>

#include "Tensor.h"

enum class UnaryOpType {
    NEG,
    RECIP,
    RELU,
    BIN,
    EXP,
    LOG,
};

__global__ void neg_gpu(scalar_t *in, scalar_t *out, size_t len) {  
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                 
    if (i < len){                                                     
        out[i] =  -in[i];                                          
    }                                                                 
}

class UnaryOp : public Tensor {
   public:
    UnaryOp(std::shared_ptr<Tensor> arg, UnaryOpType op_type)
        : Tensor(arg->dims), child(arg), op_type(op_type) {
            this->hashValue = tensor_hash();
        }

    // Equality operator for Tensor
    bool operator==(const UnaryOp& other) const {
        return this->dims == other.dims && this->op_type == other.op_type && this->child == other.child;
    }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, static_cast<size_t>(this->op_type));
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
                case UnaryOpType::LOG:
                    scalar_func = [](scalar_t x) { return std::log(x); };
                    break;
                default:
                    throw std::domain_error("bad op_type");
            }

            #pragma omp parallel for
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = scalar_func(child_data[i]);
            }
            
            // // Add it to hashmap
            Tensor::lruMap.insert(this->hashValue, data);
        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
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
IMPL_OP_FUNC(log, LOG)

#undef IMPL_OP_FUNC

// Operator overloads:

inline static std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> t) {
    return neg(t);
}
