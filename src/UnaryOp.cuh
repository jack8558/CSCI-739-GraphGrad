#pragma once

#include <cmath>
#include <memory>

#include "Tensor.h"
#include "cuda_helpers.h"

enum class UnaryOpType {
    NEG,
    RECIP,
    RELU,
    BIN,
    EXP,
    LOG,
};


#define IMPL_POINTWISE_UNARY_OP(__name, __expr)                                               \
    __global__ void kernel_unary_##__name(const scalar_t* in, CudaArrayRef out) {             \
        size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;                               \
                                                                                              \
        if (index < out.length) {                                                             \
            scalar_t a = in[index];                                                           \
            out.ptr[index] = (__expr);                                                        \
        }                                                                                     \
    }                                                                                         \
                                                                                              \
    inline void unary_compute_data_##__name(Tensor* self, const scalar_t* child_data) {       \
        if (self->on_gpu) {                                                                   \
            auto& data = self->allocate_data_gpu();                                           \
                                                                                              \
            kernel_unary_##__name<<<num_blocks(data.length), BLOCK_SIZE>>>(child_data, data); \
        } else {                                                                              \
            auto& data = self->allocate_data_cpu();                                           \
                                                                                              \
            _Pragma("omp parallel for")                                                       \
            for (size_t i = 0; i < data.size(); i++) {                                        \
                using std::max, std::exp, std::log;                                           \
                scalar_t a = child_data[i];                                                   \
                data[i] = (__expr);                                                           \
            }                                                                                 \
        }                                                                                     \
    }

IMPL_POINTWISE_UNARY_OP(neg, -a)
IMPL_POINTWISE_UNARY_OP(reciprocal, 1 / a)
IMPL_POINTWISE_UNARY_OP(relu, max(a, 0.0))
IMPL_POINTWISE_UNARY_OP(binarilize, scalar_t(a > 0))
IMPL_POINTWISE_UNARY_OP(exp, exp(a))
IMPL_POINTWISE_UNARY_OP(log, log(a))

#undef IMPL_POINTWISE_UNARY_OP


class UnaryOp : public Tensor {
   public:
    UnaryOp(std::shared_ptr<Tensor> arg, UnaryOpType op_type)
        : Tensor(arg->dims), child(arg), op_type(op_type) {
        this->on_gpu = arg->on_gpu;
        this->hashValue = tensor_hash();
    }

    size_t tensor_hash() {
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, static_cast<size_t>(this->op_type));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    void compute_data() override {
        // Evaluate the child node and get its data.
        const scalar_t* child_data = this->child->eval();

        // Call the corresponding function to compute the result.
        switch (this->op_type) {
            case UnaryOpType::NEG:
                unary_compute_data_neg(this, child_data);
                break;
            case UnaryOpType::RECIP:
                unary_compute_data_reciprocal(this, child_data);
                break;
            case UnaryOpType::RELU:
                unary_compute_data_relu(this, child_data);
                break;
            case UnaryOpType::BIN:
                unary_compute_data_binarilize(this, child_data);
                break;
            case UnaryOpType::EXP:
                unary_compute_data_exp(this, child_data);
                break;
            case UnaryOpType::LOG:
                unary_compute_data_log(this, child_data);
                break;
            default:
                throw std::domain_error("bad op_type");
        }
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
