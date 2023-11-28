#pragma once

#include <memory>

#include "Tensor.h"
#include "utils.h"
#include "cuda_helpers.h"

enum class ReductionOpType {
    SUM,
};


__global__ void kernel_reduction_sum(const scalar_t* in, CudaArrayRef out, size_t child_len) {
    size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < child_len) {
        atomicAdd(out.ptr, in[index]);
    }
}

inline void reduction_compute_data_sum(Tensor* self, const scalar_t* child_data, size_t child_len) {
    if (self->on_gpu) {
        auto& data = self->allocate_data_gpu();

        kernel_reduction_sum<<<num_blocks(child_len), BLOCK_SIZE>>>(child_data, data, child_len);
    } else {
        auto& data = self->allocate_data_cpu();

        scalar_t tmp = 0.0;
        _Pragma("omp parallel for reduction(+:tmp)")
        for (size_t i = 0; i < child_len; i++) {
            tmp += child_data[i];
        }
        data[0] = tmp;
    }
}


class ReductionOp : public Tensor {
   public:
    ReductionOp(std::shared_ptr<Tensor> arg, ReductionOpType op_type)
        : Tensor({}), child(arg), op_type(op_type) {
            this->on_gpu = arg->on_gpu;
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
        size_t child_data_len = product(this->child->dims);

        // Allocate the data buffer.
        // auto& data = this->allocate_data_cpu();

        switch (this->op_type) {
            case ReductionOpType::SUM: {
                reduction_compute_data_sum(this, child_data, child_data_len);
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
