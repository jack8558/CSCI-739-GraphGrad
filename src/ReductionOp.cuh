#pragma once

#include <memory>

#include "Tensor.h"
#include "utils.h"
#include "cuda_helpers.h"

enum class ReductionOpType {
    SUM,
    SUM_DIM0,
};


__global__ void kernel_reduction_sum(const scalar_t* in, CudaArrayRef out, size_t child_len) {
    scalar_t sum = 0.0;
    for (unsigned long i = 0; i < child_len; i++) {
        sum += in[i];
    }
    *out.ptr = sum;
}

inline void reduction_compute_data_sum(Tensor* self, const scalar_t* child_data, size_t child_len) {
    if (self->on_gpu) {
        auto& data = self->allocate_data_gpu();

        kernel_reduction_sum<<<1, 1>>>(child_data, data, child_len);
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


__global__ void kernel_reduction_sum_dim0(const scalar_t* in, CudaArrayRef out, size_t child_dim0) {
    size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < out.length) {
        scalar_t sum = 0.0;
        for (size_t i = 0; i < child_dim0; i++) {
            sum += in[(i * out.length) + index];
        }
        out.ptr[index] = sum;
    }
}

inline void reduction_compute_data_sum_dim0(Tensor* self, const scalar_t* child_data, size_t child_dim0) {
    if (self->on_gpu) {
        auto& data = self->allocate_data_gpu();

        kernel_reduction_sum_dim0<<<num_blocks(data.length), BLOCK_SIZE>>>(child_data, data, child_dim0);
    } else {
        auto& data = self->allocate_data_cpu();

        for (size_t index = 0; index < data.size(); index++) {
            scalar_t sum = 0.0;
            _Pragma("omp parallel for reduction(+:sum)")
            for (size_t i = 0; i < child_dim0; i++) {
                sum += child_data[(i * data.size()) + index];
            }
            data[index] = sum;
        }
    }
}


class ReductionOp : public Tensor {
   public:
    ReductionOp(std::shared_ptr<Tensor> arg, ReductionOpType op_type)
        : Tensor(verify_and_get_dims(*arg, op_type)), child(arg), op_type(op_type) {
            this->on_gpu = arg->on_gpu;
            this->hashValue = tensor_hash();
        }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("reduction"));
        Tensor::hash_combine(hashValue, static_cast<size_t>(this->op_type));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    void compute_data() override {
        // Evaluate the child node and get its data.
        const scalar_t* child_data = this->child->eval();

        switch (this->op_type) {
            case ReductionOpType::SUM: {
                size_t child_data_len = product(this->child->dims);
                reduction_compute_data_sum(this, child_data, child_data_len);
                break;
            }
            case ReductionOpType::SUM_DIM0: {
                reduction_compute_data_sum_dim0(this, child_data, this->child->dims[0]);
                break;
            }

            default:
                throw std::runtime_error("bad reduction op_type");
        }
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
    static std::vector<size_t> verify_and_get_dims(const Tensor& arg, ReductionOpType op_type) {
        switch (op_type) {
            case ReductionOpType::SUM: {
                // The result will be a 0-D scalar.
                return {};
            }
            case ReductionOpType::SUM_DIM0: {
                if (arg.dims.size() == 0) {
                    throw py::value_error("cannot sum_dim0 of a 0-D tensor");
                }

                // Remove the first dimension of the arg.
                std::vector<size_t> new_dims = arg.dims;
                new_dims.erase(new_dims.begin());
                return new_dims;
            }

            default:
                throw std::runtime_error("bad reduction op_type");
        }
    }

    std::shared_ptr<Tensor> child;
    ReductionOpType op_type;
};

// Functions:

#define IMPL_OP_FUNC(func_name, op_type)                                              \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t) {      \
        return std::shared_ptr<Tensor>(new ReductionOp(t, ReductionOpType::op_type)); \
    }

IMPL_OP_FUNC(sum, SUM)
IMPL_OP_FUNC(sum_dim0, SUM_DIM0)

#undef IMPL_OP_FUNC
