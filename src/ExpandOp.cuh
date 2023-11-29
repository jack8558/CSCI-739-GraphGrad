#pragma once

#include <memory>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "Tensor.h"
#include "cuda_helpers.h"


__global__ void kernel_expand(const scalar_t* in, size_t in_length, CudaArrayRef out) {
    size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < out.length) {
        out.ptr[index] = in[index % in_length];
    }
}


class ExpandOp : public Tensor {
   public:
    ExpandOp(std::shared_ptr<Tensor> arg, size_t new_dim0_size)
        : Tensor(get_dims(*arg, new_dim0_size)), child(arg) {
        this->on_gpu = arg->on_gpu;
        this->hashValue = tensor_hash();
    }

    size_t tensor_hash() {
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("expand"));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    void compute_data() override {
        // Evaluate the child node and get its data.
        const scalar_t* child_data = this->child->eval();
        size_t child_data_len = product(this->child->dims);

        if (this->on_gpu){
            auto& data = this->allocate_data_gpu();
            kernel_expand<<<num_blocks(data.length), BLOCK_SIZE>>>(child_data, child_data_len, data);
        } else {
            // Allocate the data buffer.
            auto& data = this->allocate_data_cpu();

            // Replicate the child data into the new buffer.
            #pragma omp parallel for
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = child_data[i % child_data_len];
            }
        }
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
    static std::vector<size_t> get_dims(const Tensor& tensor, size_t new_dim0_size) {
        if (new_dim0_size == 0) {
            throw py::value_error("cannot expand to size 0");
        }

        // Add a new dimension of size new_dim0_size to the front.
        std::vector<size_t> new_dims = tensor.dims;
        new_dims.insert(new_dims.begin(), new_dim0_size);
        return new_dims;
    }

    std::shared_ptr<Tensor> child;
};

inline static std::shared_ptr<Tensor> expand(std::shared_ptr<Tensor> t, size_t new_dim0_size) {
    return std::shared_ptr<Tensor>(new ExpandOp(t, new_dim0_size));
}
