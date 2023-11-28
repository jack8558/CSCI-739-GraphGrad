#pragma once

#include <memory>

#include "Tensor.h"
#include "cuda_helpers.h"

__global__ void kernel_transpose_2d(const scalar_t* in, CudaArrayRef out, int rows, int cols) {         
    size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;                               
                                                                                            
    if (index < out.length) {                                            
        int new_index = (index % rows) * cols + (index / rows);
        out.ptr[new_index] = in[index];                                                                                                                   
    }                                                                                     
}


class TransposeOp : public Tensor {
   public:
    TransposeOp(std::shared_ptr<Tensor> arg, size_t dim0, size_t dim1)
        : Tensor(verify_and_get_dims(*arg, dim0, dim1)), child(arg) {
        // Check if dimensions are valid
        if (dim0 >= dims.size() || dim1 >= dims.size()) {
            throw std::invalid_argument("Invalid dimensions for transpose");
        }
        this->on_gpu = arg->on_gpu;
        this->hashValue = tensor_hash();
        this->dim0 = dim0;
        this->dim1 = dim1;
    }

    size_t tensor_hash(){
        size_t hashValue = 0;
        Tensor::hash_combine(hashValue, Tensor::vector_hash(this->dims));
        Tensor::hash_combine(hashValue, std::hash<std::string>{}("transpose"));
        Tensor::hash_combine(hashValue, this->child->hashValue);
        return hashValue;
    }

    void compute_data() override {
        // Evaluate the child node and get its data.
        const scalar_t* child_data = this->child->eval();

        if (this->on_gpu){
            auto& data = this->allocate_data_gpu();    
            kernel_transpose_2d<<<num_blocks(data.length), BLOCK_SIZE>>>(child_data, data, this->dims[0], this->dims[1]);
        } else {
            // Allocate the data buffer.
            auto& data = this->allocate_data_cpu();

            // Transpose the data.
            if (this->dims.size() == 2 && this->dim0 != this->dim1) {
                // Special-case 2D matrix transpose for speed.
                size_t rows = this->dims[0];
                size_t cols = this->dims[1];

                #pragma omp parallel for
                for (size_t i = 0; i < cols; i++) {
                    for (size_t j = 0; j < rows; j++) {
                        data[j * cols + i] = child_data[i * rows + j];
                    }
                }
            } else {
                // Compute strides.
                std::vector<size_t> strides;
                std::vector<size_t> original_strides;
                if (this->dim0 != this->dim1) {
                    strides = get_transposed_strides();
                    original_strides = get_original_strides();
                }

                #pragma omp parallel for
                for (size_t i = 0; i < data.size(); ++i) {
                    if (this->dim0 == this->dim1) {
                        data[i] = child_data[i];
                    } else {
                        data[i] = child_data[find_original_index(strides, original_strides, i)];
                    }
                }
            }

        }

        
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    void backward_step() override;  // Implementation in Tensor_backward.cc

   protected:
    static std::vector<size_t> verify_and_get_dims(const Tensor& tensor, size_t dim0, size_t dim1) {
        // Multidimension
        std::vector<size_t> new_dims(tensor.dims.size());
        for (size_t i = 0; i < tensor.dims.size(); ++i) {
            if (i == dim0) {
                new_dims[dim1] = tensor.dims[i];
            } else if (i == dim1) {
                new_dims[dim0] = tensor.dims[i];
            } else {
                new_dims[i] = tensor.dims[i];
            }
        }

        return new_dims;
    }

    std::vector<size_t> get_transposed_strides() {
        std::vector<size_t> strides(dims.size(), 1);
        size_t current_strides = 1;

        for (size_t i = dims.size() - 1; i > 0; --i) {
            // Transposed strides
            current_strides *= dims[i];
            strides[i - 1] = current_strides;
        }

        return strides;
    }

    std::vector<size_t> get_original_strides() {
        std::vector<size_t> strides(dims.size(), 1);
        size_t current_strides = 1;

        for (size_t i = dims.size() - 1; i > 0; --i) {
            // Original strides
            if (i == dim0) {
                current_strides *= dims[dim1];
            } else if (i == dim1) {
                current_strides *= dims[dim0];
            } else {
                current_strides *= dims[i];
            }

            strides[i - 1] = current_strides;
        }

        return strides;
    }

    size_t find_original_index(const std::vector<size_t>& strides, const std::vector<size_t>& original_strides, size_t index) {
        size_t original_index = 0;
        if (dim0 == dims.size() - 1) {
            original_index += (index / strides[dim1] % dims[dim1]);
        } else if (dim1 == dims.size() - 1) {
            original_index += (index / strides[dim0] % dims[dim0]);
        } else {
            original_index = (index / strides[dims.size() - 1] % dims[dims.size() - 1]);
        }

        for (size_t j = 0; j < dims.size() - 1; ++j) {
            if (j == dim0) {
                original_index += (index / strides[dim1] % dims[dim1]) * original_strides[j];
            } else if (j == dim1) {
                original_index += (index / strides[dim0] % dims[dim0]) * original_strides[j];
            } else {
                original_index += (index / strides[j] % dims[j]) * original_strides[j];
            }
        }

        return original_index;
    }

    std::shared_ptr<Tensor> child;
    size_t dim0;
    size_t dim1;
};

inline static std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> t, int dim0, int dim1) {
    // check if in hashmap
    return std::shared_ptr<Tensor>(new TransposeOp(t, dim0, dim1));
}
