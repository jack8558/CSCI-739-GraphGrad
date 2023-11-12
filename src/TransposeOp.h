#pragma once

#include <memory>

#include "Tensor.h"

class TransposeOp : public Tensor {
   public:
    TransposeOp(std::shared_ptr<Tensor> arg, size_t dim0, size_t dim1)
        : Tensor(verify_and_get_dims(*arg, dim0, dim1)), child(arg) {
        // Check if dimensions are valid
        if (dim0 < 0 || dim0 >= dims.size() || dim1 < 0 || dim1 >= dims.size()) {
            throw std::invalid_argument("Invalid dimensions for transpose");
        }
        this->dim0 = dim0;
        this->dim1 = dim1;
    }

    const scalar_t* eval() override {
        if (!this->data) {
            // Evaluate the child node and get its data.
            const scalar_t* child_data = this->child->eval();

            // Allocate the data buffer.
            auto& data = this->allocate_data();

            // Computer strides
            std::vector<size_t> strides = get_transposed_strides();
            std::vector<size_t> original_strides = get_original_strides();

            // Transpose the data
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = child_data[find_original_index(strides, original_strides, i)];
            }
        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    // TODO
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

    size_t find_original_index(std::vector<size_t> strides, std::vector<size_t> original_strides, size_t index) {
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

// Functions:

#define IMPL_TRANS_FUNC(func_name)                                                                   \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t, int dim0, int dim1) { \
        return std::shared_ptr<Tensor>(new TransposeOp(t, dim0, dim1));                              \
    }

IMPL_TRANS_FUNC(transpose)

#undef IMPL_TRANS_FUNC