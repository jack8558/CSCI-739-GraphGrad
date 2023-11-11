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

            //2D transpose
            for (size_t i = 0; i < data.size(); i++) {
                size_t row = i / this->dims[1];
                size_t col = i % this->dims[1];

                data[col * (this->dims[0]) + row] = child_data[i];
            }

            // // Multidimensional transpose
            // // Compute strides
            // std::vector<size_t> strides(dims.size(), 1);
            // size_t current_strides = 1;
            // strides[dims.size() - 1] = current_strides;

            // for (size_t i = dims.size() - 1; i > 0; --i) {
            //     current_strides *= dims[i];
            //     strides[i-1] = current_strides;
            //     // printf("%ld, %ld\n", (i-1), current_strides);
            // }

            // std::swap(strides[dim0], strides[dim1]);
            
            
            // // Transpose the data
            // for (size_t i = 0; i < data.size(); ++i){
            //     // Compute original index 
            //     // dims are set during construction so current dims are transposed dimes
            //     size_t original_index = (i / strides[dims.size()-1] % dims[dims.size() - 1]);

            //     for (size_t j = 0; j < dims.size()-1; ++j){
            //         original_index += (i / strides[j] % dims[j]) * strides[j+1];
            //     }

            //     data[i] = child_data[original_index];
            //     data[i] = original_index;
            //     // data[i] = strides[dims.size() -1];
            // }
        }

        return data->data();
    }

    std::vector<Tensor*> get_children() override {
        return {this->child.get()};
    }

    // TODO
    void backward_step() override{

    }  // Implementation in Tensor_backward.cc

   protected:
    static std::vector<size_t> verify_and_get_dims(const Tensor& tensor, size_t dim0, size_t dim1) {
        // for 2D
        // std::vector<size_t> new_dims(tensor.dims.size());
        // for (size_t i = 0; i < tensor.dims.size(); ++i) {
        //     new_dims[i] = tensor.dims[tensor.dims.size() - i - 1];
        // }

        // for multidimension
        std::vector<size_t> new_dims(tensor.dims.size());
        for (size_t i = 0; i < tensor.dims.size(); ++i) {
            if (i == dim0){
                new_dims[dim1] = tensor.dims[i];
            }
            else if (i == dim1){
                new_dims[dim0] = tensor.dims[i];
            }
            else{
                new_dims[i] = tensor.dims[i];
            }
        }

        return new_dims;
    }

    std::shared_ptr<Tensor> child;
    size_t dim0;
    size_t dim1;
};

// Functions:

#define IMPL_TRANS_FUNC(func_name)                                                  \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t, int dim0, int dim1) { \
        return std::shared_ptr<Tensor>(new TransposeOp(t, dim0, dim1));                      \
    }

IMPL_TRANS_FUNC(transpose)

#undef IMPL_TRANS_FUNC