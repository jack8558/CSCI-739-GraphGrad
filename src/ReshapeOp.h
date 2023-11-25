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
        }

    const scalar_t* eval() override {
        if (!this->data) {
            // Evaluate the child node and get its data.
            const scalar_t* child_data = this->child->eval();

            // Allocate the data buffer.
            auto& data = this->allocate_data();

            #pragma omp parallel for
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = child_data[i];
            }
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

// Functions:

#define IMPL_RESHAPE_FUNC(func_name)                                                                             \
    inline static std::shared_ptr<Tensor> func_name(std::shared_ptr<Tensor> t, std::vector<size_t> new_dims) { \
        return std::shared_ptr<Tensor>(new ReshapeOp(t, new_dims));                                            \
    }

IMPL_RESHAPE_FUNC(reshape)

#undef IMPL_RESHAPE_FUNC