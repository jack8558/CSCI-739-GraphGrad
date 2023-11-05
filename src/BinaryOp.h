#pragma once

#include <memory>

#include "Tensor.h"
#include "utils.h"

enum class BinaryOpType {
    ADD,
    SUB,
    MULT,
    ELMULT,
    MATMUL,
    POW,
};

class BinaryOp : public Tensor {
   public:
    BinaryOp(std::shared_ptr<Tensor> arg1, std::shared_ptr<Tensor> arg2, BinaryOpType op_type)
        : Tensor(get_dims(*arg1, *arg2, op_type)), leftChild(arg1), rightChild(arg2), op_type(op_type) {}

    const scalar_t* eval() override {
        if (!this->data) {
            // Evaluate the child node and get its data.
            const scalar_t* left_child_data = this->leftChild->eval();
            const scalar_t* right_child_data = this->rightChild->eval();

            // Allocate the data buffer.
            auto& data = this->allocate_data();

            // Get a function to compute each value.
            scalar_t (*scalar_func)(scalar_t, scalar_t);
            switch (this->op_type) {
                case BinaryOpType::ADD:
                    scalar_func = [](scalar_t x, scalar_t y) { return x + y; };
                    break;
                case BinaryOpType::SUB:
                    scalar_func = [](scalar_t x, scalar_t y) { return x - y; };
                    break;
                case BinaryOpType::ELMULT:
                    scalar_func = [](scalar_t x, scalar_t y) { return x * y; };
                    break;
                case BinaryOpType::POW:
                    scalar_func = [](scalar_t x, scalar_t y) { return std::pow(x, y); };
                    break;
                // The below two require different dimension tensors
                case BinaryOpType::MULT:
                    scalar_func = [](scalar_t x, scalar_t y) { return x * y; };
                    break;
                case BinaryOpType::MATMUL:
                    scalar_func = [](scalar_t x, scalar_t y) { return x * y; };
                    break;
                default:
                    throw std::domain_error("bad op_type");
            }

            // Fill the buffer with computed values.
            for (size_t i = 0; i < data.size(); i++) {
                switch (this->op_type) {
                    case BinaryOpType::MULT:
                        if (product(this->rightChild->dims) > 1.0) {
                            throw std::runtime_error("second tensor is not a scalar");
                        }
                        data[i] = scalar_func(left_child_data[i], right_child_data[0]);
                        break;
                    case BinaryOpType::POW:
                        if (product(this->rightChild->dims) > 1.0) {
                            throw std::runtime_error("second tensor is not a scalar");
                        }
                        data[i] = scalar_func(left_child_data[i], right_child_data[0]);
                        break;
                    case BinaryOpType::MATMUL: {
                        if (this->leftChild->dims[1] != this->rightChild->dims[0]) {
                            throw std::runtime_error("Mismatched matmul matrix dimentions");
                        }
                        size_t cols = this->rightChild->dims[1];
                        size_t width = this->leftChild->dims[1];

                        size_t r = i / cols;
                        size_t c = i % cols;
                        for (size_t j = 0; j < width; j++) {
                            data[r * cols + c] += scalar_func(left_child_data[r * width + j], right_child_data[j * cols + c]);
                        }
                        break;
                    }
                    default:
                        data[i] = scalar_func(left_child_data[i], right_child_data[i]);
                }
            }
        }

        return data->data();
    }

   protected:
    static std::vector<size_t> get_dims(const Tensor& left, const Tensor& right, BinaryOpType op_type) {
        switch (op_type) {
            case BinaryOpType::MATMUL:
                return std::vector<size_t>{left.dims[0], right.dims[1]};

            default:
                return left.dims;
        }
    }

    std::shared_ptr<Tensor> leftChild;
    std::shared_ptr<Tensor> rightChild;
    BinaryOpType op_type;
};
