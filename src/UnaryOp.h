#pragma once

#include <Tensor.h>

#include <memory>

class UnaryOp : public Tensor {
    std::shared_ptr<Tensor> child;
    // TODO: define OpType
    // OpType operator;
};
