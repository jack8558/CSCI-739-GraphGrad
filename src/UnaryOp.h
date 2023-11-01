#pragma once
#include <tensor.h>

class UnaryOp : public Tensor {
    shared_ptr<Tensor> child;
    // TODO: define OpType
    // OpType operator;
}