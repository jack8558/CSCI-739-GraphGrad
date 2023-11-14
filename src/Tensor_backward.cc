#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cmath>
#include <unordered_set>

#include "BinaryOp.h"
#include "ReductionOp.h"
#include "ReshapeOp.h"
#include "Tensor.h"
#include "TransposeOp.h"
#include "UnaryOp.h"

// Collects all the nodes in the graph under the given root, in reverse topological order.
static void topo_sort_visit(Tensor* node, std::unordered_set<Tensor*>& visited_nodes, std::vector<Tensor*>& sorted_nodes) {
    if (visited_nodes.find(node) != visited_nodes.end()) {
        return;
    }

    for (Tensor* child : node->get_children()) {
        topo_sort_visit(child, visited_nodes, sorted_nodes);
    }

    visited_nodes.insert(node);
    sorted_nodes.push_back(node);
}

// Collects all the nodes in the graph under the given root, in reverse topological order.
static std::vector<Tensor*> get_sorted_nodes(Tensor* root) {
    std::unordered_set<Tensor*> visited_nodes;
    std::vector<Tensor*> sorted_nodes;
    topo_sort_visit(root, visited_nodes, sorted_nodes);
    return sorted_nodes;
}

void Tensor::backward() {
    if (product(this->dims) != 1) {
        throw std::runtime_error("called backward() on a non-scalar Tensor");
    }

    // Collect the set of DAG nodes in (reverse) topological order.
    std::vector<Tensor*> sorted_nodes = get_sorted_nodes(this);

    // Set the root `grad` to 1 and set all the other `grad`s to `nullptr`.
    for (Tensor* t : sorted_nodes) {
        t->grad = nullptr;
    }
    this->grad = Tensor::from_scalar(1.0);

    // Traverse the graph in topological order (starting with this (the root) node),
    // assigning/accumulating the gradients.
    for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
        Tensor* t = *it;
        t->backward_step();
    }
}

void Tensor::add_grad(std::shared_ptr<Tensor> grad) {
    if (this->grad) {
        this->grad = this->grad + grad;
    } else {
        this->grad = std::move(grad);
    }
}

// Gradient definitions for operators:

void UnaryOp::backward_step() {
    switch (this->op_type) {
        case UnaryOpType::NEG:
            this->child->add_grad(-this->grad * Tensor::ones(this->child->dims));
            break;
        case UnaryOpType::RECIP:
            this->child->add_grad(this->grad * -pow(this->child, Tensor::from_scalar(-2.0)));
            break;
        case UnaryOpType::RELU:
            this->child->add_grad(this->grad * binilarize(this->child));
            break;
        case UnaryOpType::BIN:
            throw std::runtime_error("UnaryOp::backward_step cannot compute gradient for comparison binilarize()");
            break;
        case UnaryOpType::EXP:
            this->child->add_grad(this->grad * shared_from_this());
            break;
        default:
            throw std::domain_error("bad op_type");
    }
}

void BinaryOp::backward_step() {
    switch (this->op_type) {
        case BinaryOpType::ADD:
            if (product(this->leftChild->dims) > 1) {
                this->leftChild->add_grad(this->grad + Tensor::zeros(this->leftChild->dims));
            } else {
                this->leftChild->add_grad(sum(this->grad + Tensor::zeros(this->rightChild->dims)));
            }

            if (product(this->rightChild->dims) > 1) {
                this->rightChild->add_grad(this->grad + Tensor::zeros(this->rightChild->dims));
            } else {
                this->rightChild->add_grad(sum(this->grad + Tensor::zeros(this->leftChild->dims)));
            }
            break;
        case BinaryOpType::SUB:
            if (product(this->leftChild->dims) > 1) {
                this->leftChild->add_grad(this->grad + Tensor::zeros(this->leftChild->dims));
            } else {
                this->leftChild->add_grad(sum(this->grad + Tensor::zeros(this->rightChild->dims)));
            }

            if (product(this->rightChild->dims) > 1) {
                this->rightChild->add_grad(-this->grad + Tensor::zeros(this->rightChild->dims));
            } else {
                this->rightChild->add_grad(sum(-this->grad + Tensor::zeros(this->leftChild->dims)));
            }
            break;
        case BinaryOpType::MUL:
            if (product(this->leftChild->dims) > 1) {
                this->leftChild->add_grad(this->grad * this->rightChild);
            } else {
                this->leftChild->add_grad(sum(this->grad * this->rightChild));
            }

            if (product(this->rightChild->dims) > 1) {
                this->rightChild->add_grad(this->grad * this->leftChild * Tensor::ones(this->rightChild->dims));
            } else {
                this->rightChild->add_grad(sum(this->grad * this->leftChild));
            }
            break;
        case BinaryOpType::MATMUL:
            throw std::runtime_error("BinaryOp::backward_step not implemented yet");
            break;
        case BinaryOpType::POW:
            // x^n -> n*x^(n-1)
            if (product(this->leftChild->dims) > 1){
                this->leftChild->add_grad(this->grad * this->rightChild * pow(this->leftChild, (this->rightChild - Tensor::ones(this->rightChild->dims))));
            } else {
                this->leftChild->add_grad(sum(this->grad * this->rightChild * pow(this->leftChild, (this->rightChild - Tensor::ones(this->rightChild->dims)))));
            }

            // a^x -> ln(a) a^x
            if (product(this->rightChild->dims) > 1){
                this->rightChild->add_grad(this->grad * log(this->leftChild) * pow(this->leftChild, this->rightChild));
            } else {
                this->rightChild->add_grad(sum(this->grad * log(this->leftChild) * pow(this->leftChild, this->rightChild)));
            }
            break;
        case BinaryOpType::DIV:
            if (product(this->leftChild->dims) > 1) {
                this->leftChild->add_grad((this->grad * this->rightChild) * pow(this->rightChild, Tensor::from_scalar(-2.0)));
            } else {
                this->leftChild->add_grad(sum(this->grad * this->rightChild * pow(this->rightChild, Tensor::from_scalar(-2.0))));
            }

            if (product(this->rightChild->dims) > 1) {
                this->rightChild->add_grad((-this->grad * this->leftChild * Tensor::ones(this->rightChild->dims)) * pow(this->rightChild, Tensor::from_scalar(-2.0)));
            } else {
                this->rightChild->add_grad(sum(-this->grad * this->leftChild * pow(this->rightChild, Tensor::from_scalar(-2.0))));
            }
            break;
        default:
            throw std::domain_error("bad op_type");
    }
}

void ReshapeOp::backward_step() {
    throw std::runtime_error("ReshapeOp::backward_step not implemented yet");
}

void TransposeOp::backward_step() {
    throw std::runtime_error("TransposeOp::backward_step not implemented yet");
}

void ReductionOp::backward_step() {
    switch (this->op_type) {
        case ReductionOpType::SUM:
            this->child->add_grad(sum(this->grad));
            break;

        default:
            throw std::domain_error("bad op_type");
    }
}
