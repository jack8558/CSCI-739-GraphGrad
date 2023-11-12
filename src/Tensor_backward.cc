#include <ranges>
#include <unordered_set>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "BinaryOp.h"
#include "ReductionOp.h"
#include "ReshapeOp.h"
#include "Tensor.h"
#include "TransposeOp.h"
#include "UnaryOp.h"

// Collects all the nodes in the graph under the given root, in reverse topological order.
static void topo_sort_visit(Tensor* node, std::unordered_set<Tensor*>& visited_nodes, std::vector<Tensor*>& sorted_nodes) {
    if (visited_nodes.contains(node)) {
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
    for (Tensor* t : std::views::reverse(sorted_nodes)) {
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
            this->child->add_grad(-this->grad);
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
    throw std::runtime_error("BinaryOp::backward_step not implemented yet");
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
            this->child->add_grad(this->grad);
            break;

        default:
            throw std::domain_error("bad op_type");
    }
}
