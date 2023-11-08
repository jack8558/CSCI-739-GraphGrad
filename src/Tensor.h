#pragma once

#include <cstdlib>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

using scalar_t = double;

class Tensor {
   public:
    // Construct a Tensor without any data buffer.
    // This should only be called by subclasses.
    explicit Tensor(std::vector<size_t> dims) : dims(std::move(dims)) {}

    // Construct a Tensor with the given data buffer.
    explicit Tensor(std::vector<size_t> dims, std::vector<scalar_t> data) : dims(std::move(dims)), data(std::move(data)) {
        assert(product(this->dims) == this->data->size());
    }

    // Delete the copy constructor and copy assignment operator so Tensors won't
    // be implicitly copied.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor.
    Tensor(Tensor&& other) noexcept = default;

    // Move assignment operator.
    Tensor& operator=(Tensor&& other) noexcept = delete;

    // Destructor.
    virtual ~Tensor() {}

    // Factory functions:

    // Create a new tensor filled with random values in the range [0, 1).
    static std::shared_ptr<Tensor> rand(std::vector<size_t> dims) {
        // Allocate a new tensor with the given dims.
        auto result = std::make_shared<Tensor>(dims);
        auto& data = result->allocate_data();

        // Fill the data with random values.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        for (scalar_t& value : data) {
            value = dis(gen);
        }

        return result;
    }

    // Other methods:

    // Evalutates this tensor node (if it has not yet been evaluated), then returns the pointer to
    // the resulting data buffer.
    virtual const scalar_t* eval() {
        if (this->data) {
            return this->data->data();
        } else {
            throw std::runtime_error("called eval() on a Tensor with no data");
        }
    }

    // Converts the tensor to a human-readable string.
    virtual std::string to_string() const {
        std::string result = "<Tensor: dims=";
        result += vector_to_string(this->dims);
        if (this->data) {
            result += ", data=";
            result += vector_to_string(*this->data);
        }
        result += ">";
        return result;
    }

    std::shared_ptr<Tensor> reshape(std::vector<size_t> new_dims) {
        if(product(this->dims) != product(new_dims))
            throw std::runtime_error("Mismatched dims in reshape");

        // Allocate a new tensor with the given dims.
        auto result = std::make_shared<Tensor>(new_dims);
        result->allocate_data();
        this->eval();
        result->data = this->data;

        return result;
    }

    // The Tensor's dimensions.
    std::vector<size_t> dims;

   protected:
    // Allocate the data buffer for this tensor and return a reference to it.
    // The buffer size is equal to the product of dims.
    // Throws an exception if this tensor already has data allocated.
    std::vector<scalar_t>& allocate_data() {
        if (this->data) {
            throw std::runtime_error("called allocate_data() on a Tensor with data already allocated");
        }
        this->data.emplace(product(this->dims));
        return *this->data;
    }

    // The Tensor's data buffer.
    // May be empty if this Tensor has no cached data.
    std::optional<std::vector<scalar_t>> data;
};
