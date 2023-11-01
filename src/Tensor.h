#pragma once

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "utils.h"

using scalar_t = double;

class Tensor {
   public:
    // Construct a Tensor without any data buffer.
    explicit Tensor(std::vector<size_t> dims) : dims(dims) {}

    // Delete the copy constructor and copy assignment operator so Tensors won't
    // be implicitly copied.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Destructor.
    virtual ~Tensor() {}

    // Factory functions:

    // Create a new tensor filled with random values in the range [0, 1).
    static std::shared_ptr<Tensor> rand(std::vector<size_t> dims) {
        auto result = std::make_shared<Tensor>(dims);
        result->data.resize(product(dims));

        // Fill the data with random values.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        for (scalar_t& value : result->data) {
            value = dis(gen);
        }

        return result;
    }

    // Other methods:

    // Converts the tensor to a human-readable string.
    std::string to_string() const {
        std::string result = "<Tensor: dims=";
        result += vector_to_string(this->dims);
        result += ", data=";
        result += vector_to_string(this->data);
        result += ">";
        return result;
    }

   protected:
    // Move constructor.
    Tensor(Tensor&& other) noexcept = default;

    // Move assignment operator.
    Tensor& operator=(Tensor&& other) noexcept = delete;

   private:
    // The Tensor's dimensions.
    std::vector<size_t> dims;
    // The Tensor's data buffer.
    // May be empty if this Tensor has no cached data.
    std::vector<scalar_t> data;
};
