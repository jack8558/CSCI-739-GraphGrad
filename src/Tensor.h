#pragma once

#include <cstdlib>
#include <vector>

using scalar_t = double;

class Tensor {
   public:
    // Delete the copy constructor and copy assignment operator so Tensors won't
    // be implicitly copied.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Destructor.
    virtual ~Tensor() {
        if (data != nullptr) {
            // TODO: free data
        }
    }

    // Other methods...

   protected:
    // Construct a Tensor without any data buffer.
    explicit Tensor(std::vector<size_t> dims) : data(nullptr), dims(dims) {}

    // Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), dims(std::move(other.dims)) {
        other.data = nullptr;  // Ensure that 'other' won't delete the data
    }

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept = delete;

   private:
    scalar_t* data;
    std::vector<size_t> dims;
};
