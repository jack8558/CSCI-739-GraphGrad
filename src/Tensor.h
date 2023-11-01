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
    virtual ~Tensor() {}

    // Other methods...

   protected:
    // Construct a Tensor without any data buffer.
    explicit Tensor(std::vector<size_t> dims) : dims(dims) {}

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
