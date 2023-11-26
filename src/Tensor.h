#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "utils.h"
#include "LRUMap.h"

using scalar_t = double;

class Tensor : public std::enable_shared_from_this<Tensor> {
   public:
    // Construct a Tensor without any data buffer.
    // This should only be called by subclasses.
    explicit Tensor(std::vector<size_t> dims) : dims(std::move(dims)) {
        this->hashValue = tensor_hash();
    }

    // Construct a Tensor with the given data buffer.
    explicit Tensor(std::vector<size_t> dims, std::vector<scalar_t> data) : dims(std::move(dims)), data(std::move(data)) {
        assert(product(this->dims) == this->data->size());
        this->hashValue = tensor_hash();
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

    // Equality operator for Tensor
    bool operator==(const Tensor& other) const {
        return this->dims == other.dims && this->data.value() == other.data.value();
    }

    // Factory functions:

    // Create a new tensor containing the given scalar value.
    static std::shared_ptr<Tensor> from_scalar(scalar_t value) {
        return std::make_shared<Tensor>(std::vector<size_t>{}, std::vector<scalar_t>{value});
    }

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

    // Create a new tensor filled with ones.
    static std::shared_ptr<Tensor> ones(std::vector<size_t> dims) {
        // Allocate a new tensor with the given dims.
        auto result = std::make_shared<Tensor>(dims);
        auto& data = result->allocate_data();

        // Fill the data with ones.
        for (scalar_t& value : data) {
            value = (scalar_t)1;
        }

        return result;
    }

    // Create a new tensor filled with zeros.
    static std::shared_ptr<Tensor> zeros(std::vector<size_t> dims) {
        // Allocate a new tensor with the given dims.
        auto result = std::make_shared<Tensor>(dims);
        auto& data = result->allocate_data();

        // Fill the data with zeros.
        for (scalar_t& value : data) {
            value = (scalar_t)0;
        }

        return result;
    }

    // Automatic differentiation:

    // For all descendant nodes `d` of this tensor, assigns the gradient of this tensor with respect
    // to `d` to `d.grad`. The resulting `grad` tensors will be lazily-evaluated graph nodes. Any
    // previous `grad`s will be overwritten.
    //
    // Throws an error if this tensor is not a scalar.
    void backward();  // Implementation in Tensor_backward.cc

    // Returns the direct child nodes of this tensor in the graph.
    virtual std::vector<Tensor*> get_children() { return {}; }

    // Propagates `this->grad` backward to the direct children of this tensor.
    // Client code should not call this; use `backward()` instead.
    //
    // Requires / can assume that `this->grad` is not null.
    virtual void backward_step() {}

    // Adds the given tensor to `this->grad`, or assigns it if `this->grad` is null.
    void add_grad(std::shared_ptr<Tensor> grad);

    // Other methods:

    // Helper function to combine hash values
    // Used to combine hashvalue and object
    template <typename T>
    static void hash_combine(size_t& seed, const T& val) {
        seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Used to combine two hashvalue
    static void hash_combine(size_t& seed, size_t val) {
        seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Compute hash of vector
    template <typename T>
    size_t vector_hash(const std::vector<T>& v) {
        size_t hashValue = 0;
        for (const auto& element : v) {
            hash_combine(hashValue, std::hash<T>{}(element));
        }
        return hashValue;
    }

    // Compute hash for Tensor class
    size_t tensor_hash(){
        size_t hashValue = 0;
        hash_combine(hashValue, vector_hash(this->dims));
        if (this->data){
            hash_combine(hashValue, vector_hash(this->data.value()));
        }
        return hashValue;
    }

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


    // The Tensor's dimensions.
    std::vector<size_t> dims;

    // The Tensor's gradient.
    // May be null if this Tensor has no gradient assigned.
    std::shared_ptr<Tensor> grad = nullptr;

    // Static hashmap for common subextpression. Removes least used element if exceeds capacity.
    inline static LRUMap<size_t, std::vector<double>> lruMap{1000};

    // hashValue of tensor
    size_t hashValue;

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
    std::optional<std::vector<scalar_t>> data = std::nullopt;
};
