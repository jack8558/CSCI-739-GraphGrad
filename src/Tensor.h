#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

#include "utils.h"
#include "globals.h"
#include "LRUMap.h"
#include "cuda_array.h"

class Tensor : public std::enable_shared_from_this<Tensor> {
   public:
    // Construct a Tensor without any data buffer.
    // This should only be called by subclasses.
    explicit Tensor(std::vector<size_t> dims) : on_gpu(use_gpu), dims(std::move(dims)) {}

    // Construct a Tensor with the given data buffer.
    // If the global `use_gpu` flag is `true`, then the data will be copied into a `CudaArray`.
    explicit Tensor(std::vector<size_t> dims, std::vector<scalar_t> data) : on_gpu(use_gpu), dims(std::move(dims)) {
        assert(product(this->dims) == data.size());

        this->hashValue = tensor_hash(data);

        if (this->on_gpu) {
            this->data.emplace(std::in_place_type<CudaArray>, data);
        } else {
            this->data = std::move(data);
        }
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

    // Create a new tensor containing the given scalar value.
    static std::shared_ptr<Tensor> from_scalar(scalar_t value) {
        return std::make_shared<Tensor>(std::vector<size_t>{}, std::vector<scalar_t>{value});
    }

    // Create a new tensor filled with random values in the range [0, 1).
    static std::shared_ptr<Tensor> rand(std::vector<size_t> dims) {
        // Allocate a data buffer.
        std::vector<scalar_t> data(product(dims));

        // Fill the data with random values.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        for (scalar_t& value : data) {
            value = dis(gen);
        }

        return std::make_shared<Tensor>(dims, data);
    }

    // Create a new tensor filled with ones.
    static std::shared_ptr<Tensor> ones(std::vector<size_t> dims) {
        return std::make_shared<Tensor>(dims, std::vector<scalar_t>(product(dims), scalar_t(1)));
    }

    // Create a new tensor filled with zeros.
    static std::shared_ptr<Tensor> zeros(std::vector<size_t> dims) {
        return std::make_shared<Tensor>(dims, std::vector<scalar_t>(product(dims), scalar_t(0)));
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
    size_t tensor_hash(const std::vector<scalar_t>& data){
        size_t hashValue = 0;
        hash_combine(hashValue, vector_hash(this->dims));
        hash_combine(hashValue, vector_hash(data));
        return hashValue;
    }

    // Evalutates this tensor node (if it has not yet been evaluated), then returns the pointer to
    // the resulting data buffer.
    virtual const scalar_t* eval() {
        if (!this->data) {
            // Return cached data if it exists.
            auto result = Tensor::lruMap.get(this->hashValue);
            if (result.has_value()) {
                // The key was found
                assert((*result)->dims == this->dims);
                return (*result)->eval();
            }

            // Otherwise, compute this Tensor's data.
            this->compute_data();

            // Add it to the hashmap.
            Tensor::lruMap.insert(this->hashValue, this->shared_from_this());
        }

        // Return the pointer to the now-populated data buffer.
        if (this->data) {
            if (this->on_gpu) {
                return std::get<CudaArray>(*this->data).ptr;
            } else {
                return std::get<std::vector<scalar_t>>(*this->data).data();
            }
        } else {
            throw std::runtime_error("called eval() on a Tensor with no data");
        }
    }

    // Computes the data field for this tensor. Subclasses must override this.
    virtual void compute_data() {
        // Leaf Tensors have nothing to do here.
    }

    // Evaluates this tensor and returns a CPU buffer containing a copy of the evaluated data.
    std::vector<scalar_t> eval_to_cpu() {
        size_t size = product(this->dims);
        const scalar_t* data = this->eval();
        if (this->on_gpu) {
            std::vector<scalar_t> result(size);
            cudaMemcpy(result.data(), data, size * sizeof(scalar_t), cudaMemcpyDefault);
            assert_no_cuda_error();
            return result;
        } else {
            return std::vector<scalar_t>(data, data + size);
        }
    }


    // The Tensor's dimensions.
    std::vector<size_t> dims;

    // The Tensor's gradient.
    // May be null if this Tensor has no gradient assigned.
    std::shared_ptr<Tensor> grad = nullptr;

    // Static hashmap for common subextpression. Removes least used element if exceeds capacity.
    inline static LRUMap<size_t, std::shared_ptr<Tensor>> lruMap{200};

    // hashValue of tensor
    size_t hashValue;

    // Allocate the data buffer for this tensor and return a reference to it.
    // The buffer size is equal to the product of dims.
    // Throws an exception if this tensor already has data allocated.
    std::vector<scalar_t>& allocate_data_cpu() {
        if (this->data) {
            throw std::runtime_error("called allocate_data_cpu() on a Tensor with data already allocated");
        }
        if (this->on_gpu) {
            throw std::runtime_error("called allocate_data_cpu() on a Tensor with on_gpu=true");
        }
        this->data.emplace(std::vector<scalar_t>(product(this->dims)));
        return std::get<std::vector<scalar_t>>(*this->data);
    }

    // Allocate the data buffer for this tensor and return a reference to it.
    // The buffer size is equal to the product of dims.
    // Throws an exception if this tensor already has data allocated.
    CudaArray& allocate_data_gpu() {
        if (this->data) {
            throw std::runtime_error("called allocate_data_gpu() on a Tensor with data already allocated");
        }
        if (!this->on_gpu) {
            throw std::runtime_error("called allocate_data_gpu() on a Tensor with on_gpu=false");
        }
        this->data.emplace(CudaArray(product(this->dims)));
        return std::get<CudaArray>(*this->data);
    }

    // The Tensor's data buffer.
    // May be empty if this Tensor has no cached data.
    std::optional<std::variant<std::vector<scalar_t>, CudaArray>> data = std::nullopt;

    // Whether the Tensor's data/computation will be on GPU (versus CPU).
    bool on_gpu;
};
