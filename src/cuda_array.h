#pragma once

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "globals.h"

#ifdef __CUDACC__
inline void assert_no_cuda_error() {
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + std::to_string(status) + ": " + cudaGetErrorString(status));
    }
}
#endif

template <typename T>
struct CudaArrayRef {
    T* ptr;
    size_t length;

    CudaArrayRef(T* ptr, size_t length) : ptr(ptr), length(length) {}
};

struct CudaArray {
    void* ptr;
    size_t length_bytes;

    // Allocate an empty array with the given number of elements.
    explicit CudaArray(size_t length_bytes);

    // Copy a CPU vector into a CUDA array.
    explicit CudaArray(const std::vector<scalar_t>& data) : CudaArray(data.data(), data.size() * sizeof(scalar_t)) {}

    // Copy a CPU array into a CUDA array.
    explicit CudaArray(const void* data, size_t length_bytes) : CudaArray(length_bytes) {
        copy_from_range(data, length_bytes);
    }

    // CudaArray is non-copyable.
    CudaArray(CudaArray const&) = delete;
    void operator=(CudaArray const& x) = delete;

    // CudaArray is moveable.
    // CudaArray(CudaArray&& other) noexcept;

    ~CudaArray();

    // Copy data from a CPU vector into this CUDA array.
    template <typename T>
    void copy_from(const std::vector<T>& data) {
        assert(data.size() * sizeof(T) == length_bytes);
        copy_from_range(data.data(), length_bytes);
    }

    // Copy data from a CPU buffer into this CUDA array.
    void copy_from_range(const void* src, size_t count);

    // Copy the array into a new CPU vector.
    std::vector<scalar_t> to_vector() const;

    // Implicitly convert CudaArray to CudaArrayRef, for passing to kernels.
    template <typename T>
    operator CudaArrayRef<T>() {
        assert(length_bytes % sizeof(T) == 0);
        return CudaArrayRef{static_cast<T*>(ptr), length_bytes / sizeof(T)};
    }
};
