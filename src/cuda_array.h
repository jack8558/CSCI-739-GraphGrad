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

struct CudaArrayRef {
    scalar_t* ptr;
    size_t length;

    CudaArrayRef(scalar_t* ptr, size_t length) : ptr(ptr), length(length) {}
};

struct CudaArray {
    scalar_t* ptr;
    size_t length;

    // Allocate an empty array with the given number of elements.
    explicit CudaArray(size_t length);

    // Copy a CPU vector into a CUDA array.
    explicit CudaArray(const std::vector<scalar_t>& data) : CudaArray(data.data(), data.size()) {}

    // Copy a CPU array into a CUDA array.
    explicit CudaArray(const scalar_t* data, size_t length) : CudaArray(length) {
        copy_from_range(data, length);
    }

    // CudaArray is non-copyable.
    CudaArray(CudaArray const&) = delete;
    void operator=(CudaArray const& x) = delete;

    // CudaArray is moveable.
    CudaArray(CudaArray&& other) noexcept : ptr(other.ptr), length(other.length) {
        other.ptr = nullptr;
    }

    ~CudaArray();

    // Copy data from a CPU buffer into this CUDA array.
    void copy_from_range(const scalar_t* src, size_t count);

    // Copy the array into a new CPU vector.
    std::vector<scalar_t> to_vector() const;

    // Implicitly convert CudaArray to CudaArrayRef, for passing to kernels.
    operator CudaArrayRef() {
        return CudaArrayRef{ptr, length};
    }
};
