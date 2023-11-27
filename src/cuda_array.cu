#include "cuda_array.h"

CudaArray::CudaArray(size_t length_bytes) : length_bytes(length_bytes) {
    cudaMalloc(&ptr, length_bytes);
    assert_no_cuda_error();
}

CudaArray::~CudaArray() {
    cudaFree(ptr);
    assert_no_cuda_error();
}

// Copy data from a CPU buffer into this CUDA array.
void CudaArray::copy_from_range(const void* src, size_t count) {
    assert(count <= length_bytes);
    cudaMemcpy(ptr, src, count, cudaMemcpyDefault);
    assert_no_cuda_error();
}

// Copy the array into a new CPU vector.
std::vector<scalar_t> CudaArray::to_vector() const {
    assert(length_bytes % sizeof(scalar_t) == 0);
    std::vector<scalar_t> result(length_bytes / sizeof(scalar_t));
    cudaMemcpy(result.data(), ptr, length_bytes, cudaMemcpyDefault);
    assert_no_cuda_error();
    return result;
}
