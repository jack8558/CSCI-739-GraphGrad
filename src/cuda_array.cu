#include "cuda_array.h"

CudaArray::CudaArray(size_t length) : length(length) {
    cudaMalloc(&ptr, length * sizeof(scalar_t));
    assert_no_cuda_error();
}

CudaArray::~CudaArray() {
    if (ptr != nullptr) {
        cudaFree(ptr);
        assert_no_cuda_error();
    }
}

// Copy data from a CPU buffer into this CUDA array.
void CudaArray::copy_from_range(const scalar_t* src, size_t count) {
    assert(count <= length);
    cudaMemcpy(ptr, src, count * sizeof(scalar_t), cudaMemcpyDefault);
    assert_no_cuda_error();
}

// Copy the array into a new CPU vector.
std::vector<scalar_t> CudaArray::to_vector() const {
    std::vector<scalar_t> result(length);
    cudaMemcpy(result.data(), ptr, length * sizeof(scalar_t), cudaMemcpyDefault);
    assert_no_cuda_error();
    return result;
}
