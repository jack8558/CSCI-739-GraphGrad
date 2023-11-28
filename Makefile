# Define the source files and output file.
SOURCE_FILES := src/graphgrad.cu src/cuda_array.cu
HEADER_FILES := src/Tensor.h src/utils.h src/UnaryOp.cuh src/BinaryOp.h src/TransposeOp.cuh src/ReshapeOp.h src/ReductionOp.cuh src/python_data_to_tensor.h src/LRUMap.h src/Tensor_backward.cuh src/cuda_array.h src/globals.h
INPUT_FILES := $(SOURCE_FILES) $(HEADER_FILES)
OUTPUT_FILE := graphgrad$(shell python3-config --extension-suffix)

# The default target builds the output file.
all: $(OUTPUT_FILE)

# Build the output file.  
$(OUTPUT_FILE): $(INPUT_FILES)
	nvcc \
    -O3 -Xptxas -O3 -arch=native -shared \
    --compiler-options '-O3 -fPIC -march=native -fopenmp -Wall' \
    $(SOURCE_FILES) \
    -o graphgrad$$(python3-config --extension-suffix) \
    -std=c++17 \
    $$(python3 -m pybind11 --includes)

# The clean target removes the output file.
clean:
	rm -f $(OUTPUT_FILE)

.PHONY: clean all
