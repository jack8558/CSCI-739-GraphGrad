#pragma once

#include <cstdlib>
#include <string>
#include <vector>

// Returns the product of the given vector of integers.
// If the vector is empty, returns 1.
static size_t product(const std::vector<size_t>& dims) {
    size_t product = 1;
    for (auto d : dims) {
        product *= d;
    }
    return product;
}

// Converts a vector of values to a string, calling to_string on each element.
template <typename T>
static std::string vector_to_string(const std::vector<T>& vec) {
    std::string result = "[";

    using std::to_string;
    for (size_t i = 0; i < vec.size(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += to_string(vec[i]);
    }

    result += "]";
    return result;
}
