#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "Tensor.h"

// Attempts to cast a Python object to the given type.
template <typename T>
std::optional<T> try_cast(py::object obj) {
    try {
        return obj.cast<T>();
    } catch (py::type_error&) {
        return std::nullopt;
    }
}

void _convert_data_helper(py::object py_data, size_t depth, std::vector<size_t>& dims, std::vector<scalar_t>& data) {
    const size_t UNKNOWN_SIZE_SENTINEL = std::numeric_limits<size_t>::max();

    if (py::isinstance<py::str>(py_data)) {
        throw py::type_error("expected a value, but got a str instance");
    }
    if (py::isinstance<py::bytes>(py_data)) {
        throw py::type_error("expected a value, but got a bytes instance");
    }

    if (auto iterable = try_cast<py::iterable>(py_data)) {
        if (!dims.empty() && depth >= dims.size()) {
            // Given the depth, we expected a scalar, but this is an iterable.
            auto py_data_repr = py::repr(py_data).cast<std::string>();
            throw py::type_error("expected a scalar value, but got " + py_data_repr);
        }

        // Recurse on all the items in the iterable, while counting the number of items.
        size_t num_items = 0;
        for (auto& item : *iterable) {
            _convert_data_helper(item.cast<py::object>(), depth + 1, dims, data);
            num_items++;
        }

        if (dims.empty()) {
            // We went through the whole iterable without filling out the dims.
            // This is only possible if the iterable had no items.
            assert(num_items == 0);
            // Fill out the dims, giving the last dim size 0.
            while (dims.size() < depth) {
                dims.push_back(UNKNOWN_SIZE_SENTINEL);
            }
            dims.push_back(0);
        } else {
            assert(dims.size() > depth);

            if (dims[depth] == UNKNOWN_SIZE_SENTINEL) {
                // This is the first iterable we have finished at this depth.
                // Record the dim size.
                dims[depth] = num_items;
            } else {
                // This is not the first iterable we have finished at this depth.
                // Verify that the iterable had the correct length.
                if (num_items != dims[depth]) {
                    std::string error_message = "expected an iterable of length ";
                    error_message += std::to_string(dims[depth]);
                    error_message += ", but got an iterable of length ";
                    error_message += std::to_string(num_items);
                    throw py::value_error(error_message);
                }
            }
        }
    } else {
        // The value is not iterable, so attempt to convert it to a number.
        // This will raise TypeError if the cast fails.
        scalar_t value;
        try {
            value = py_data.cast<scalar_t>();
        } catch (py::cast_error&) {
            auto py_data_repr = py::repr(py_data).cast<std::string>();
            throw py::type_error("expected a scalar value, but got " + py_data_repr);
        }

        if (dims.empty()) {
            // This is the first scalar encountered. ndim == depth.
            // Fill out the dims vector with placeholders.
            while (dims.size() < depth) {
                dims.push_back(UNKNOWN_SIZE_SENTINEL);
            }
        } else {
            // This is not the first scalar encountered.
            // Verify that this scalar is at the correct depth.
            if (depth != dims.size()) {
                throw py::value_error("expected an iterable, but got scalar " + std::to_string(value));
            }
        }

        data.push_back(value);
    }
}

// Creates a Tensor from a NumPy array.
// Special-cased for efficiency.
Tensor numpy_array_to_tensor(py::array_t<scalar_t, py::array::c_style | py::array::forcecast> array) {
    std::vector<size_t> dims(array.ndim());
    for (size_t i = 0; i < dims.size(); i++) {
        dims[i] = array.shape(i);
    }

    std::vector<scalar_t> data(array.size());
    std::copy_n(array.data(), array.size(), data.begin());

    return Tensor(std::move(dims), std::move(data));
}

// Creates a Tensor from a (nested) Python iterable of data values.
//
// Throws ValueError if the data isn't rectangular.
// Throws TypeError if the data values aren't numbers.
Tensor python_data_to_tensor(py::object py_data) {
    std::vector<size_t> dims;
    std::vector<scalar_t> data;

    _convert_data_helper(py_data, 0, dims, data);

    return Tensor(std::move(dims), std::move(data));
}
