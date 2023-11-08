import pytest
import torch
import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import graphgrad as gg
import numpy as np


@pytest.fixture(scope="class")
def gg_scalar1():
    return gg.Tensor.rand([1])


@pytest.fixture(scope="class")
def gg_scalar2():
    return gg.Tensor.rand([1, 1])


@pytest.fixture(scope="class")
def gg_scalar3():
    return gg.Tensor.rand([1, 1, 1])



class TestBinaryOP:
    # The different pointwise/scalar binary ops to test.
    # Each item is a tuple of (GraphGrad op, equivalent PyTorch op).
    # Each op has tests for all ways to call it from graphgrad to verify correct pybind setup
    # Note: Does not include matmul, which requires different checks
    BINARY_OPS = [
        # Addition
        (gg.add, torch.add),
        (lambda gg_tensor, gg_tensor2: gg_tensor.add(gg_tensor2), torch.add),

        # Subtraction
        (gg.subtract, torch.subtract),
        (lambda gg_tensor, gg_tensor2: gg_tensor.subtract(gg_tensor2), torch.subtract),

        # Multiplication
        # (gg.mul, torch.mul),
        # (lambda gg_tensor, gg_tensor2: gg_tensor.mul(gg_tensor2), torch.mul),

        # Power
        (gg.pow, torch.pow),
        (lambda gg_tensor, gg_tensor2: gg_tensor.pow(gg_tensor2), torch.pow),
    ]

    BINARY_INPUTS = [
        # tensor, tensor
        ("gg_tensor_5_10", "gg_tensor_5_10"),
        ("gg_tensor_10_10", "gg_tensor_10_10"),
        ("gg_tensor_50_100", "gg_tensor_50_100"),

        # tensor, scalar
        ("gg_tensor_5_10", "gg_scalar1"),
        ("gg_tensor_10_10", "gg_scalar2"),
        ("gg_tensor_50_100", "gg_scalar3"),

        # scalar, tensor
        ("gg_scalar1", "gg_tensor_5_10"),
        ("gg_scalar2", "gg_tensor_10_10"),
        ("gg_scalar3", "gg_tensor_50_100"),
    ]

    @pytest.mark.parametrize("gg_func, torch_func", BINARY_OPS)
    @pytest.mark.parametrize("gg_left, gg_right", BINARY_INPUTS)
    def test_elementwise_binary_op(
        self, 
        gg_left,
        gg_right,
        gg_func, 
        torch_func,
        request,
    ):
        gg_left, gg_right =  request.getfixturevalue(gg_left), request.getfixturevalue(gg_right)
        gg_result = gg_func(gg_left, gg_right)
        torch_left = torch.tensor(gg_left.to_list())
        torch_right = torch.tensor(gg_right.to_list())
        torch_result = torch_func(torch_left, torch_right)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()

    @pytest.mark.parametrize("gg_op", [gg_op for gg_op, _ in BINARY_OPS])
    def test_elementwise_binary_op_shape_mismatch_raises(self, gg_op):
        tensor1 = gg.Tensor.rand([3, 4])
        tensor2 = gg.Tensor.rand([4, 3])
        with pytest.raises(ValueError):
            gg_op(tensor1, tensor2)

    # Matmul tests separate?
    def test_matmul(self, gg_tensor, torch_tensor, gg_tensor3, torch_tensor3):
        print(gg.matmul(gg_tensor, gg_tensor3))
        assert np.isclose(
            gg.matmul(gg_tensor, gg_tensor3).to_list(),
            torch.matmul(torch_tensor, torch_tensor3).tolist(),
            rtol=1e-4,
        ).all()

    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            # non-2D inputs
            ([6], [6]),
            ([3], [3, 4]),
            ([3, 4], [4]),
            # middle dimension doesn't match
            ([3, 4], [3, 4]),
            ([10, 32], [10, 64]),
            ([10, 32], [5, 64]),
        ],
    )
    def test_matmul_shape_mismatch_raises(self, shape1, shape2):
        tensor1 = gg.Tensor.rand(shape1)
        tensor2 = gg.Tensor.rand(shape2)
        with pytest.raises(ValueError):
            gg.matmul(tensor1, tensor2)
