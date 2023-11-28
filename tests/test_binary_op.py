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
    return gg.rand([1])


@pytest.fixture(scope="class")
def gg_scalar2():
    return gg.rand([1, 1])


@pytest.fixture(scope="class")
def gg_scalar3():
    return gg.rand([1, 1, 1])


@pytest.fixture(scope="class")
def gg_tensor_10_5():
    return gg.rand([10, 5])


@pytest.fixture(scope="class")
def gg_tensor_10_1():
    return gg.rand([10, 1])

@pytest.fixture(scope="class")
def gg_tensor_1_10():
    return gg.rand([1, 10])


def to_torch_tensor(value, **kwargs):
    """Convert a gg tensor or a python scalar to a pytorch tensor."""
    if isinstance(value, gg.tensor):
        value = value.to_list()
    return torch.tensor(value, dtype=torch.float64, **kwargs)


class TestBinaryOP:
    # The different pointwise/scalar binary ops to test.
    # Each item is a tuple of (GraphGrad op, equivalent PyTorch op).
    # Each op has tests for all ways to call it from graphgrad to verify correct pybind setup
    # Note: Does not include matmul, which requires different checks
    BINARY_OPS = [
        # Addition
        (True, gg.add, torch.add),
        (True, lambda gg_tensor, gg_tensor2: gg_tensor + gg_tensor2, torch.add),
        (False, lambda gg_tensor, gg_tensor2: gg_tensor.add(gg_tensor2), torch.add),
        # Subtraction
        (True, gg.subtract, torch.subtract),
        (True, lambda gg_tensor, gg_tensor2: gg_tensor - gg_tensor2, torch.subtract),
        (False, lambda gg_tensor, gg_tensor2: gg_tensor.subtract(gg_tensor2), torch.subtract),
        # Multiplication
        (True, gg.mul, torch.mul),
        (True, lambda gg_tensor, gg_tensor2: gg_tensor * gg_tensor2, torch.mul),
        (False, lambda gg_tensor, gg_tensor2: gg_tensor.mul(gg_tensor2), torch.mul),
        # Division
        (True, gg.div, torch.div),
        (True, lambda gg_tensor, gg_tensor2: gg_tensor / gg_tensor2, torch.div),
        (False, lambda gg_tensor, gg_tensor2: gg_tensor.div(gg_tensor2), torch.div),
        # Power
        (True, gg.pow, torch.pow),
        (False, lambda gg_tensor, gg_tensor2: gg_tensor.pow(gg_tensor2), torch.pow),
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
        # scalar, scalar
        ("gg_scalar1", "gg_scalar1"),
        ("gg_scalar2", "gg_scalar2"),
        ("gg_scalar3", "gg_scalar3"),
        # tensor, Python scalar
        ("gg_tensor_5_10", 3.14),
        ("gg_tensor_10_10", 3.14),
        ("gg_tensor_50_100", 3.14),
        # Python scalar, tensor
        (3.14, "gg_tensor_5_10"),
        (3.14, "gg_tensor_10_10"),
        (3.14, "gg_tensor_50_100"),
    ]

    @pytest.mark.parametrize("scalar_lhs_ok, gg_func, torch_func", BINARY_OPS)
    @pytest.mark.parametrize("gg_left, gg_right", BINARY_INPUTS)
    def test_binary_op(
        self,
        gg_left,
        gg_right,
        gg_func,
        torch_func,
        scalar_lhs_ok,
        request,
    ):
        if isinstance(gg_left, str):
            gg_left = request.getfixturevalue(gg_left)
        elif not scalar_lhs_ok:
            return  # This op isn't supposed to be tested with a scalar LHS
        if isinstance(gg_right, str):
            gg_right = request.getfixturevalue(gg_right)
        gg_result = gg_func(gg_left, gg_right)
        torch_left = to_torch_tensor(gg_left)
        torch_right = to_torch_tensor(gg_right)
        torch_result = torch_func(torch_left, torch_right)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()

    @pytest.mark.parametrize("gg_op", [gg_op for _, gg_op, _ in BINARY_OPS])
    def test_binary_op_shape_mismatch_raises(self, gg_op):
        tensor1 = gg.rand([3, 4])
        tensor2 = gg.rand([4, 3])
        with pytest.raises(ValueError):
            gg_op(tensor1, tensor2)

    @pytest.mark.parametrize("scalar_lhs_ok, gg_func, torch_func", BINARY_OPS)
    @pytest.mark.parametrize("gg_left, gg_right", BINARY_INPUTS)
    def test_binary_op_backward(
        self,
        gg_left,
        gg_right,
        gg_func,
        torch_func,
        scalar_lhs_ok,
        request,
    ):
        if isinstance(gg_left, str):
            gg_left = gg.tensor(request.getfixturevalue(gg_left).to_list())
        elif not scalar_lhs_ok:
            return  # This op isn't supposed to be tested with a scalar LHS
        if isinstance(gg_right, str):
            gg_right = gg.tensor(request.getfixturevalue(gg_right).to_list())
        gg_result = gg_func(gg_left, gg_right)
        torch_left = to_torch_tensor(gg_left, requires_grad=True)
        torch_right = to_torch_tensor(gg_right, requires_grad=True)
        torch_result = torch_func(torch_left, torch_right)

        gg_result.sum().backward()
        torch_result.sum().backward()
        if isinstance(gg_left, gg.tensor):
            assert np.isclose(gg_left.grad.to_list(), torch_left.grad, rtol=1e-4).all()
        if isinstance(gg_right, gg.tensor):
            assert np.isclose(gg_right.grad.to_list(), torch_right.grad, rtol=1e-4).all()

    MATMUL_INPUTS = [
        # 1D scalar, 1D scalar
        ("gg_scalar1", "gg_scalar1"),

        # 1D scalar, 2D scalar
        ("gg_scalar1", "gg_scalar2"),

        # 2D scalar, 1D scalar
        ("gg_scalar2", "gg_scalar1"),

        # 2D scalar, 2D scalar
        ("gg_scalar2", "gg_scalar2"),

        # 1D scalar, 1D tensor
        ("gg_scalar1", "gg_tensor_1_10"),

        # 2D tensor, 1D scalar
        ("gg_tensor_10_1", "gg_scalar1"),

        # 2D tensor, 2D tensor
        ("gg_tensor_5_10", "gg_tensor_10_10"),
        ("gg_tensor_5_10", "gg_tensor_10_5"),
        ("gg_tensor_10_5", "gg_tensor_5_10"),
        ("gg_tensor_5_10", "gg_tensor_10_1"),
        ("gg_tensor_10_10", "gg_tensor_10_1"),
    ]

    @pytest.mark.parametrize("gg_left, gg_right", MATMUL_INPUTS)
    def test_matmul(self, gg_left, gg_right, request):
        gg_left, gg_right = request.getfixturevalue(gg_left), request.getfixturevalue(
            gg_right
        )
        torch_left = to_torch_tensor(gg_left)
        torch_right = to_torch_tensor(gg_right)
        torch_result = torch.matmul(torch_left, torch_right)

        for matmul_func in [gg.matmul, lambda a, b: a.matmul(b), lambda a, b: a @ b]:
            gg_result = matmul_func(gg_left, gg_right)
            assert gg_result.dims() == list(torch_result.size())
            assert np.isclose(
                gg_result.to_list(),
                torch_result,
                rtol=1e-4,
            ).all()

    @pytest.mark.parametrize("gg_left, gg_right", MATMUL_INPUTS)
    def test_matmul_backward(self, gg_left, gg_right, request):
        gg_left = gg.tensor(request.getfixturevalue(gg_left).to_list())
        gg_right = gg.tensor(request.getfixturevalue(gg_right).to_list())
        gg_result = gg.matmul(gg_left, gg_right)
        torch_left = to_torch_tensor(gg_left, requires_grad=True)
        torch_right = to_torch_tensor(gg_right, requires_grad=True)
        torch_result = torch.matmul(torch_left, torch_right)

        gg_result.sum().backward()
        torch_result.sum().backward()
        print("gg_left", gg_left.grad)
        print("gg_right", gg_right.grad)
        print("gg_left_tensor", torch_left.grad)
        print("gg_right_tensor", torch_right.grad)
        assert np.isclose(gg_left.grad.to_list(), torch_left.grad, rtol=1e-4).all()
        assert np.isclose(gg_right.grad.to_list(), torch_right.grad, rtol=1e-4).all()

    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            # non-2D inputs
            # ([6], [6]),
            # ([3], [3, 4]),
            # ([3, 4], [4]),
            # middle dimension doesn't match
            ([3, 4], [3, 4]),
            ([10, 32], [10, 64]),
            ([10, 32], [5, 64]),
        ],
    )
    def test_matmul_shape_mismatch_raises(self, shape1, shape2):
        tensor1 = gg.rand(shape1)
        tensor2 = gg.rand(shape2)
        with pytest.raises(ValueError):
            gg.matmul(tensor1, tensor2)
