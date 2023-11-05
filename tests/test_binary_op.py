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
def gg_tensor2():
    tensor = gg.Tensor.rand([5, 10])
    return tensor


@pytest.fixture(scope="class")
def torch_tensor2(gg_tensor2):
    tensor = torch.tensor(gg_tensor2.to_list())
    assert np.isclose(gg_tensor2.to_list(), tensor.tolist(), rtol=1e-4).all()
    return tensor


@pytest.fixture(scope="class")
def gg_tensor3():
    tensor = gg.Tensor.rand([10, 5])
    return tensor


@pytest.fixture(scope="class")
def torch_tensor3(gg_tensor3):
    tensor = torch.tensor(gg_tensor3.to_list())
    assert np.isclose(gg_tensor3.to_list(), tensor.tolist(), rtol=1e-4).all()
    return tensor


class TestBinaryOP:
    # The different pointwise binary ops to test.
    # Each item is a tuple of (GraphGrad op, equivalent PyTorch op).
    BINARY_OPS = [
        (gg.add, torch.add),
        (gg.subtract, torch.subtract),
        # (gg.mult, torch.mul),
        (gg.elementwise_mult, torch.mul),
        # (gg.pow, torch.pow),
    ]

    @pytest.mark.parametrize("gg_func, torch_func", BINARY_OPS)
    def test_pointwise_binary_op(
        self, gg_tensor, torch_tensor, gg_tensor2, torch_tensor2, gg_func, torch_func
    ):
        gg_result = gg_func(gg_tensor, gg_tensor2)
        torch_result = torch_func(torch_tensor, torch_tensor2)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()

    @pytest.mark.parametrize("gg_op", [gg_op for gg_op, _ in BINARY_OPS])
    def test_pointwise_binary_op_shape_mismatch_raises(self, gg_op):
        tensor1 = gg.Tensor.rand([3, 4])
        tensor2 = gg.Tensor.rand([4, 3])
        with pytest.raises(ValueError):
            gg_op(tensor1, tensor2)

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
