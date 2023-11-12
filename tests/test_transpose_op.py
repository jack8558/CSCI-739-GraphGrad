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
def gg_tensor_50_50_50():
    return gg.Tensor.rand([50, 50, 50])


@pytest.fixture(scope="class")
def gg_tensor_50_100_200():
    return gg.Tensor.rand([50, 100, 200])


@pytest.fixture(scope="class")
def gg_tensor_100_20_200_30():
    return gg.Tensor.rand([100, 20, 200, 30])


class TestTransposeOp:
    GG_TENSORS = [
        ("gg_tensor_5_10", [5, 10], 0, 1),
        ("gg_tensor_10_10", [10, 10], 0, 1),
        ("gg_tensor_50_100", [50, 100], 0, 1),
        ("gg_tensor_50_50_50", [50, 50, 50], 0, 1),
        ("gg_tensor_50_100_200", [50, 100, 200], 0, 1),
        ("gg_tensor_100_20_200_30", [100, 20, 200, 30], 0, 1),
    ]

    @pytest.mark.parametrize("gg_tensor, dims, dim0, dim1", GG_TENSORS)
    def test_transpose_op(self, gg_tensor, dims, dim0, dim1, request):
        gg_tensor = request.getfixturevalue(gg_tensor)

        torch_tensor = torch.tensor(gg_tensor.to_list(), dtype=torch.float64).view(dims)
        torch_result = torch.transpose(torch_tensor, dim0, dim1)

        gg_result = gg_tensor.transpose(dim1, dim0)
        assert gg_result.dims() == list(torch_result.size())
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()

        gg_result = gg.transpose(gg_tensor, dim0, dim1)
        assert gg_result.dims() == list(torch_result.size())
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()
