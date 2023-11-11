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

GG_TENSORS = [
        ("gg_tensor_5_10", [50]),
        ("gg_tensor_5_10", [2, 25]),

        ("gg_tensor_10_10", [10, 2, 5]),
        ("gg_tensor_10_10", [50, 2]),

        ("gg_tensor_50_100", [50,100]),
        ("gg_tensor_50_100", [50,10, 10]),
    ]


class TestReshapeOp:
    @pytest.mark.parametrize("gg_tensor, dims, ", GG_TENSORS)
    def test_unary_op(self, gg_tensor, dims, request):
        gg_tensor = request.getfixturevalue(gg_tensor)

        torch_tensor = torch.tensor(gg_tensor.to_list(), dtype=torch.float64).view(dims)
        torch_result = torch.reshape(torch_tensor, tuple(dims))

        gg_result = gg_tensor.reshape(dims)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()

        gg_result = gg.reshape(gg_tensor, dims)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()

