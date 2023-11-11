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

# TRANSPOSE_OP = [
#     # Transpose
#     (gg.transpose, torch.transpose),
#     (lambda gg_tensor: gg_tensor.transpose(), torch.transpose),
# ]

GG_TENSORS = [
        ("gg_tensor_5_10", [5,10], 0, 1),
        ("gg_tensor_10_10", [10,10], 0, 1),
        ("gg_tensor_50_100", [50,100],  0, 1),
        ("gg_tensor_50_50_50", [50,50,50], 0, 1),
        ("gg_tensor_50_100_200", [50,100,200], 0, 1),
        ("gg_tensor_100_20_200_30", [100,20,200,30], 0, 1),
    ]


class TestTransposeOp:
    # @pytest.mark.parametrize("gg_func, torch_func", TRANSPOSE_OP)
    @pytest.mark.parametrize("gg_tensor, dims, dim0, dim1", GG_TENSORS)
    def test_unary_op(self, gg_tensor, dims, dim0, dim1, request):
        gg_tensor = request.getfixturevalue(gg_tensor)

        gg_result = gg_tensor.transpose(dim1, dim0)
        torch_tensor = torch.tensor(gg_tensor.to_list(), dtype=torch.float64).view(dims)
        torch_result = torch.transpose(torch_tensor, dim0, dim1)
        # print(gg_tensor)
        # print()

        # # print(gg_tensor)
        # # print(torch_tensor)
        # print(gg_result.to_list())
        # print()
        # print(torch_result)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()