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


class TestUnaryOp:
    # The different unary ops to test.
    # Each item is a tuple of (GraphGrad op, equivalent PyTorch op).
    # Each op has tests for all ways to call it from graphgrad to verify correct pybind setup
    UNARY_OPS = [
        # Negation
        (gg.neg, torch.neg),
        (lambda gg_tensor: -gg_tensor, torch.neg),
        (lambda gg_tensor: gg_tensor.neg(), torch.neg),

        # Reciprocal
        (gg.reciprocal, torch.reciprocal),
        (lambda gg_tensor: gg_tensor.reciprocal(), torch.reciprocal),

        # ReLU
        (gg.relu, torch.nn.functional.relu),
        (lambda gg_tensor: gg_tensor.relu(), torch.nn.functional.relu),

        # Binilarize
        (gg.binilarize, lambda torch_tensor: (torch_tensor > 0.0).double()),
        (lambda gg_tensor: gg_tensor.binilarize(), lambda torch_tensor: (torch_tensor > 0.0).double()),

        # Exponential
        (gg.exp, torch.exp),
        (lambda gg_tensor: gg_tensor.exp(), torch.exp),
    ]

    GG_TENSORS = [
        "gg_tensor_5_10",
        "gg_tensor_10_10",
        "gg_tensor_50_100"
    ]


    @pytest.mark.parametrize("gg_func, torch_func", UNARY_OPS)
    @pytest.mark.parametrize("gg_tensor", GG_TENSORS)
    def test_unary_op(self, gg_tensor, gg_func, torch_func, request):
        gg_tensor = request.getfixturevalue(gg_tensor)
        gg_result = gg_func(gg_tensor)
        torch_tensor = torch.tensor(gg_tensor.to_list(), dtype=torch.float64)
        torch_result = torch_func(torch_tensor)
        assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()
