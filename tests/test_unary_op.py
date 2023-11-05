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


# The different unary ops to test.
# Each item is a tuple of (GraphGrad op, equivalent PyTorch op).
UNARY_OPS = [
    (gg.neg, torch.neg),
    (lambda gg_tensor: -gg_tensor, torch.neg),
    (gg.reciprocal, torch.reciprocal),
    (gg.relu, torch.nn.functional.relu),
    (gg.binilarize, lambda torch_tensor: (torch_tensor > 0.0).double()),
    (gg.exp, torch.exp),
]


@pytest.mark.parametrize("gg_func, torch_func", UNARY_OPS)
def test_unary_op(gg_tensor, torch_tensor, gg_func, torch_func):
    gg_result = gg_func(gg_tensor)
    torch_result = torch_func(torch_tensor)
    assert np.isclose(gg_result.to_list(), torch_result, rtol=1e-4).all()
