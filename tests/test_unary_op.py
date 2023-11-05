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


@pytest.mark.parametrize(
    "gg_func, pytorch_func",
    [
        [gg.neg, torch.neg],
        [lambda gg_tensor: -gg_tensor, torch.neg],
        [gg.reciprocal, torch.reciprocal],
        [gg.relu, torch.nn.functional.relu],
        [gg.binilarize, lambda torch_tensor: (torch_tensor > 0.0).double()],
        [gg.exp, torch.exp],
    ],
)
def test_unary_op(gg_tensor, torch_tensor, gg_func, pytorch_func):
    gg_result = gg_func(gg_tensor)
    pytorch_result = pytorch_func(torch_tensor)
    assert np.isclose(gg_result.to_list(), pytorch_result, rtol=1e-4).all()
