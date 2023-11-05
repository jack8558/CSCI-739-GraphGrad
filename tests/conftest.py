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


@pytest.fixture(scope="session")
def gg_tensor():
    tensor = gg.Tensor.rand([5, 10])
    return tensor


@pytest.fixture(scope="session")
def torch_tensor(gg_tensor):
    tensor = torch.tensor(gg_tensor.to_list())
    assert np.isclose(gg_tensor.to_list(), tensor.tolist(), rtol=1e-4).all()
    return tensor
