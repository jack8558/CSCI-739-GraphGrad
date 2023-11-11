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
def gg_tensor_5_10():
    return gg.Tensor.rand([5, 10])


@pytest.fixture(scope="session")
def gg_tensor_10_10():
    return gg.Tensor.rand([10, 10])


@pytest.fixture(scope="session")
def gg_tensor_50_100():
    return gg.Tensor.rand([50, 100])

@pytest.fixture(scope="session")
def gg_tensor_50_50_50():
    return gg.Tensor.rand([50, 50, 50])

@pytest.fixture(scope="session")
def gg_tensor_50_100_200():
    return gg.Tensor.rand([50, 100, 200])

@pytest.fixture(scope="session")
def gg_tensor_100_20_200_30():
    return gg.Tensor.rand([100, 20, 200, 30])
