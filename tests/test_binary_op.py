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


@pytest.fixture(scope='class')
def gg_tensor2():
    tensor = gg.Tensor.rand([5, 10])
    return tensor

@pytest.fixture(scope='class')
def torch_tensor2(gg_tensor2):
    tensor = torch.tensor(gg_tensor2.to_list())
    assert np.isclose(gg_tensor2.to_list(), tensor.tolist(), rtol=1e-4).all()
    return tensor

@pytest.fixture(scope='class')
def gg_tensor3():
    tensor = gg.Tensor.rand([10, 5])
    return tensor

@pytest.fixture(scope='class')
def torch_tensor3(gg_tensor3):
    tensor = torch.tensor(gg_tensor3.to_list())
    assert np.isclose(gg_tensor3.to_list(), tensor.tolist(), rtol=1e-4).all()
    return tensor


class TestBinaryOP:

    def test_add(self, gg_tensor, torch_tensor, gg_tensor2, torch_tensor2):
        assert np.isclose(gg.add(gg_tensor, gg_tensor2).to_list(),
                          torch.add(torch_tensor, torch_tensor2).tolist(), rtol=1e-4).all()

    def test_subtract(self, gg_tensor, torch_tensor, gg_tensor2, torch_tensor2):
        assert np.isclose(gg.subtract(gg_tensor, gg_tensor2).to_list(),
                          torch.subtract(torch_tensor, torch_tensor2).tolist(), rtol=1e-4).all()

    # def test_mult(self, gg_tensor, torch_tensor):
    #     assert np.isclose(gg.mult(gg_tensor, 7).to_list(),
    #                       torch.mul(torch_tensor, 7).tolist(), rtol=1e-4).all()

    def test_elementwise_mult(self, gg_tensor, torch_tensor, gg_tensor2, torch_tensor2):
        assert np.isclose(gg.elementwise_mult(gg_tensor, gg_tensor2).to_list(),
                          torch.mul(torch_tensor, torch_tensor2).tolist(), rtol=1e-4).all()

    # def test_pow(self, gg_tensor, torch_tensor):
    #     assert np.isclose(gg.pow(gg_tensor, 3).to_list(),
    #                       torch.pow(torch_tensor, 3).tolist(), rtol=1e-4).all()

    def test_matmul(self, gg_tensor, torch_tensor, gg_tensor3, torch_tensor3):
        print(gg.matmul(gg_tensor, gg_tensor3))
        assert np.isclose(gg.matmul(gg_tensor, gg_tensor3).to_list(),
                          torch.matmul(torch_tensor, torch_tensor3).tolist(), rtol=1e-4).all()
