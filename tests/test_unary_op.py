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


class TestUnaryOP:

    def test_neg(self, gg_tensor, torch_tensor):
        assert np.isclose(gg.neg(gg_tensor).to_list(), torch.neg(torch_tensor).tolist(), rtol=1e-4).all()

    def test_reciprocal(self, gg_tensor, torch_tensor):
        assert np.isclose(gg.reciprocal(gg_tensor).to_list(), torch.reciprocal(torch_tensor).tolist(), rtol=1e-4).all()

    def test_relu(self, gg_tensor, torch_tensor):
        assert np.isclose(gg.relu(gg_tensor).to_list(), torch.nn.functional.relu(torch_tensor).tolist(), rtol=1e-4).all()

    def test_binilarize(self, gg_tensor, torch_tensor):
        assert np.isclose(gg.binilarize(gg_tensor).to_list(), (torch_tensor > 0.0).double(), rtol=1e-4).all()

    def test_exp(self, gg_tensor, torch_tensor):
        assert np.isclose(gg.exp(gg_tensor).to_list(), torch.exp(torch_tensor).tolist(), rtol=1e-4).all()
