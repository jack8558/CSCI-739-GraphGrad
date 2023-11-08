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
    "data",
    [
        0,
        True,
        -3.14,
        9,
        [9],
        [[[[9]]]],
        [[[[9]], [[9]]]],
        [1, 2, 3],
        (1, 2, 3),
        range(100),
        ([10.1, 20.2, 30.3], range(3)),
        [],
        [[], [], [], []],
        [[1, 2], [3, 4]],
        [[[1], [2], [8]], [[3], [4], [9]]],
        np.ones((3, 4, 5, 6)),
        torch.ones((3, 4, 5, 6)),
    ],
)
def test_tensor_data_constructor(data):
    """Verify that the data constructor builds the same array as NumPy."""
    gg_tensor = gg.Tensor(data)
    np_array = np.array(data)

    np_array_from_gg = np.array(gg_tensor.to_list())
    assert np_array_from_gg.shape == np_array.shape
    assert np.allclose(np_array_from_gg, np_array)


@pytest.mark.parametrize(
    "data",
    [
        None,
        "not a tensor",
        b"not a tensor",
        ["not a value"],
        [1, 2, 5, "three, sir!", 3],
        [1, 2, [3, 4]],
        [1, 2, []],
        [[], 3],
        [[], [3]],
        [[1, 2, 3], 1],
        [[1, 2, 3], []],
        [[1, 2, 3], [1, 2]],
        [[1, 2, 3], [1, 2, 3, 4]],
        [[[[1, 2, 3]]], [[[]]]],
    ],
)
def test_tensor_data_constructor_raises(data):
    """Verify that the data constructor raises an error when given invalid array data."""
    with pytest.raises((ValueError, TypeError)):
        gg.Tensor(data)
