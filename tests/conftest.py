import pytest
import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import graphgrad as gg


def pytest_addoption(parser):
    parser.addoption("--use_gpu", action="store_true")


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    gpu = session.config.getoption("use_gpu")
    print("use_gpu:", gpu)
    gg.use_gpu(gpu)


@pytest.fixture(scope="session")
def gg_tensor_5_10():
    return gg.rand([5, 10])


@pytest.fixture(scope="session")
def gg_tensor_10_10():
    return gg.rand([10, 10])


@pytest.fixture(scope="session")
def gg_tensor_50_100():
    return gg.rand([50, 100])
