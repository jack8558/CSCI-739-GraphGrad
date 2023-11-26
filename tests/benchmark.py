import time
import torch
import numpy as np
import os
import sys
import math

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import graphgrad as gg


np.random.seed(1234567)


def get_data(M, N, threshold):
    ret = np.random.normal(size=[M, N])
    ret[np.abs(ret) < threshold] = 0
    return ret


def test1():
    x = get_data(3, 5, 0.5)
    y = get_data(5, 7, 0.5)
    z = get_data(7, 1, 0.5)

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.Tensor(x)
    tensor_y = gg.Tensor(y)
    tensor_z = gg.Tensor(z)

    gg_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)

    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x)
    tensor_y = torch.tensor(y)
    tensor_z = torch.tensor(z)

    torch_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)

    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time


def test2():
    x = get_data(300, 5000, 1.0)
    y = get_data(5000, 1000, 1.0)
    z = get_data(1000, 1, 1.0)

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.Tensor(x)
    tensor_y = gg.Tensor(y)
    tensor_z = gg.Tensor(z)

    gg_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)

    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x)
    tensor_y = torch.tensor(y)
    tensor_z = torch.tensor(z)

    torch_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)

    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time


def test3(epochs):
    N = 1000
    M = 500
    P = 700
    learning_rate = 0.01

    x = get_data(N, M, 1)
    y = np.random.uniform(low=-1.0, high=1.0, size=[M, P]) / math.sqrt(M)
    z = np.random.uniform(low=-1.0, high=1.0, size=[P, 1]) / math.sqrt(P)
    label = np.random.uniform(size=[1000])

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.Tensor(x)
    tensor_y = gg.Tensor(y)
    tensor_z = gg.Tensor(z)
    tensor_label = gg.Tensor(label)
    gg_lr = gg.Tensor([learning_rate])
    gg_const = gg.Tensor([1.0 / N])

    for i in range(epochs):
        h1 = tensor_x.matmul(tensor_y)  # N*P
        h2 = h1.relu()  # N*P
        out = h2.matmul(tensor_z)  # N*1
        grad_out = out.reshape([N]).subtract(tensor_label).mul(gg_const)  # N
        grad_tensor_z = h2.transpose(0, 1).matmul(grad_out.reshape([N, 1]))  # P*1
        grad_h2 = grad_out.reshape([N, 1]).matmul(tensor_z.transpose(0, 1))
        grad_h1 = h1.binilarize().mul(grad_h2)
        grad_tensor_y = tensor_x.transpose(0, 1).matmul(grad_h1)

        tensor_y = tensor_y.subtract(grad_tensor_y.mul(gg_lr))
        tensor_z = tensor_z.subtract(grad_tensor_z.mul(gg_lr))

    gg_res = out
    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x)
    tensor_y = torch.tensor(y)
    tensor_z = torch.tensor(z)
    tensor_label = torch.tensor(label)

    for i in range(epochs):
        h1 = tensor_x.matmul(tensor_y)  # N*P
        h2 = h1.relu()  # N*P
        out = h2.matmul(tensor_z)  # N*1
        grad_out = out.reshape([N]).subtract(tensor_label).mul(1.0 / N)  # N
        grad_tensor_z = h2.transpose(0, 1).matmul(grad_out.reshape([N, 1]))  # P*1
        grad_h2 = grad_out.reshape([N, 1]).matmul(tensor_z.transpose(0, 1))
        grad_h1 = torch.where(h1 > 0.0, 1.0, 0.0).mul(grad_h2)
        grad_tensor_y = tensor_x.transpose(0, 1).matmul(grad_h1)

        tensor_y = tensor_y.subtract(grad_tensor_y.mul(learning_rate))
        tensor_z = tensor_z.subtract(grad_tensor_z.mul(learning_rate))

    torch_res = out
    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time

def cse_test(epochs):
    N = 1000
    M = 500
    P = 700
    learning_rate = 0.01

    x = get_data(N, M, 1)
    y = np.random.uniform(low=-1.0, high=1.0, size=[M, P]) / math.sqrt(M)
    z = np.random.uniform(low=-1.0, high=1.0, size=[P, 1]) / math.sqrt(P)
    label = np.random.uniform(size=[1000])

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.Tensor(x)
    tensor_y = gg.Tensor(y)
    tensor_z = gg.Tensor(z)
    tensor_label = gg.Tensor(label)
    gg_lr = gg.Tensor([learning_rate])
    gg_const = gg.Tensor([1.0 / N])

    for i in range(epochs):
        h1 = tensor_x.matmul(tensor_y)  # N*P
        h2 = h1.relu()  # N*P
        h3 = h2  # Common subexpression with h2
        h4 = h2  # Common subexpression with h2
        out = h2.matmul(tensor_z) + h3.matmul(tensor_z) + h4.matmul(tensor_z)  # N*1
        grad_out = out.reshape([N]).subtract(tensor_label).mul(gg_const)  # N
        grad_tensor_z = h2.transpose(0, 1).matmul(grad_out.reshape([N, 1]))  # P*1
        grad_h2 = grad_out.reshape([N, 1]).matmul(tensor_z.transpose(0, 1))
        grad_h3 = grad_h2  # Derivative of relu is 1 for positive values
        grad_h4 = grad_h2  # Derivative of relu is 1 for positive values
        grad_tensor_y = tensor_x.transpose(0, 1).matmul(grad_h2 + grad_h3 + grad_h4)

        tensor_y = tensor_y.subtract(grad_tensor_y.mul(gg_lr))
        tensor_z = tensor_z.subtract(grad_tensor_z.mul(gg_lr))

    gg_res = out
    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x)
    tensor_y = torch.tensor(y)
    tensor_z = torch.tensor(z)
    tensor_label = torch.tensor(label)

    for i in range(epochs):
        h1 = tensor_x.matmul(tensor_y)  # N*P
        h2 = h1.relu()  # N*P
        h3 = h2  # Common subexpression with h2
        h4 = h2  # Common subexpression with h2
        out = h2.matmul(tensor_z) + h3.matmul(tensor_z) + h4.matmul(tensor_z)  # N*1
        grad_out = out.reshape([N]).subtract(tensor_label).mul(1.0 / N)  # N
        grad_tensor_z = h2.transpose(0, 1).matmul(grad_out.reshape([N, 1]))  # P*1
        grad_h2 = grad_out.reshape([N, 1]).matmul(tensor_z.transpose(0, 1))
        grad_h3 = grad_h2  # Derivative of relu is 1 for positive values
        grad_h4 = grad_h2  # Derivative of relu is 1 for positive values
        grad_tensor_y = tensor_x.transpose(0, 1).matmul(grad_h2 + grad_h3 + grad_h4)

        tensor_y = tensor_y.subtract(grad_tensor_y.mul(learning_rate))
        tensor_z = tensor_z.subtract(grad_tensor_z.mul(learning_rate))

    torch_res = out
    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time

if __name__ == "__main__":
    print("Starting test 1...")
    _, gg_time, _, torch_time = test1()
    print(f"Test 1 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 2...")
    _, gg_time, _, torch_time = test2()
    print(f"Test 2 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 3...")
    _, gg_time, _, torch_time = test3(10)
    print(f"Test 3 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 4...")
    _, gg_time, _, torch_time = test3(100)
    print(f"Test 4 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 5...")
    _, gg_time, _, torch_time = cse_test(100)
    print(f"Test 5 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")
