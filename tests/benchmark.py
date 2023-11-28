import time
import torch
import numpy as np
import os
import sys
import math
import argparse

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


def hw3_test1(device):
    x = get_data(3, 5, 0.5)
    y = get_data(5, 7, 0.5)
    z = get_data(7, 1, 0.5)

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.tensor(x)
    tensor_y = gg.tensor(y)
    tensor_z = gg.tensor(z)

    gg_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)

    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x, dtype=torch.float64).to(device)
    tensor_y = torch.tensor(y, dtype=torch.float64).to(device)
    tensor_z = torch.tensor(z, dtype=torch.float64).to(device)

    torch_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z).to(device)

    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time


def hw3_test2(device):
    x = get_data(300, 5000, 1.0)
    y = get_data(5000, 1000, 1.0)
    z = get_data(1000, 1, 1.0)

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.tensor(x)
    tensor_y = gg.tensor(y)
    tensor_z = gg.tensor(z)

    gg_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)

    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x, dtype=torch.float64).to(device)
    tensor_y = torch.tensor(y, dtype=torch.float64).to(device)
    tensor_z = torch.tensor(z, dtype=torch.float64).to(device)

    torch_res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z).to(device)

    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time


def hw3_test3(device, epochs):
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

    tensor_x = gg.tensor(x)
    tensor_y = gg.tensor(y)
    tensor_z = gg.tensor(z)
    tensor_label = gg.tensor(label)
    gg_lr = gg.tensor([learning_rate])
    gg_const = gg.tensor([1.0 / N])

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

    tensor_x = torch.tensor(x, dtype=torch.float64).to(device)
    tensor_y = torch.tensor(y, dtype=torch.float64).to(device)
    tensor_z = torch.tensor(z, dtype=torch.float64).to(device)
    tensor_label = torch.tensor(label, dtype=torch.float64).to(device)

    for i in range(epochs):
        h1 = tensor_x.matmul(tensor_y).to(device)  # N*P
        h2 = h1.relu().to(device)  # N*P
        out = h2.matmul(tensor_z).to(device)  # N*1
        grad_out = out.reshape([N]).subtract(tensor_label).mul(1.0 / N).to(device)  # N
        grad_tensor_z = h2.transpose(0, 1).matmul(grad_out.reshape([N, 1])).to(device)  # P*1
        grad_h2 = grad_out.reshape([N, 1]).matmul(tensor_z.transpose(0, 1)).to(device)
        grad_h1 = torch.where(h1 > 0.0, 1.0, 0.0).mul(grad_h2).to(device)
        grad_tensor_y = tensor_x.transpose(0, 1).matmul(grad_h1).to(device)

        tensor_y = tensor_y.subtract(grad_tensor_y.mul(learning_rate)).to(device)
        tensor_z = tensor_z.subtract(grad_tensor_z.mul(learning_rate)).to(device)

    torch_res = out
    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time

def cse_test(device, epochs):
    N = 1000
    M = 500
    P = 700

    x = get_data(N, M, 1)
    y = np.random.uniform(low=-1.0, high=1.0, size=[M, P]) / math.sqrt(M)
    z = np.random.uniform(low=-1.0, high=1.0, size=[P, 1]) / math.sqrt(P)
    label = np.random.uniform(size=[1000])

    # Graph Grad
    start = time.perf_counter()

    tensor_x = gg.tensor(x)
    tensor_y = gg.tensor(y)
    tensor_z = gg.tensor(z)

    out = gg.tensor([0])
    for i in range(epochs):
        x_y = tensor_x.matmul(tensor_y)
        y_z = x_y.matmul(tensor_z)
        out1 = y_z * y_z
        out2 = y_z * out1
        out += out2.sum()

    gg_res = out
    print("GraphGrad result:")
    print(gg_res)
    print()

    end = time.perf_counter()
    gg_time = end - start

    # Torch
    start = time.perf_counter()

    tensor_x = torch.tensor(x, dtype=torch.float64).to(device)
    tensor_y = torch.tensor(y, dtype=torch.float64).to(device)
    tensor_z = torch.tensor(z, dtype=torch.float64).to(device)

    out = torch.tensor([0.0], dtype=torch.float64).to(device)
    for i in range(epochs):
        x_y = tensor_x.matmul(tensor_y).to(device)
        y_z = x_y.matmul(tensor_z).to(device)
        out1 = (y_z * y_z).to(device)
        out2 = (y_z * out1).to(device)
        out += out2.sum().to(device)

    torch_res = out
    print("Torch result:")
    print(torch_res)
    print()

    end = time.perf_counter()
    torch_time = end - start

    assert np.isclose(gg_res.to_list(), torch_res, rtol=1e-4).all()

    return gg_res, gg_time, torch_res, torch_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional command line flag.')

    # Adding an optional flag
    parser.add_argument('-use_gpu', '--use_gpu', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()

    device = "cpu"
    if args.use_gpu:
        gg.use_gpu(True)
        if torch.cuda.is_available():
            device = torch.device("cuda")


    print("Starting test 1...")
    _, gg_time, _, torch_time = hw3_test1(device)
    print(f"Test 1 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 2...")
    _, gg_time, _, torch_time = hw3_test2(device)
    print(f"Test 2 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 3...")
    _, gg_time, _, torch_time = hw3_test3(device, 10)
    print(f"Test 3 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 4...")
    _, gg_time, _, torch_time = hw3_test3(device, 100)
    print(f"Test 4 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")

    print("Starting test 5...")
    _, gg_time, _, torch_time = cse_test(device, 1000)
    print(f"Test 5 finished.\n\tGraphGrad time: {gg_time}\n\tTorch time: {torch_time}")

    print("\n=========================\n")
