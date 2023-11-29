import torch
from sklearn import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import sys
import numpy as np

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import graphgrad as gg


def xavier_init(fan_in: int, dims: 'list[int]') -> gg.tensor:
    scale = np.sqrt(1 / fan_in)
    tensor = (gg.rand(dims) * (2*scale)) - scale
    return gg.eval(tensor)


class Parameter:
    def __init__(self, value: gg.tensor) -> None:
        self.value = value


class MLP:
    def __init__(self, input_size: int, hidden_sizes: 'list[int]', output_size: int):
        self.parameters: 'list[Parameter]' = []
        self.layers: 'list[tuple[Parameter, Parameter]]' = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:]):
            w = Parameter(xavier_init(fan_in, [fan_in, fan_out]))
            b = Parameter(xavier_init(fan_in, [fan_out]))
            self.layers.append((w, b))
            self.parameters.extend([w, b])

    def forward(self, x: gg.tensor) -> gg.tensor:
        for i, (w, b) in enumerate(self.layers):
            w = w.value
            b = b.value

            x = (x @ w) + b
            if i != len(self.layers) - 1:
                x = gg.relu(x)
        return x

def main():
    gg.use_gpu(True)
    gg.set_cuda_device(9)

    # Initialize the model.
    model = MLP(8, [], 1)

    # Create the dataset.
    X, Y = datasets.fetch_california_housing(return_X_y=True)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    train_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=1000,
        shuffle=True,
    )

    learning_rate = 0.05
    epochs = 30

    for _ in range(epochs):
        for features, targets in train_dataloader:
            features = gg.tensor(((features - X_mean) / X_std).numpy())
            targets = gg.tensor(targets.numpy())
            [batch_size] = targets.dims()

            # Get the model predictions.
            predictions = model.forward(features).reshape([batch_size])

            # Get the loss.
            loss = (predictions - targets).pow(2).sum() / batch_size
            print('Loss:', loss.to_list())

            # Backpropagate to get parameter gradients.
            loss.backward()

            # Update the parameters.
            for param in model.parameters:
                param.value = gg.eval(param.value - learning_rate * param.value.grad)

            gg.clear_cache()


if __name__ == "__main__" :
    main()
