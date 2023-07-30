import sys
import os

import numpy as np
from numpy.typing import NDArray

import maso

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root_dir)

import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Optional
import torch.utils.data
from torchvision import datasets
from torchvision import transforms as T


import matplotlib.pyplot as plt

import maso, train, utils


def get_data(type_: str) -> Tuple[Tensor, ...]:
    train_set, test_set = None, None
    if type_ == "mnist":
        mnist_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((32, 32), antialias=True),
            ]
        )
        train_set = datasets.MNIST(
            root="data", train=True, download=True, transform=mnist_transform
        )
        test_set = datasets.MNIST(root="data", train=False, download=True, transform=mnist_transform)
        train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])
    elif type_ == "cifar10":
        cifar10_transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR10(
            root="data", train=True, download=True, transform=cifar10_transform
        )
        test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=cifar10_transform)

        train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])
    else:
        raise ValueError(f"No such dataset type: {type_}")

    return train_set, val_set, test_set


def run(
    dataset: str = "mnist",
    model: str = "smallCNN",
    lr: float = 0.01,
    batch_size: int = 32,
    epochs: int = 1,
    bias: bool = True,
) -> None:
    if dataset == "mnist":
        train_set, val_set, test_set = get_data("mnist")
        input_channels = 1
    elif dataset == "cifar10":
        train_set, val_set, test_set = get_data("cifar10")
        input_channels = 3
    else:
        raise ValueError(f"No such dataset: {dataset}")

    if model == "smallCNN":
        net = maso.smallCNN(input_channels, bias=bias)
    elif model == "largeCNN":
        net = maso.largeCNN(input_channels, bias=bias)
    else:
        raise ValueError(f"No such model: {model}")
    net = nn.Sequential(net, nn.Flatten(), nn.LazyLinear(10))
    print(net)

    # Train the network
    train.train(
        net,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        train_set=train_set,
        val_set=val_set,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_classes=10,
        pbar=True,
    )

    maso_net: maso.MASODN = net[0]

    # Get only input of the test set
    x_test = [x[0] for x in test_set]
    x_test = torch.stack(x_test)

    partitions = maso_net.layer_local_vq_distance(x_test[:1500, ...])

    for p in partitions:
        neighbors_idx = nearest_neighbors_from_pdist(p, k=5, n=5)
        fig, axs = plt.subplots(nrows=5, ncols=6)
        fig.tight_layout(pad=0.01)

        for i in range(5):
            axs[i, 0].imshow(x_test[i, ...].permute(1, 2, 0))
            axs[i, 0].set_axis_off()
            for j in range(5):
                axs[i, j + 1].imshow(x_test[neighbors_idx[i, j], ...].permute(1, 2, 0))
                axs[i, j + 1].set_axis_off()

        plt.show()


def nearest_neighbors_from_pdist(distance_matrix: NDArray, k: int = 10, n: int = 10) -> NDArray:
    """
    Given a distance matrix, return the indices of the nearest neighbors
    """
    neighbors = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        neighbors[i, :] = np.argsort(distance_matrix[i, :])[1:k+1]
    return neighbors


if __name__ == "__main__":
    run(dataset="cifar10", lr=0.0005, batch_size=128, epochs=10, model="smallCNN", bias=False)
