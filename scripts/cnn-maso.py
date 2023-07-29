import sys
import os

import numpy as np

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
        net = maso.smallCNN(input_channels)
    elif model == "largeCNN":
        net = maso.largeCNN(input_channels)
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

    partitions = maso_net.assign_global_partition(x_test, l=1)
    num_partitions = partitions.shape[1]
    print(f"Number of partitions: {num_partitions}")


if __name__ == "__main__":
    run()
