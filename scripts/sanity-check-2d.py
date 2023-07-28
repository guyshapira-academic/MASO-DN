"""
Includes a sanity check for the 2D case, which allows
for visual inspection of the results.
"""
import sys
import os

import numpy as np

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root_dir)

import torch
from torch import Tensor
from typing import Tuple

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

import maso, train, utils


def get_data(
    type_: str = "moons", n_samples: int = 10000, noise: float = 0.1
) -> Tuple[Tensor, ...]:
    total_size = int(n_samples / 0.8)
    x, y = None, None
    if type_ == "moons":
        x, y = datasets.make_moons(n_samples=total_size, noise=noise)
    elif type_ == "circles":
        x, y = datasets.make_circles(n_samples=total_size, noise=noise)
    else:
        raise ValueError(f"No such dataset type: {type_}")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train = torch.from_numpy(x_train).to(torch.float)
    y_train = torch.from_numpy(y_train.reshape(-1, 1)).to(torch.float)

    x_test = torch.from_numpy(x_test).to(torch.float)
    y_test = torch.from_numpy(y_test.reshape(-1, 1)).to(torch.float)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Hyperparameters
    epochs = 50
    batch_size = 8
    lr = 0.01

    net = maso.fc_network((2, 4, 4, 4, 1))
    x_train, y_train, x_test, y_test = get_data("circles", 10000, 0.05)

    # Train the network
    train.train(
        net,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Extract partitioning
    range_x = x_train[:, 0].min().item(), x_train[:, 0].max().item()
    range_y = x_train[:, 1].min().item(), x_train[:, 1].max().item()
    X, Y = torch.linspace(*range_x, 500), torch.linspace(*range_y, 500)
    X, Y = torch.meshgrid(X, Y, indexing="xy")
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    grid = torch.cat((X, Y), dim=1)
    Z = net(grid).detach().numpy().reshape(500, 500)

    global_partitions = net.assign_global_partition(grid, remove_redundant=True)
    num_partitions = global_partitions.shape[1]
    print(f"Number of partitions: {num_partitions}")

    # Plot partitions
    grid = grid.reshape(500, 500, -1)
    global_partitions = global_partitions.reshape(500, 500, -1).detach().numpy()
    boundary = utils.get_class_boundary(global_partitions)
    boundary = np.flipud(boundary)
    X = X.reshape(500, 500)
    Y = Y.reshape(500, 500)
    plt.contourf(X, Y, Z, levels=16, cmap="RdBu_r", alpha=0.5)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 0], cmap="RdBu_r", marker="x", alpha=0.05)
    plt.imshow(boundary, extent=[range_x[0], range_x[1], range_y[0], range_y[1]], cmap="Greys", alpha=1)
    plt.show()



