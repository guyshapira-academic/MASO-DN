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
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

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
        y = y.astype(float)
    elif type_ == "circles":
        x, y = datasets.make_circles(n_samples=total_size, noise=noise)
        y = y.astype(float)
    elif type_ == "both":
        x_moons, y_moons = datasets.make_moons(n_samples=total_size, noise=noise)
        x_circles, y_circles = datasets.make_circles(
            n_samples=total_size, noise=noise, factor=0.5
        )
        x_moons = x_moons + 1.5
        y_moons = y_moons + 2
        x = np.concatenate((x_moons, x_circles))
        y = np.concatenate((y_moons, y_circles))

    else:
        raise ValueError(f"No such dataset type: {type_}")

    # Normalize
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train = torch.from_numpy(x_train).to(torch.float)
    y_train = torch.from_numpy(y_train.reshape(-1, 1))

    x_test = torch.from_numpy(x_test).to(torch.float)
    y_test = torch.from_numpy(y_test.reshape(-1, 1))

    if type_ == "both":
        y_train = y_train.flatten().to(torch.long)
        y_test = y_test.flatten().to(torch.long)
    else:
        y_train = y_train.to(torch.float)
        y_test = y_test.to(torch.float)


    return x_train, y_train, x_test, y_test


def run(
    dataset: str = "moons",
    n_samples: int = 10000,
    noise: float = 0.1,
    lr: float = 0.01,
    batch_size: int = 32,
    epochs: int = 100,
    name: Optional[str] = None,
) -> None:
    out_dim = 4 if dataset == "both" else 1
    maso_net = maso.fc_network((2, 8, 4, 4, out_dim), bias=False, bn=True)

    if dataset == "both":
        net = nn.Sequential(
            maso_net,
            nn.Softmax(dim=1),
        )
    else:
        net = nn.Sequential(
            maso_net,
            nn.Sigmoid(),
        )
    x_train, y_train, x_test, y_test = get_data(dataset, n_samples, noise)

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
        num_classes=4 if dataset == "both" else 2,
        pbar=False
    )

    net = net[0]

    # Extract partitioning
    range_x = x_train[:, 0].min().item(), x_train[:, 0].max().item()
    range_y = x_train[:, 1].min().item(), x_train[:, 1].max().item()
    X, Y = torch.linspace(*range_x, 500), torch.linspace(*range_y, 500)
    X, Y = torch.meshgrid(X, Y, indexing="xy")
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    grid = torch.cat((X, Y), dim=1)
    if dataset != "both":
        Z = net(grid).detach().numpy().reshape(500, 500)
    else:
        Z = None

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
    if dataset != "both":
        plt.contourf(X, Y, Z, levels=16, cmap="RdBu_r", alpha=0.5)
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="RdBu_r",
        marker="x",
        alpha=0.05,
    )
    plt.imshow(
        boundary,
        extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
        cmap="Greys",
        alpha=1,
    )
    if name is not None:
        plt.savefig(f"{name}.png")
    plt.show()


if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    lr = 0.01

    run(
        "moons",
        n_samples=10000,
        noise=0.1,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        name="moons",
    )
    run(
        "circles",
        n_samples=10000,
        noise=0.1,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        name="circles",
    )
    run(
        "both",
        n_samples=10000,
        noise=0.1,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        name="both",
    )
