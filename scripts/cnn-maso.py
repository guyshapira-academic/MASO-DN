import sys
import os

import torch
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray
import cv2
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
        test_set = datasets.MNIST(
            root="data", train=False, download=True, transform=mnist_transform
        )
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
        test_set = datasets.CIFAR10(
            root="data", train=False, download=True, transform=cifar10_transform
        )

        train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])
    elif type_ == "cifar100":
        cifar100_transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR100(
            root="data", train=True, download=True, transform=cifar100_transform
        )
        test_set = datasets.CIFAR100(
            root="data", train=False, download=True, transform=cifar100_transform
        )

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
    batch_norm: bool = False,
    name: str = "untrained",
) -> None:
    if dataset == "mnist":
        train_set, val_set, test_set = get_data("mnist")
        input_channels = 1
        n_classes = 10
    elif dataset == "cifar10":
        train_set, val_set, test_set = get_data("cifar10")
        input_channels = 3
        n_classes = 10
    elif dataset == "cifar100":
        train_set, val_set, test_set = get_data("cifar100")
        input_channels = 3
        n_classes = 100
    elif dataset == "hybrid":
        train_set, val_set, _ = get_data("cifar100")
        _, _, test_set = get_data("cifar10")
        input_channels = 3
        n_classes = 100
    else:
        raise ValueError(f"No such dataset: {dataset}")

    if model == "smallCNN":
        net = maso.smallCNN(input_channels, bias=bias, batch_norm=batch_norm)
    elif model == "largeCNN":
        net = maso.largeCNN(input_channels, bias=bias, batch_norm=batch_norm)
    else:
        raise ValueError(f"No such model: {model}")
    net = nn.Sequential(net, nn.Flatten(), nn.LazyLinear(n_classes))
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
        num_classes=n_classes,
        pbar=True,
    )

    maso_net: maso.MASODN = net[0]

    # Get only input of the test set
    x_test = [x[0] for x in test_set]
    y_test = [x[1] for x in test_set]
    x_test = torch.stack(x_test)
    y_test = torch.tensor(y_test)

    N = 5000

    partitions, l2_distances = maso_net.layer_local_vq_distance(x_test[:N, ...])
    indices = torch.randperm(N)[:10]

    metrics_table = []
    for i, (p, l) in enumerate(zip(partitions, l2_distances)):
        show_images(x_test[:N], 10, 15, p, indices, i+1, name=name)
        semantic_metric = utils.semantic_metric(p, y_test[:N], 10).mean()
        l2_semantic_metric = utils.semantic_metric(l, y_test[:N], 10).mean()
        print(f"Layer {i} - VQ Semantic Metric: {semantic_metric:.4f}, CS Semantic Metric: {l2_semantic_metric:.4f}")

        metrics_table.append([semantic_metric, l2_semantic_metric])

    with open(f"images_2/table_{name}.txt", "w") as f:
        f.write(generate_latex_table(metrics_table, columns=["VQ", "CS"], rows=[f"{i}" for i in range(1, 6)]))


def show_images(images, num_images, k, distance_matrix, indices=None, layer_idx=1, name="images"):
    images = images.permute(0, 2, 3, 1).numpy()
    if indices is None:
        # get the indices of num_images random images
        indices = torch.randperm(len(images))[:num_images]
    else:
        if len(indices) != num_images:
            raise ValueError("Length of indices must be equal to num_images.")

    # fig, axs = plt.subplots(num_images, k + 1, figsize=(20, 20))

    for i, image_index in enumerate(indices):
        # find the k nearest neighbors for each image
        distances = distance_matrix[image_index]
        neighbor_indices = np.argsort(distances)[
            : k + 1
        ]  # k+1 to include the image itself

        fig, axs = plt.subplots(int(np.sqrt(k + 1)), int(np.sqrt(k + 1)), figsize=(10, 10))
        # fig.suptitle(f"Layer #{layer_idx}", fontsize=32)

        # the first column in each row is the image
        for j in range(int(np.sqrt(k + 1))):
            for l in range(int(np.sqrt(k + 1))):
                if j == 0 and k == 0:
                    axs[j, l].imshow(images[image_index].squeeze(), cmap="gray")
                    axs[j, l].axis("off")
                else:
                    axs[j, l].imshow(images[neighbor_indices[j * int(np.sqrt(k + 1)) + l]].squeeze(), cmap="gray")
                    axs[j, l].axis("off")

        plt.tight_layout()
        plt.savefig(f"images/cnn_image_{i}_layer_{layer_idx}_{name}.png"
                    f"", bbox_inches='tight', pad_inches=0)
        plt.close()

        # axs[i, 0].imshow(images[image_index].squeeze(), cmap="gray")
        # axs[i, 0].axis("off")

        # the rest of the columns are the k nearest neighbors
        # for j, neighbor_index in enumerate(neighbor_indices[1:], start=1):
        #     axs[i, j].imshow(images[neighbor_index].squeeze(), cmap="gray")
        #     axs[i, j].axis("off")

    # plt.tight_layout()
    # plt.show()


def generate_latex_table(data, columns, rows):
    num_columns = len(columns)
    if not all(len(row) == num_columns for row in data):
        raise ValueError("All data rows must have the same length as columns list")

    latex_table = "\\begin{{tabular}}{{|c|{}|}}\n".format("c|"*num_columns)
    latex_table += "\\hline\n"

    # Add column names
    latex_table += " & " + " & ".join(columns) + "\\\\\n"
    latex_table += "\\hline\n"

    # Add rows
    for i, row_data in enumerate(data):
        formatted_row_data = []
        for x in row_data:
            if isinstance(x, float):
                formatted_row_data.append("{:.4f}".format(x))
            else:
                formatted_row_data.append(str(x))
        latex_table += rows[i] + " & " + " & ".join(formatted_row_data) + "\\\\\n"
        latex_table += "\\hline\n"

    latex_table += "\\end{tabular}"

    return latex_table


if __name__ == "__main__":
    run(
        dataset="hybrid",
        lr=0.00005,
        batch_size=128,
        epochs=30,
        model="smallCNN",
        bias=False,
        batch_norm=True,
        name="trained"
    )
