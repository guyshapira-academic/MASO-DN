from typing import List, Union, Tuple
import itertools

import numpy as np
from numpy.typing import NDArray, ArrayLike

import torch
import torch.nn as nn
from torch import Tensor


def dec_to_bin(dec: int) -> List[int]:
    """
    Converts from decimal int to binary list.

    Parameters:
        dec (int): Decimal integer to be converted.
    """
    bin_list = []
    while dec > 0:
        bin_list.append(dec % 2)
        dec = dec // 2
    return bin_list[::-1]


def bin_to_dec(bin_tensor: NDArray) -> NDArray:
    """
    Converts from binary list to decimal int.

    Parameters:
        bin_tensor (np.ndarray or torch.Tensor): Binary list to be converted.
    """
    idx = np.arange(bin_tensor.shape[-1])
    idx = 2**idx
    idx = idx[::-1]

    idx = idx.reshape((-1, 1))

    dec = bin_tensor @ idx
    return dec


def get_class_boundary(grid: NDArray) -> NDArray:
    """
    Returns the class boundary of a grid.

    Parameters:
        grid (np.ndarray or torch.Tensor): boolean spatial grid, with shape
            (d1, d2, ..., dn, C), where C is the number of classes.

    Returns:
        np.ndarray or torch.Tensor: Class boundary of the grid.
    """
    grad_axis = tuple(range(len(grid.shape) - 1))
    grad = np.gradient(grid.astype(int), axis=grad_axis)
    grad = np.concatenate(grad, axis=-1)
    b = grad > 0
    b = b.any(axis=-1)
    return b


def conv2d_to_linear(conv_layer, input_shape: Tuple[int, int, int]):
    """
    Converts a 2D convolutional layer linear parameters.

    Parameters:
        conv_layer (nn.Conv2d): 2D convolutional layer
        input_shape (tuple): Shape of the input to the layer

    Returns:
        tuple: Linear parameters (A, b)
    """
    if not isinstance(conv_layer, nn.Conv2d):
        raise ValueError("Input layer must be a nn.Conv2d layer.")
    dummy_input = torch.randn(size=(1, *input_shape))
    dummy_output = conv_layer(dummy_input)

    output_size = dummy_output.numel()
    input_size = dummy_input.numel()

    # Create identical layer with no bias
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    groups = conv_layer.groups
    new_conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False,
    )  # Setting bias to False removes the bias term

    # Copy the weights from the original layer to the new layer
    new_conv_layer.weight.data = conv_layer.weight.data.clone()

    I = torch.eye(input_size)
    I = I.reshape((input_size, *input_shape))
    A = new_conv_layer(I)
    A = A.reshape((input_size, output_size))

    x = torch.zeros(size=(1, input_size))
    x = x.reshape((1, *input_shape))
    b = conv_layer(x)
    b = b.reshape((output_size,))

    return A.T, b


def count_ones(comb):
    # Helper function to count the number of ones in a combination
    return sum(1 for elem in comb if elem == 1)


def custom_sort(comb):
    # Sorting function based on the count of ones in each combination
    return count_ones(comb)


def iter_combinations(r):
    # Generate all combinations of 0's and 1's of length 'r'
    combinations = list(itertools.product([0, 1], repeat=r))
    combinations.sort(key=custom_sort)
    return combinations


def nearest_neighbors_from_pdist(
    distance_matrix: NDArray, k: int = 10
) -> NDArray:
    """
    Given a distance matrix, return the indices of the nearest neighbors

    Parameters:
        distance_matrix (np.ndarray or torch.Tensor): Distance matrix.
        k (int): Number of neighbors to return.
    """
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.cpu().numpy()

    n = distance_matrix.shape[0]
    neighbors_idx = np.zeros((n, k), dtype=int)
    for i in range(n):
        neighbors_idx[i] = np.argsort(distance_matrix[i])[1: k + 1]
    return neighbors_idx


def semantic_metric_calc(neighbors_idx: NDArray, classes: NDArray) -> NDArray:
    """
    Given a set of images and the indices of the neighbors, computes the percentage
    of neighbors of each image that belong to its class.

    Parameters:
        neighbors_idx (np.ndarray or torch.Tensor): Index of neighbors.
        classes (np.ndarray or torch.Tensor): Classes of the neighbors.

    Returns:
        np.ndarray or torch.Tensor: Semantic metric of the neighbors.
    """
    if isinstance(neighbors_idx, torch.Tensor):
        neighbors_idx = neighbors_idx.cpu().numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.cpu().numpy()

    n_neighbors = neighbors_idx.shape[1]

    semantic_metric = np.zeros((neighbors_idx.shape[0]))
    for i in range(n_neighbors):
        neighbor_class = classes[neighbors_idx[:, i]]
        semantic_metric += (neighbor_class == classes)
    semantic_metric /= n_neighbors
    return semantic_metric


def semantic_metric(pdist: NDArray, classes: NDArray, k: int) -> NDArray:
    """
    Given a set of images and the indices of the neighbors, computes the percentage
    of neighbors of each image that belong to its class.

    Parameters:
        pdist (np.ndarray or torch.Tensor): Pairwise distance matrix.
        classes (np.ndarray or torch.Tensor): Classes of the neighbors.
        k (int): Number of neighbors.

    Returns:
        np.ndarray or torch.Tensor: Semantic metric of the neighbors.
    """
    neighbors_idx = nearest_neighbors_from_pdist(pdist, k=k, n=pdist.shape[0])
    return semantic_metric_calc(neighbors_idx, classes)
