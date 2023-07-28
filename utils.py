from typing import List, Union, Tuple

import numpy as np
from numpy.typing import NDArray

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
    new_conv_layer = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=False)  # Setting bias to False removes the bias term

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
