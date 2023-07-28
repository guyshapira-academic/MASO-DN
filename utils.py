from typing import List, Union

import numpy as np
from numpy.typing import NDArray

import torch
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
    grad = np.gradient(grid, axis=grad_axis)
    print(grad)
    b = grad > 0
    print(b)
    b = b.any(axis=-1)
    return b
