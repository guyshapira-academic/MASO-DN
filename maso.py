from typing import Optional, Dict, Any
from itertools import product

import torch
import torch.nn as nn
from torch import Tensor


class MASOLayer(nn.Module):
    """
    This class is an abstraction of a basic MASO layer, which consists of a
    linear operator, followed by a non-linear activation function.

    Parameters:
        linear_operator (str): The linear operator to be used in the layer. Can
            be one of the following:
                - 'fc': Fully connected layer
            Default: 'fc'
        linear_operator_kwargs (dict): Keyword arguments to be passed to the
            linear operator. Default: {}
        activation_function (str): The activation function to be used in the
            layer. Can be one of the following:
                - 'relu': ReLU activation function
                - 'leaky_relu': Leaky ReLU activation function
                - 'identity': Identity activation function
            Default: 'relu'
        activation_function_kwargs (dict): Keyword arguments to be passed to
            the activation function. Default: {}
    """
    def __init__(
        self,
        linear_operator: str = "fc",
        linear_operator_kwargs: Optional[Dict[str, Any]] = None,
        activation_function: str = "relu",
        activation_function_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Initialize linear operator keyword arguments
        if linear_operator_kwargs is None:
            linear_operator_kwargs = {}
        if activation_function_kwargs is None:
            activation_function_kwargs = {}

        # Initialize linear operator
        if linear_operator == "fc":
            self.linear_operator = nn.Linear(**linear_operator_kwargs)
        else:
            raise ValueError(
                f"Invalid linear operator: {linear_operator}. "
                "Must be one of: 'fc'"
            )

        # Initialize activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU(**activation_function_kwargs)
        elif activation_function == "leaky_relu":
            self.activation_function = nn.LeakyReLU(
                **activation_function_kwargs
            )
        elif activation_function == "identity":
            self.activation_function = nn.Identity(
                **activation_function_kwargs
            )
        else:
            raise ValueError(
                f"Invalid activation function: {activation_function}. "
                "Must be one of: 'relu', 'leaky_relu', 'abs'"
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer.

        Parameters:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.linear_operator(x)
        x = self.activation_function(x)
        return x

    @property
    def partitions_per_dim(self):
        if type(self.activation_function) in [nn.ReLU, nn.LeakyReLU]:
            return 2
        elif type(self.activation_function) == nn.Identity:
            return 1
        else:
            return 0

    @property
    def num_partitions(self) -> int:
        """
        Returns the number of partitions induced by the layer.

        Returns:
            int: Number of partitions
        """
        if isinstance(self.linear_operator, nn.Linear):
            dim_out = self.linear_operator.out_features
        else:
            return 0

        return self.partitions_per_dim ** dim_out

    def get_local_partitions_descriptor(self) -> Tensor:
        """
        Returns the affine parameters that define the local partitions.

        Parameters:
            k (int): Index of the dimension

        Returns:
            torch.Tensor: Local partition descriptor
        """
        if isinstance(self.linear_operator, nn.Linear):
            A = self.linear_operator.weight
            b = self.linear_operator.bias
            return A, b
        else:
            raise ValueError(
                "The linear operator must be an instance of nn.Linear"
            )

    def assign_local_partitions(self, x: Tensor, remove_redundant: bool) -> Tensor:
        """
        Takes a tensor of input values and assigns each value to a local
        partition.

        Parameters:
            x (torch.Tensor): Input tensor
            remove_redundant (bool): If True, drops partitions with no members.

        Returns:
            one-hot encoding of the assignment.
        """
        A, b = self.get_local_partitions_descriptor()
        z = x @ A.T + b
        partitions = list()
        for comb in product([0, 1], repeat=z.shape[1]):
            r = torch.ones(size=(z.shape[0],))
            for column in range(z.shape[1]):
                col = z[:, column] > 0
                if not comb[column]:
                    col = torch.logical_not(col)
                r = torch.logical_and(r, col)
            partitions.append(r)
        partitions = torch.stack(partitions, dim=1)
        if remove_redundant:
            partitions = partitions[:, partitions.any(dim=0)]
        return partitions
