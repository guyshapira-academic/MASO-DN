from typing import Optional, Dict, Any

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
                - 'abs': Absolute value activation function
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
        elif activation_function == "abs":
            self.activation_function = nn.ReLU(**activation_function_kwargs)
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
