from typing import Optional, Dict, Any, Tuple
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
        elif linear_operator == "conv":
            self.linear_operator = nn.Conv2d(**linear_operator_kwargs)
        else:
            raise ValueError(
                f"Invalid linear operator: {linear_operator}. " "Must be one of: 'fc'"
            )

        # Initialize activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU(**activation_function_kwargs)
        elif activation_function == "leaky_relu":
            self.activation_function = nn.LeakyReLU(**activation_function_kwargs)
        elif activation_function == "identity":
            self.activation_function = nn.Identity(**activation_function_kwargs)
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

        return self.partitions_per_dim**dim_out

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
            raise ValueError("The linear operator must be an instance of nn.Linear")

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
        if isinstance(self.activation_function, nn.Identity):
            return torch.ones(size=(x.shape[0], 1), dtype=torch.bool)
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


class MASODN(nn.Sequential):
    """
    This class is an abstraction of a basic MASO deep network, which consists
    of a sequence of MASO layers.

    Parameters:
        layers (list): List of MASO layers
    """

    def __init__(self, *layers: Tuple[MASOLayer, ...]):
        super().__init__(*layers)

    @property
    def num_partitions(self) -> int:
        """
        Returns the number of partitions induced by the network.

        Returns:
            int: Number of partitions
        """
        num_partitions = 1
        for layer in self:
            num_partitions *= layer.num_partitions
        return num_partitions

    def assign_global_partition(
        self, x: Tensor, l: Optional[int] = None, remove_redundant: bool = True
    ) -> Tensor:
        """
        Takes a tensor of input values and assigns each value to a global
        partition.

        Parameters:
            x (torch.Tensor): Input tensor
            l (int): Index of the layer to use for the assignment. If None,
                the last layer is used.
            remove_redundant (bool): If True, drops partitions with no members.

        Returns:
            int: Index of the global partition
        """
        l = len(self) - 1 if l is None else l
        z = x

        global_partitions = [
            torch.ones((x.shape[0],), dtype=torch.bool)
        ]  # An updating list of partitions
        # For each layer, get the partition assignments for the output of the
        # previous layer, and update the list of partitions by taking the
        # intersection of the previous partitions and the new partitions.
        for idx in range(l):
            layer = self[idx]
            local_partitions = layer.assign_local_partitions(
                z, remove_redundant=remove_redundant
            )
            new_global_partitions = list()
            for i, j in product(
                range(local_partitions.shape[1]), range(len(global_partitions))
            ):
                old_partition = global_partitions[j]
                new_partition = local_partitions[:, i]
                new_global = torch.logical_and(old_partition, new_partition)
                if new_global.any():
                    new_global_partitions.append(new_global)
            z = layer(z)
            global_partitions = new_global_partitions
        global_partitions = torch.stack(global_partitions, dim=1)
        return global_partitions


def fc_network(n_feature: Tuple[int, ...] = (2, 4, 4, 1)) -> MASODN:
    """
    Returns a fully connected network with the specified number of features.

    Parameters:
        n_feature (tuple): Number of features per layer

    Returns:
        MASODN: Fully connected network
    """
    layers = list()
    for idx in range(len(n_feature) - 1):
        activation_function = "relu" if idx < len(n_feature) - 2 else "identity"
        layers.append(
            MASOLayer(
                linear_operator="fc",
                linear_operator_kwargs={
                    "in_features": n_feature[idx],
                    "out_features": n_feature[idx + 1],
                },
                activation_function=activation_function,
            )
        )
    return MASODN(*layers)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import utils

    # Create a simple network
    net = MASODN(
        MASOLayer(
            linear_operator="fc",
            linear_operator_kwargs={"in_features": 2, "out_features": 4},
            activation_function="relu",
        ),
        MASOLayer(
            linear_operator="fc",
            linear_operator_kwargs={"in_features": 4, "out_features": 4},
            activation_function="relu",
        ),
        MASOLayer(
            linear_operator="fc",
            linear_operator_kwargs={"in_features": 4, "out_features": 1},
            activation_function="identity",
        ),
    )

    # Create grid of points
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = torch.cat((X, Y), dim=1)

    # Assign partitions
    global_partitions = net.assign_global_partition(Z, remove_redundant=True)
    num_partitions = global_partitions.shape[1]
    print(f"Number of partitions: {num_partitions}")

    # Plot partitions
    global_partitions = global_partitions.detach().numpy()
    global_partitions = global_partitions.reshape(100, 100, num_partitions)
    boundary = utils.get_class_boundary(global_partitions)
    plt.imshow(boundary, cmap="Greys", extent=[-1, 1, -1, 1])
    plt.show()
