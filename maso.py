from typing import Optional, Dict, Any, Tuple, List
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from tqdm import tqdm, trange

import einops
from scipy.spatial.distance import squareform, pdist

import utils


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
        batch_norm: bool = False,
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
        elif linear_operator == "identity":
            self.linear_operator = nn.Identity(**linear_operator_kwargs)
        else:
            raise ValueError(
                f"Invalid linear operator: {linear_operator}. "
                "Must be one of: "
                "'fc', 'conv', 'identity'"
            )

        # Initialize activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU(**activation_function_kwargs)
        elif activation_function == "leaky_relu":
            self.activation_function = nn.LeakyReLU(**activation_function_kwargs)
        elif activation_function == "maxpool":
            self.activation_function = nn.MaxPool2d(
                kernel_size=2, **activation_function_kwargs
            )
        elif activation_function == "identity":
            self.activation_function = nn.Identity(**activation_function_kwargs)
        else:
            raise ValueError(
                f"Invalid activation function: {activation_function}. "
                "Must be one of: 'relu', 'leaky_relu', 'abs', 'maxpool'"
            )

        self.batch_norm = batch_norm
        if self.batch_norm and isinstance(self.linear_operator, nn.Linear):
            self.bn = nn.BatchNorm1d(self.linear_operator.out_features)
        elif self.batch_norm and isinstance(self.linear_operator, nn.Conv2d):
            self.bn = nn.BatchNorm2d(self.linear_operator.out_channels)

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
        if self.batch_norm:
            x = self.bn(x)
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

    def vq_pdist(self, x: Tensor) -> Tensor:
        """
        Returns the number of shared partitions between two tensors.

        Parameters:
            x (torch.Tensor): First tensor

        Returns:
            torch.Tensor: Number of shared partitions
        """
        if isinstance(self.activation_function, nn.MaxPool2d):
            x = self.linear_operator(x)
            if x.shape[2] % 2 != 0:
                x = x[:, :, :-1, :]
            if x.shape[3] % 2 != 0:
                x = x[:, :, :, :-1]
            x = einops.rearrange(
                x,
                "b c (h p1) (w p2) -> b (c h w) (p1 p2)",
                p1=self.activation_function.kernel_size,
                p2=self.activation_function.kernel_size,
            )

            x = torch.argmax(x, dim=-1)

            x = nn.functional.one_hot(x, 4)

            x = einops.rearrange(x, "b a d -> b (a d)").to(torch.float)
            n = x.shape[1]
            d = nn.functional.pdist(x, p=1) / np.sqrt(n)
            return d

        elif isinstance(self.linear_operator, nn.Linear):
            x = self.linear_operator(x) > 0
            x = torch.concatenate([x, torch.logical_not(x)], dim=1)
            x = x.to(torch.int).to(torch.float)
            n = x.shape[1]
            d = nn.functional.pdist(x, p=1) / np.sqrt(n)
            return d
        elif isinstance(self.linear_operator, nn.Conv2d):
            n = x.shape[0]
            x = self.linear_operator(x) > 0
            x = x.reshape(n, -1).to(torch.float)
            n = x.shape[1]
            d = nn.functional.pdist(x, p=1) / np.sqrt(n)
            return d

    def get_local_partitions_descriptor(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        input_: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns a tensor that describes the local partitions induced by the
        layer. When the activation function is a ReLU or LeakyReLU, the
        partitions are described by the affine parameters of the linear operator.
        When the activation function is a max pooling layer, the partitions are
        described by the partition assignment itself (since it is input dependent)

        Parameters:
            k (int): Index of the dimension
            input_shape (tuple): Shape of the input tensor to the layer
                required only if the linear operator is a convolutional layer.
            input_ (torch.Tensor): Input tensor to the layer required only if
                the activation is a max pooling layer.

        Returns:
            torch.Tensor: Local partition descriptor
        """
        if isinstance(self.linear_operator, nn.Linear):
            A = self.linear_operator.weight

            if self.linear_operator.bias is not None:
                b = self.linear_operator.bias
            else:
                b = torch.zeros(A.shape[0])
        elif isinstance(self.linear_operator, nn.Conv2d):
            A, b = utils.conv2d_to_linear(self.linear_operator, input_shape)
        elif isinstance(self.linear_operator, nn.Identity):
            A = torch.eye(input_.shape[-1])
            b = torch.zeros(input_.shape[-1])
        else:
            raise ValueError("The linear operator must be an instance of nn.Linear")

        if isinstance(self.activation_function, nn.MaxPool2d):
            input_ = self.linear_operator(input_)
            input_shape = input_.shape
            # Make sure the height and width are even
            if input_shape[2] % 2 != 0:
                input_ = input_[..., :-1, :]
            if input_shape[3] % 2 != 0:
                input_ = input_[..., :-1]

            input_ = einops.rearrange(
                input_,
                "b c (h p1) (w p2) -> b (c h w) (p1 p2)",
                p1=self.activation_function.kernel_size,
                p2=self.activation_function.kernel_size,
            )
            h = input_shape[2] // self.activation_function.kernel_size
            w = input_shape[3] // self.activation_function.kernel_size
            input_argmax = input_.argmax(dim=-1)
            unique, partition_idx = input_argmax.unique(dim=0, return_inverse=True)

            n_partitions = unique.shape[0]
            partitions = torch.zeros((input_.shape[0], n_partitions))
            partitions[torch.arange(input_.shape[0]), partition_idx] = 1
            return partitions.to(torch.bool)

        return A, b

    def assign_local_partitions(
        self, x: Tensor, remove_redundant: bool = True
    ) -> Tensor:
        """
        Takes a tensor of input values and assigns each value to a local
        partition.

        Parameters:
            x (torch.Tensor): Input tensor
            remove_redundant (bool): If True, drops partitions with no members.

        Returns:
            one-hot encoding of the assignment.
        """
        input_shape_ = x.shape
        if isinstance(self.activation_function, nn.Identity):
            return torch.ones(size=(x.shape[0], 1), dtype=torch.bool)
        if isinstance(self.linear_operator, nn.Conv2d):
            x = x.reshape(x.shape[0], -1)
        if isinstance(self.activation_function, nn.MaxPool2d):
            return self.get_local_partitions_descriptor(
                input_shape=input_shape_[1:], input_=x
            )
        A, b = self.get_local_partitions_descriptor(
            input_shape=input_shape_[1:], input_=x
        )
        z = x @ A.T + b
        if remove_redundant:
            # Convert each row from binary to decimal
            z_bool = (z > 0).to(torch.int)
            z_integers = utils.bin_to_dec(z_bool.detach().numpy())
            unique, partition_idx = np.unique(z_integers, return_inverse=True)
            n_partitions = len(unique)
            partitions = torch.zeros(size=(z.shape[0], n_partitions), dtype=torch.bool)
            for i in range(z.shape[0]):
                partitions[i, partition_idx[i]] = True
            return partitions
        else:
            partitions = list()
            element_sum = torch.zeros(size=(z.shape[0],))
            for comb in tqdm(utils.iter_combinations(z.shape[1])):
                r = torch.ones(size=(z.shape[0],))
                for column in range(z.shape[1]):
                    col = z[:, column] > 0
                    if not comb[column]:
                        col = torch.logical_not(col)
                    r = torch.logical_and(r, col)
                partitions.append(r)
                element_sum += r
                if element_sum.all():
                    break
            partitions = torch.stack(partitions, dim=1)
            return partitions


class MASOLinear(MASOLayer):
    """
    This class defines a fully connected MASO layer.

    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        activation_function (str): Name of the activation function
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_function: str = "relu",
        bias: bool = True,
    ) -> None:
        operator_kwargs = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
        }
        super().__init__("fc", operator_kwargs, activation_function)


class MASOConv2d(MASOLayer):
    """
    This class defines a convolutional MASO layer.

    Parameters:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution
        padding (int): Padding of the convolution
        activation_function (str): Name of the activation function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_function: str = "relu",
        bias: bool = True,
        batch_norm: bool = False,
    ) -> None:
        operator_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias,
        }
        super().__init__("conv", operator_kwargs, activation_function, batch_norm=batch_norm)


class MASOMaxPool2d(MASOLayer):
    """
    This class defines a max pooling MASO layer.

    Parameters:
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution
        padding (int): Padding of the convolution
    """

    def __init__(self) -> None:
        super().__init__("identity", activation_function="maxpool")


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
        for idx in trange(l):
            layer = self[idx]
            local_partitions = layer.assign_local_partitions(
                z, remove_redundant=remove_redundant
            )
            if remove_redundant:
                global_partitions = torch.stack(global_partitions, dim=1)
                p = torch.concatenate([global_partitions, local_partitions], dim=1)

                # Convert each row from binary to decimal
                p = utils.bin_to_dec(p.detach().numpy())
                unique, partition_idx = np.unique(p, return_inverse=True)
                n_partitions = len(unique)
                partitions = torch.zeros(
                    size=(z.shape[0], n_partitions), dtype=torch.bool
                )
                for i in trange(z.shape[0]):
                    partitions[i, partition_idx[i]] = True

                global_partitions = [partitions[:, i] for i in range(n_partitions)]

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

    def assign_local_partitions(
        self, x: Tensor, remove_redundant: bool = True
    ) -> Tensor:
        """
        Takes a tensor of input values and assigns each input to a local partition
        for the layer.

        Parameters:
            x (torch.Tensor): Input tensor
            remove_redundant (bool): If True, drops partitions with no members.

        Returns:
            list[Tensor]: List of tensors indicating the local partition
                assignments for each input
        """
        z = x
        local_partitions = list()
        for layer in self:
            local_partitions.append(layer.assign_local_partitions(z))
            z = layer(z)
        return local_partitions

    def layer_local_vq_distance(
        self, x: Tensor, max_layer: Optional[int] = None
    ) -> List[Tensor]:
        """
        For each layer, calculates the pairwise VQ distance matrix.

        Parameters:
            x (torch.Tensor): Input tensor
            max_layer (int): Index of the last layer to use. If None, all layers
                are used.
        """
        z = x

        if max_layer is None:
            max_layer = len(self) - 1

        local_vq_distances = list()
        l2_dist = list()
        for idx in trange(max_layer + 1):
            layer = self[idx]
            d = layer.vq_pdist(z)
            d = d.detach().numpy()
            d = squareform(d)
            local_vq_distances.append(d)
            z = layer(z)
            n = z.shape[0]
            layer_output = z.detach().numpy().reshape(n, -1)
            l2 = pdist(layer_output, metric="cosine")
            l2 = squareform(l2)
            l2_dist.append(l2)
        return local_vq_distances, l2_dist


def fc_network(n_feature: Tuple[int, ...] = (2, 4, 4, 1), bias: bool = True, bn: bool = False) -> MASODN:
    """
    Returns a fully connected network with the specified number of features.

    Parameters:
        n_feature (tuple): Number of features per layer
        bias (bool): If True, use bias terms
        bn (bool): If True, use batch normalization

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
                    "bias": bias,
                },
                activation_function=activation_function,
                batch_norm=bn,
            )
        )
    return MASODN(*layers)


def smallCNN(input_channels: int, bias: bool = True, batch_norm: bool = False) -> MASODN:
    """
    Implements a small CNN.
    """
    layers = [
        MASOConv2d(
            input_channels, 32, 3, padding=0, activation_function="relu", bias=bias,
            batch_norm=batch_norm
        ),
        MASOMaxPool2d(),
        MASOConv2d(32, 64, 3, padding=0, activation_function="relu", bias=bias, batch_norm=batch_norm),
        MASOMaxPool2d(),
        MASOConv2d(64, 128, 1, padding=0, activation_function="relu", bias=bias, batch_norm=batch_norm),
    ]
    return MASODN(*layers)


def largeCNN(input_channels: int, bias: bool = True) -> MASODN:
    """
    Implements a large CNN.
    """
    layers = [
        MASOConv2d(
            input_channels, 96, 3, padding=0, activation_function="relu", bias=bias
        ),
        MASOConv2d(96, 96, 3, padding=1, activation_function="relu", bias=bias),
        MASOConv2d(96, 96, 3, padding=1, activation_function="relu", bias=bias),
        MASOMaxPool2d(),
        MASOConv2d(96, 192, 3, padding=0, activation_function="relu", bias=bias),
        MASOConv2d(192, 192, 3, padding=1, activation_function="relu", bias=bias),
        MASOConv2d(192, 192, 3, padding=0, activation_function="relu", bias=bias),
        MASOMaxPool2d(),
        MASOConv2d(192, 192, 3, padding=0, activation_function="relu", bias=bias),
        MASOConv2d(192, 192, 1, padding=0, activation_function="relu", bias=bias),
        MASOMaxPool2d(),
    ]
    return MASODN(*layers)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from scipy import interpolate

    x, y = load_digits(return_X_y=True)

    net = fc_network(n_feature=(64, 16))
    x = torch.from_numpy(x).float() / 255.0

    partitions = net.layer_local_vq_distance(x)
    for p in partitions:
        plt.matshow(p)
        plt.show()
