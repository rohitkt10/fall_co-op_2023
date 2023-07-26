import torch 
from torch import nn 
from torch.nn import functional as F
from copy import deepcopy

class Activation(nn.Module):
    """
    Convert an abitrary python callable into a torch
    nn.Module instance.
    """
    def __init__(self, fn):
        super().__init__()
        assert callable(fn) and (not isinstance(fn, nn.Module))
        self.fn = fn
    def forward(self, input):
        return self.fn(input)
    def __repr__(self):
        return f"{self.fn.__name__}()"

def get_activation(activation, activation_kwargs={}):
    # activation is a nn.Module subclass type
    if type(activation)==activation and issubclass(activation, nn.Module):
        out = activation(**activation_kwargs)

    # activation is an instance of nn.Module
    elif isinstance(activation, nn.Module):
        out = deepcopy(activation)

    # activation is a python callable
    elif callable(activation):
        out = Activation(activation)
    else:
        raise ValueError("Invalid format for activation function.")
    return out

def conv_block(in_channels, out_channels, kernel_size, bias=True, bn=True, dropout=None, activation=None, activation_kwargs={}):
    layers = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', bias=bias)]
    if bn:
        layers.append(nn.BatchNorm1d(num_features=out_channels))
    if activation:
        activation = get_activation(activation, activation_kwargs)
        layers.append(activation)
    if dropout:
        assert dropout > 0. and dropout < 1.
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def dense_block(in_features, out_features, bias=True, bn=True, dropout=None, activation=None, activation_kwargs={}):
    layers = [nn.Linear(in_features, out_features, bias)]
    if bn:
        layers.append(nn.BatchNorm1d(num_features=out_features))
    if activation:
        activation = get_activation(activation, activation_kwargs)
        layers.append(activation)
    if dropout:
        assert dropout > 0. and dropout < 1.
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class DeepCNNModel(nn.Module):
    """
    A basic convolutional Neural Network model with 3 blocks of convolutions followed
    by 2 blocks of dense connections.

    Architecture:

    Input -> Conv -> Conv -> Pool -> Conv -> Pool -> Dense -> Dense-> Output.
    """
    def __init__(
                self,
                sequence_length=200,
                filter_sizes=[24, 32, 48, 64],
                conv_kernel_sizes = [19, 7, 5, 5],
                dropouts = [0.1, 0.2, 0.3, 0.4, 0.5],
                pool_sizes = [4, 4, 4],
                bn=True,
                activation=F.relu,
                activation_kwargs={}
                ):
        super().__init__()

        self.conv1 = conv_block(4, filter_sizes[0], kernel_size=conv_kernel_sizes[0], bias=True, \
                                bn=bn, dropout=dropouts[0], activation=activation, activation_kwargs=activation_kwargs)
        self.conv2 = conv_block(filter_sizes[0], filter_sizes[1], kernel_size=conv_kernel_sizes[1], \
                                bias=True, bn=bn, dropout=dropouts[1], activation=activation, activation_kwargs=activation_kwargs)
        self.pool1 = nn.MaxPool1d(pool_sizes[0])
        self.conv3 = conv_block(filter_sizes[1], filter_sizes[2], kernel_size=conv_kernel_sizes[2], \
                                bias=True, bn=bn, dropout=dropouts[2], activation=activation, activation_kwargs=activation_kwargs)
        self.pool2 = nn.MaxPool1d(pool_sizes[1])
        self.conv4 = conv_block(filter_sizes[2], filter_sizes[3], kernel_size=conv_kernel_sizes[3], \
                                bias=True, bn=bn, dropout=dropouts[3], activation=activation, activation_kwargs=activation_kwargs)
        self.pool3 = nn.MaxPool1d(pool_sizes[2])
        self.flatten = nn.Flatten(start_dim=1)
        num_features = sequence_length
        for p in pool_sizes:
            num_features = num_features // p
        num_features *= filter_sizes[-1]
        self.dense1 = dense_block(in_features=num_features, out_features=96, activation=F.relu, dropout=dropouts[-1])
        self.dense2 = nn.Linear(in_features=96, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x