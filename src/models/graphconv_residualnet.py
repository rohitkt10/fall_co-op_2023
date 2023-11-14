import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ChebConv, global_add_pool


class GraphConvResidualNet(nn.Module):
    """
    A 4-layer graph convolutional network with residual connections.
    """
    def __init__(self, dim, num_features, num_classes, num_layers=4, conv_type=GraphConv, dropout=0.5):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes 
        self.dim = dim
        self.conv_type = conv_type
        self.dropout = dropout
        self.num_layers = num_layers

        # Create convolution layers dynamically based on the number of layers
        self.conv_layers = nn.ModuleList([
            self.conv_type(num_features, dim) if i == 0 else self.conv_type(dim, dim)
            for i in range(num_layers)
        ])

        # Create batch normalization layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(num_layers)])

        if conv_type == ChebConv:
            self.conv1 = self.conv_type(num_features, dim, K=2) # try K=3,
        else:
            self.conv1 = self.conv_type(num_features, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        if conv_type == ChebConv:
            self.conv2 = self.conv_type(dim, dim, K=2)
        else:
            self.conv2 = self.conv_type(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        if conv_type == ChebConv:
            self.conv3 = self.conv_type(dim, dim, K=2)
        else:
            self.conv3 = self.conv_type(dim, dim)
        self.bn3 = nn.BatchNorm1d(dim)
        if conv_type == ChebConv:
            self.conv4 = self.conv_type(dim, dim, K=2)
        else:
            self.conv4 = self.conv_type(dim, dim)
        self.bn4 = nn.BatchNorm1d(dim)

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, self.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        # Apply the convolution layers with residual connections
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_weight)
            x = self.bn_layers[i](x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout layer
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the last classification layer
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

