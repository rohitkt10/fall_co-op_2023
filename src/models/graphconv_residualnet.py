import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ChebConv, global_add_pool


class GraphConvResidualNet(nn.Module):
    """
    A 4-layer graph convolutional network with residual connections.
    """
    def __init__(self, dim, num_features, num_classes, num_layers=4, conv_type=GraphConv, dropout=0.5, conv_kwargs=None):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes 
        self.dim = dim
        self.conv_type = conv_type
        self.dropout = dropout
        self.num_layers = num_layers
        self.conv_kwargs = conv_kwargs if conv_kwargs is not None else {}

        # Create convolution layers dynamically based on the number of layers
        self.conv_layers = nn.ModuleList([
            self._create_conv_layer(num_features, dim, **self.conv_kwargs) if i == 0
            else self._create_conv_layer(dim, dim, **self.conv_kwargs)
            for i in range(num_layers)
        ])

        # Create batch normalization layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(num_layers)])

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, self.num_classes)

    def _create_conv_layer(self, in_channels, out_channels, **kwargs):
        if self.conv_type == ChebConv:
            return self.conv_type(in_channels, out_channels, K=kwargs.get('K', 2))
        else:
            return self.conv_type(in_channels, out_channels, **kwargs)

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

