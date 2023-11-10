import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ChebConv, global_add_pool


class GraphConvResidualNet(nn.Module):
    """
    A 4-layer graph convolutional network with residual connections.
    """
    def __init__(self, dim, num_features, num_classes, conv_type=GraphConv, dropout=0.5):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes 
        self.dim = dim
        self.conv_type = conv_type
        self.dropout = dropout

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
        # Apply the first convolution layer
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = self.bn1(x1)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Apply the second convolution layer with residual connection
        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = self.bn2(x2)
        x2 = x2.relu()
        x2 = x2 + x1  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Apply the third convolution layer with residual connection
        x3 = self.conv3(x2, edge_index, edge_weight)
        x3 = self.bn3(x3)
        x3 = x3.relu()
        x3 = x3 + x2  # Residual connection
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # Apply the fourth convolution layer with residual connection
        x4 = self.conv4(x3, edge_index, edge_weight)
        x4 = self.bn4(x4)
        x4 = x4.relu()
        x4 = x4 + x3  # Residual connection
        x4 = F.dropout(x4, p=self.dropout, training=self.training)

        # Readout layer
        x = global_add_pool(x4, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the last classification layer
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

