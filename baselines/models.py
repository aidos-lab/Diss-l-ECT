import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing, GINConv
from torch_geometric.datasets import Planetoid, Reddit, WebKB
from torch_geometric.nn.norm import LayerNorm
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops, degree
from torch.nn import Linear, Sequential
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc_final = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # First convolution layer
        x = F.relu(self.conv1(x, edge_index))
        x_clone = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x_clone)  # Skip connection
        x = self.norm1(x)                  # Layer normalization

        # Second convolution layer
        x_clone = x
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(self.fc3(x))
        x = self.fc4(x) + x_clone  # Skip connection
        x = self.norm2(x)          # Layer normalization

        # Final classification layer
        x = self.fc_final(x)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels=8, heads=8):
        super(GAT, self).__init__()

        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=heads, dropout=0.2)
        self.norm1 = LayerNorm(hidden_channels * heads)
        self.fc1 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.fc2 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads)
        self.norm2 = LayerNorm(hidden_channels * heads)
        self.fc3 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.fc4 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.fc_final = torch.nn.Linear(hidden_channels * heads, dataset.num_classes)

    def forward(self, x, edge_index):
        # First GAT layer
        x = F.relu(self.conv1(x, edge_index))
        x_clone = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x_clone)  # Skip connection
        x = self.norm1(x)  # Layer normalization

        # Second GAT layer
        x_clone = x
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(self.fc3(x))
        x = self.fc4(x) + x_clone  # Skip connection
        x = self.norm2(x)  # Layer normalization

        # Final classification layer
        x = self.fc_final(x)
        return F.log_softmax(x, dim=1)

class H2GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels=64, dropout=0.5, use_bn=True):
        super(H2GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.use_bn = use_bn

        # Separate transformations for ego and neighbor features
        self.ego_linear = nn.Linear(dataset.num_features, hidden_channels, bias=False)
        self.neigh_linear = nn.Linear(dataset.num_features, hidden_channels, bias=False)

        # Higher-order aggregation layers
        self.hop1_linear = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.hop2_linear = nn.Linear(hidden_channels, hidden_channels, bias=False)

        # Batch normalization
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Final classification layer
        self.classifier = nn.Linear(2 * hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # Add self-loops for ego embeddings
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Ego and neighbor feature transformations
        x_ego = self.ego_linear(x)
        x_neigh = self._aggregate_neighbors(x, edge_index)
        x_neigh = self.neigh_linear(x_neigh)

        # Higher-order propagation
        x_1hop = F.relu(self.hop1_linear(x_ego + x_neigh))
        if self.use_bn:
            x_1hop = self.bn1(x_1hop)
        x_1hop = F.dropout(x_1hop, p=self.dropout, training=self.training)

        x_2hop = F.relu(self.hop2_linear(self._aggregate_neighbors(x_1hop, edge_index)))
        if self.use_bn:
            x_2hop = self.bn2(x_2hop)
        x_2hop = F.dropout(x_2hop, p=self.dropout, training=self.training)

        # Concatenate 1-hop and 2-hop features
        x_out = torch.cat([x_1hop, x_2hop], dim=1)

        # Final classification
        x_out = self.classifier(x_out)
        return F.log_softmax(x_out, dim=1)

    def _aggregate_neighbors(self, x, edge_index):
        # Aggregate features from neighbors
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype).clamp(min=1)
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        return out / deg.view(-1, 1)  # Normalize by degree

class GIN(torch.nn.Module):
    def __init__(self, dataset=None, hidden_channels=64):
        super(GIN, self).__init__()
        if dataset is None:
            raise ValueError("Dataset must be provided to infer model dimensions.")

        # Infer feature and class dimensions from the dataset
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes

        # Define GIN layers
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
        )
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # No global pooling for node-level tasks
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
