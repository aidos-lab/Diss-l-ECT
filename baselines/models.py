import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid, Reddit, WebKB
from torch_geometric.nn.norm import LayerNorm
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