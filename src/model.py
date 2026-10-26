"""
FedShield — GCN Model Architecture
=====================================
A 2-layer Graph Convolutional Network (GCN) for node-level binary
classification (fraud vs. non-fraud).

Design decisions:
  - Kept intentionally simple per the brief: 2 GCNConv layers.
  - Dropout between layers to reduce overfitting on the imbalanced data.
  - Output is a 2-class logit vector (non-fraud, fraud); CrossEntropyLoss
    is applied externally with class weights to handle imbalance.
  - Hidden dimension is configurable for easy experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FraudGCN(nn.Module):
    """
    2-layer GCN for node-level fraud classification.

    Architecture:
        Input (10 features)
            |
        GCNConv(10 -> hidden_dim)
            |
        ReLU + Dropout
            |
        GCNConv(hidden_dim -> hidden_dim // 2)
            |
        ReLU + Dropout
            |
        Linear(hidden_dim // 2 -> 2)
            |
        Logits [non-fraud, fraud]
    """

    def __init__(self, in_channels: int = 10, hidden_dim: int = 64, dropout: float = 0.3):
        super(FraudGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification head
        x = self.classifier(x)
        return x

    def get_embeddings(self, x, edge_index):
        """Return node embeddings (before final linear) — useful for analysis."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
