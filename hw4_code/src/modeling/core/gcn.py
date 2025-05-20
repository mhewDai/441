import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_layers: int = 2):
        super(GCN, self).__init__()
        # TODO: add 2 layers of GCN

        self.dropout = dropout

        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            # print("Hidden layer is added :D")
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.layers.append(GCNLayer(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # given the input node features, and the adjacency matrix, run GCN
        # The order of operations should roughly be:
        # 1. Apply the first GCN layer
        # 2. Apply Relu
        # 3. Apply Dropout
        # 4. Apply the second GCN layer

        # TODO: your code here

        for i in range(len(self.layers)):
            x = self.layers[i](x, adj)

            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout)

        return x

