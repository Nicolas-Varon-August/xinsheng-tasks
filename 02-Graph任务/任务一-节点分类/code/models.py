"""四种主流 GNN 的节点分类模型（统一接口，便于横向对比）。

接口：model(x, edge_index) -> logits，形状 (N, num_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class GNN(nn.Module):
    """按 conv 类型堆叠 N 层 GNN，层间使用 ReLU + Dropout。"""

    def __init__(self, name: str, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        assert num_layers >= 2, "至少两层才能体现邻居聚合"
        self.name = name
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(num_layers):
            is_last = layer == num_layers - 1
            in_c = in_dim if layer == 0 else hidden_dim
            out_c = out_dim if is_last else hidden_dim
            self.convs.append(self._make_conv(name, in_c, out_c, heads))
            self.norms.append(nn.LayerNorm(out_c))

    @staticmethod
    def _make_conv(name, in_c, out_c, heads):
        if name == "gcn":
            return GCNConv(in_c, out_c)
        if name == "sage":
            return SAGEConv(in_c, out_c)
        if name == "gat":
            return GATConv(in_c, out_c, heads=heads, concat=False)
        if name == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c)
            )
            return GINConv(mlp, train_eps=True)
        raise ValueError(f"未知模型：{name}")

    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_model(name: str, in_dim: int, hidden_dim: int, out_dim: int,
                num_layers: int = 2, heads: int = 4, dropout: float = 0.5) -> GNN:
    return GNN(name, in_dim, hidden_dim, out_dim, num_layers, heads, dropout)
