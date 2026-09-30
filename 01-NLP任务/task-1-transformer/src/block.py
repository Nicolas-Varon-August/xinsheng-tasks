"""Transformer Block：多头注意力 + 前馈网络 + 两处残差 + 两处 LayerNorm。"""
import torch.nn as nn

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络：d_model -> d_ff -> d_model。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN 结构：x = x + Attn(LN(x))；x = x + FFN(LN(x))。

    Pre-LN 相比原论文的 Post-LN 更容易训稳，不加 warmup 也不容易发散。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None, return_weights: bool = False):
        if return_weights:
            h, w = self.attn(self.ln1(x), mask, return_weights=True)
            x = x + h
            x = x + self.ffn(self.ln2(x))
            return x, w
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
