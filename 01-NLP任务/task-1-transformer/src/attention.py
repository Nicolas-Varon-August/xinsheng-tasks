"""手写的缩放点积注意力与多头注意力。

接口约定（见 task-1-transformer/README.upstream.md「实现约定」）：
    scaled_dot_product_attention(Q, K, V, mask=None)
        Q/K/V 形状 (B, H, T, D)；mask 可广播到 (B, H, T, T)，
        布尔张量中 True 表示该位置被屏蔽。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention_scores(Q, K, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 关键点：用 -inf 而不是乘 0，否则 softmax 之后被屏蔽位置仍分到概率
        scores = scores.masked_fill(mask, float("-inf"))
    return scores


def scaled_dot_product_attention(Q, K, V, mask=None):
    """注意力核心：softmax(Q K^T / sqrt(d_k)) V，形状 (B, H, T, D)。"""
    scores = _attention_scores(Q, K, mask)
    attn = F.softmax(scores, dim=-1)
    # 整行被屏蔽时 softmax 会产生 NaN，这里兜底为 0
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, V)


def attention_weights(Q, K, mask=None):
    """与上面同一套打分逻辑，但返回权重矩阵；仅供可视化使用。"""
    scores = _attention_scores(Q, K, mask)
    attn = F.softmax(scores, dim=-1)
    return torch.nan_to_num(attn, nan=0.0)


class MultiHeadAttention(nn.Module):
    """多头自注意力：线性投影 -> 分头 -> 缩放点积 -> 拼头 -> 输出投影。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x, B, T):
        # (B, T, d_model) -> (B, H, T, d_k)
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()

    def forward(self, x, mask=None, return_weights: bool = False):
        B, T, _ = x.shape
        Q = self._split_heads(self.w_q(x), B, T)
        K = self._split_heads(self.w_k(x), B, T)
        V = self._split_heads(self.w_v(x), B, T)

        if return_weights:
            weights = attention_weights(Q, K, mask)      # (B, H, T, T)
            out = torch.matmul(weights, V)
        else:
            weights = None
            out = scaled_dot_product_attention(Q, K, V, mask)

        # (B, H, T, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(out)
        if return_weights:
            return out, weights
        return out


def build_causal_mask(T: int, device=None) -> torch.Tensor:
    """上三角为 True 的因果掩码，(T, T)，True 表示「未来位置，屏蔽」。"""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def build_padding_mask(ids, pad_id: int = 0) -> torch.Tensor:
    """由 id 张量 (B, T) 生成 key 侧 padding 掩码，形状 (B, 1, 1, T)。"""
    return (ids == pad_id)[:, None, None, :]
