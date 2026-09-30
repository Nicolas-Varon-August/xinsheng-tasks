"""Transformer 分类模型，以及自检脚本需要的 load_for_eval 工厂函数。"""
import math
from pathlib import Path

import torch
import torch.nn as nn

from .attention import build_padding_mask
from .block import TransformerBlock
from .tokenizer import PAD_ID, CharTokenizer


class SinusoidalPositionalEncoding(nn.Module):
    """经典正弦位置编码（不可学习），直接加到词嵌入上。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    """N 层 Transformer encoder + 池化 + 线性分类头。"""

    def __init__(self, vocab_size: int, num_classes: int = 2, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, d_ff: int = 256,
                 max_len: int = 256, dropout: float = 0.1, pad_id: int = PAD_ID,
                 pooling: str = "mean", use_pe: bool = True):
        super().__init__()
        assert pooling in ("mean", "cls")
        self.pad_id = pad_id
        self.pooling = pooling
        self.max_len = max_len
        self.use_pe = use_pe

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout) if use_pe else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            with torch.no_grad():
                module.weight[PAD_ID].zero_()

    def encode(self, ids, return_attn: bool = False):
        """把 id 序列编码成句向量；return_attn=True 时同时返回各层注意力权重。"""
        mask = build_padding_mask(ids, self.pad_id)      # (B, 1, 1, T)
        x = self.token_emb(ids)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.drop(x)

        all_weights = []
        for block in self.blocks:
            if return_attn:
                x, w = block(x, mask, return_weights=True)
                all_weights.append(w)
            else:
                x = block(x, mask)
        x = self.ln_f(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            keep = (ids != self.pad_id).unsqueeze(-1).to(x.dtype)   # (B, T, 1)
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

        if return_attn:
            return pooled, all_weights
        return pooled

    def forward(self, ids, return_attn: bool = False):
        if return_attn:
            pooled, weights = self.encode(ids, return_attn=True)
            return self.classifier(pooled), weights
        return self.classifier(self.encode(ids))


def build_model(config: dict) -> TransformerClassifier:
    """按配置字典构造模型（train.py 与 load_for_eval 共用）。"""
    keys = ("vocab_size", "num_classes", "d_model", "n_heads", "n_layers",
            "d_ff", "max_len", "dropout", "pooling", "use_pe")
    kwargs = {k: config[k] for k in keys if k in config}
    return TransformerClassifier(**kwargs)


def load_for_eval(ckpt_path: str):
    """自检脚本入口：返回 (model, tokenize_fn)，模型在 CPU 上且已 eval()。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])

    def tokenize_fn(text: str) -> torch.Tensor:
        return tokenizer.encode(text)

    return model, tokenize_fn


def save_checkpoint(path, model, tokenizer, config, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path
