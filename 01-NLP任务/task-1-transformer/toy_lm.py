"""DoD 的 M4：给同一个注意力加上因果掩码，跑一个字符级 toy 语言模型。

用法（在 task-1-transformer 目录下执行）：
    python toy_lm.py
语料：data/poetryFromTang.txt（唐诗，约 4.7 万行）
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.attention import build_causal_mask, scaled_dot_product_attention  # noqa: E402
from src.block import TransformerBlock                                     # noqa: E402
from src.console import setup_console                                      # noqa: E402


class CausalLM(nn.Module):
    """极简 decoder-only 语言模型：词嵌入 + 位置嵌入 + N 层因果 Transformer。"""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=256, max_len=64, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok(ids) + self.pos(pos))
        mask = build_causal_mask(T, device=ids.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))


class CharLMData(Dataset):
    def __init__(self, ids, block=64):
        self.ids = ids
        self.block = block

    def __len__(self):
        return max(0, len(self.ids) - self.block - 1)

    def __getitem__(self, i):
        chunk = self.ids[i: i + self.block + 1]
        return chunk[:-1], chunk[1:]


def check_no_leakage():
    """验证因果掩码：改未来位置的 V，过去位置的输出不变。"""
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 6, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = build_causal_mask(T)

    out = scaled_dot_product_attention(Q, K, V, mask)
    V2 = V.clone()
    V2[:, :, -1] = 999.0
    out2 = scaled_dot_product_attention(Q, K, V2, mask)
    leaked = (out[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
    print(f"[因果掩码自检] 过去位置输出最大差异 = {leaked:.3e} "
          f"({'通过' if leaked < 1e-6 else '失败'})")
    return leaked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(ROOT / "data" / "poetryFromTang.txt"))
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-len", type=int, default=40)
    args = parser.parse_args()

    setup_console()
    leaked = check_no_leakage()

    corpus = Path(args.corpus)
    if not corpus.exists():
        sys.exit(f"找不到语料 {corpus}")
    text = corpus.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(len(data) * 0.95)
    print(f"语料长度 = {len(text)} 字符，字表 = {len(chars)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CharLMData(data[:n_train], args.block)
    val_ds = CharLMData(data[n_train:], args.block)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = CausalLM(len(chars), d_model=args.d_model, n_layers=args.n_layers,
                     max_len=args.block).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def perplexity(loader):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x).reshape(-1, len(chars)), y.reshape(-1))
                total += loss.item()
                count += 1
        model.train()
        return math.exp(total / max(1, count))

    print(f"device={device}  随机初始化困惑度 ≈ {perplexity(val_loader):.1f}")
    t0 = time.time()
    step = 0
    while step < args.steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).reshape(-1, len(chars)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{args.steps} loss={loss.item():.4f} "
                      f"ppl={math.exp(loss.item()):.1f} elapsed={time.time()-t0:.1f}s")
            if step >= args.steps:
                break
    ppl = perplexity(val_loader)
    print(f"\n训练完成，验证集困惑度 = {ppl:.2f}（{args.steps} 步，"
          f"耗时 {time.time()-t0:.1f}s）")

    model.eval()
    prompt = "床前明月光"
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    with torch.no_grad():
        for _ in range(args.sample_len):
            logits = model(ids[:, -args.block:])[:, -1]
            nxt = torch.multinomial(torch.softmax(logits / 0.8, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
    print("生成样例：" + "".join(itos[int(i)] for i in ids[0]))
    print(f"因果掩码自检差异 = {leaked:.3e}")


if __name__ == "__main__":
    main()
