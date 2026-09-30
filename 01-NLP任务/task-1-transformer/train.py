"""在 ChnSentiCorp 上训练手写 Transformer 情感分类器。

用法（在 task-1-transformer 目录下执行）：
    python train.py
    python train.py --epochs 8 --d-model 128 --n-heads 4 --n-layers 4
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import build_model, save_checkpoint          # noqa: E402
from src.tokenizer import CharTokenizer, PAD_ID, pad_batch  # noqa: E402
from src.console import setup_console                       # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: CharTokenizer):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.texts[idx]), self.labels[idx]


def make_collate(pad_id: int = PAD_ID):
    def collate(batch):
        seqs, labels = zip(*batch)
        ids = pad_batch(list(seqs), pad_id)
        return ids, torch.tensor(labels, dtype=torch.long)
    return collate


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """先线性 warmup 再余弦衰减到 0 的 lr 调度。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / max(1, total), loss_sum / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(ROOT / "ckpt" / "best.pt"))
    parser.add_argument("--tag", default="base", help="实验标签，用于日志文件名")
    args = parser.parse_args()

    setup_console()
    set_seed(args.seed)
    device = pick_device(args.device)

    data_dir = Path(args.data_dir)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    dev_df = pd.read_parquet(data_dir / "validation.parquet")
    print(f"device={device}  train={len(train_df)}  dev={len(dev_df)}")

    tokenizer = CharTokenizer.build(train_df["text"].astype(str).tolist(),
                                    max_len=args.max_len, min_freq=args.min_freq)
    print(f"词表大小 = {tokenizer.vocab_size}")

    train_loader = DataLoader(SentimentDataset(train_df, tokenizer), batch_size=args.batch_size,
                              shuffle=True, collate_fn=make_collate(), num_workers=0)
    dev_loader = DataLoader(SentimentDataset(dev_df, tokenizer), batch_size=128,
                            shuffle=False, collate_fn=make_collate(), num_workers=0)

    config = dict(vocab_size=tokenizer.vocab_size, num_classes=2, d_model=args.d_model,
                  n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
                  max_len=args.max_len, dropout=args.dropout, pooling=args.pooling,
                  use_pe=True)
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 = {n_params/1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_acc = 0.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        correct = 0
        for step, (ids, labels) in enumerate(train_loader, 1):
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += labels.size(0)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running/seen:.4f} acc={correct/seen:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        dev_acc, dev_loss = evaluate(model, dev_loader, device)
        train_acc = correct / max(1, seen)
        history.append(dict(epoch=epoch, train_loss=round(running / max(1, seen), 4),
                            train_acc=round(train_acc, 4), dev_loss=round(dev_loss, 4),
                            dev_acc=round(dev_acc, 4)))
        print(f"[epoch {epoch}] train_loss={running/seen:.4f} train_acc={train_acc:.4f} "
              f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} "
              f"elapsed={time.time()-t0:.1f}s")

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            save_checkpoint(args.out, model, tokenizer, config,
                            extra={"dev_acc": dev_acc, "epoch": epoch,
                                   "history": history, "args": vars(args)})
            print(f"  -> 保存新的最优 checkpoint（dev_acc={dev_acc:.4f}）")

    total_time = time.time() - t0
    summary = dict(tag=args.tag, best_dev_acc=round(best_acc, 4), best_epoch=best_epoch,
                   params_m=round(n_params / 1e6, 3), vocab_size=tokenizer.vocab_size,
                   train_seconds=round(total_time, 1), device=str(device),
                   config=config, args=vars(args), history=history)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"train_log_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最佳 dev 准确率 = {best_acc:.4f}（epoch {best_epoch}），"
          f"总耗时 {total_time:.1f}s")
    print(f"日志写入 results/train_log_{args.tag}.json")


if __name__ == "__main__":
    main()
