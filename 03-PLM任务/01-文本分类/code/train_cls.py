"""基于预训练语言模型的中文评论八分类（对应 transformers_tasks/text_classification）。

与原仓库 train.py 的差异（见实验报告「遇到的问题」）：
  原仓库 requirements 固定 transformers==4.22.1，与本机 torch 2.11 不兼容，
  因此这里用等价的手写训练循环复刻其核心流程：
      AutoTokenizer + AutoModelForSequenceClassification + AdamW + 线性 warmup
      + 交叉熵 / Focal Loss + 每个类别单独的 P/R/F1

用法（在「03-PLM任务/01-文本分类」目录下执行）：
    python code/train_cls.py --tag bert
    python code/train_cls.py --model hfl/rbt3 --tag rbt3 --epochs 30
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_scheduler)

ROOT = Path(__file__).resolve().parents[1]     # 03-PLM任务/01-文本分类


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:      # noqa: BLE001
                pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CommentDataset(Dataset):
    """把 label\\ttext 的文本行切成模型输入。"""

    def __init__(self, path, tokenizer, max_seq_len, has_label=True):
        self.items = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if has_label and len(parts) >= 2:
                label, text = int(parts[0]), "\t".join(parts[1:])
                self.items.append((text, label))
            else:
                self.items.append((line, -1))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label = self.items[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_seq_len,
                             padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class FocalLoss(nn.Module):
    """Focal Loss，用于缓解类别不平衡（原仓库 FocalLoss.py 的等价实现）。"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, golds, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
        preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
        golds.extend(batch["labels"].cpu().tolist())
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(np.array(preds) == np.array(golds))),
        "precision": float(precision_score(golds, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(golds, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(golds, preds, average="macro", zero_division=0)),
        "per_class_f1": [round(x, 4) for x in
                         f1_score(golds, preds, average=None,
                                  labels=list(range(max(golds) + 1)), zero_division=0)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-chinese")
    parser.add_argument("--train-path", default=str(ROOT / "data/comment_classify/train.txt"))
    parser.add_argument("--dev-path", default=str(ROOT / "data/comment_classify/dev.txt"))
    parser.add_argument("--num-labels", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="bert")
    args = parser.parse_args()

    setup_console()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    print(f"device={device}  backbone={args.model}  loss={args.loss}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=args.num_labels).to(device)

    train_ds = CommentDataset(args.train_path, tokenizer, args.max_seq_len)
    dev_ds = CommentDataset(args.dev_path, tokenizer, args.max_seq_len)
    print(f"train={len(train_ds)}  dev={len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=int(args.warmup_ratio * total_steps),
                              num_training_steps=total_steps)
    criterion = (nn.CrossEntropyLoss() if args.loss == "cross_entropy"
                 else FocalLoss())

    save_dir = ROOT / "ckpt" / args.tag
    history, best_f1, best_epoch = [], -1.0, -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids"))
            loss = criterion(out.logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            steps += 1

        metrics = evaluate(model, dev_loader, device)
        metrics.update({"epoch": epoch, "train_loss": round(running / max(1, steps), 4)})
        history.append(metrics)
        print(f"[epoch {epoch:2d}] train_loss={metrics['train_loss']:.4f} "
              f"dev_acc={metrics['accuracy']:.4f} P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f} macroF1={metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1, best_epoch = metrics["f1"], epoch
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  -> 保存最优模型（macroF1={best_f1:.4f}）至 {save_dir}")

    elapsed = time.time() - t0
    summary = {
        "tag": args.tag, "backbone": args.model, "loss": args.loss,
        "device": str(device), "train_size": len(train_ds), "dev_size": len(dev_ds),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "max_seq_len": args.max_seq_len, "best_epoch": best_epoch,
        "best_macro_f1": round(best_f1, 4),
        "best_metrics": history[best_epoch - 1] if best_epoch > 0 else None,
        "train_seconds": round(elapsed, 1),
        "history": history,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最佳 macroF1 = {best_f1:.4f}（epoch {best_epoch}），耗时 {elapsed:.1f}s")
    print(f"指标写入 {out.relative_to(ROOT)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ep = [h["epoch"] for h in history]
        axes[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train loss")
        axes[0].plot(ep, [h["loss"] for h in history], marker="s", label="dev loss")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
        axes[0].set_title(f"{args.tag}: loss"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(ep, [h["accuracy"] for h in history], marker="o", label="dev acc")
        axes[1].plot(ep, [h["f1"] for h in history], marker="s", label="dev macro F1")
        axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1)
        axes[1].set_title(f"{args.tag}: dev metrics"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout()
        fig_dir = ROOT / "figures"; fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f"train_curve_{args.tag}.png", dpi=200)
        print(f"曲线图写入 figures/train_curve_{args.tag}.png")
    except Exception as e:                       # noqa: BLE001
        print(f"[警告] 画图失败：{e}")


if __name__ == "__main__":
    main()
