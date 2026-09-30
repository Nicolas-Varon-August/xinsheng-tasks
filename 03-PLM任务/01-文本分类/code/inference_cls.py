"""加载微调后的模型做单句 / 批量预测。

用法（在「03-PLM任务/01-文本分类」目录下执行）：
    python code/inference_cls.py --text "苹果很甜，个头也大"
    python code/inference_cls.py --file data/comment_classify/dev.txt --limit 10
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = {}
for line in (ROOT / "data/comment_classify/label_mapping.txt").read_text(
        encoding="utf-8").splitlines():
    if line.strip():
        idx, name = line.split(",", 1)
        LABELS[int(idx)] = name.strip()


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:      # noqa: BLE001
                pass


@torch.no_grad()
def predict(model, tokenizer, texts, device, max_seq_len=128):
    enc = tokenizer(list(texts), truncation=True, max_length=max_seq_len,
                    padding=True, return_tensors="pt").to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1).cpu().tolist(), probs.max(dim=-1).values.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(ROOT / "ckpt" / "rbt3"))
    parser.add_argument("--text", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_console()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).to(device)
    model.eval()
    print(f"加载模型：{args.ckpt}  device={device}\n")

    if args.text:
        texts = [args.text]
    elif args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()[: args.limit]
        texts = [ln.split("\t")[-1] for ln in lines if ln.strip()]
    else:
        texts = ["苹果很甜，个头也大", "快递太慢了，客服态度还很差", "这本书内容一般，纸张也薄"]

    preds, confs = predict(model, tokenizer, texts, device)
    for text, p, c in zip(texts, preds, confs):
        print(f"[{LABELS.get(p, p)}] (p={c:.3f})  {text[:60]}")


if __name__ == "__main__":
    main()
