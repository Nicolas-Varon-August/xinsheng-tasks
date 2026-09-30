"""画出注意力热图（DoD 的 M5）。

用法（在 task-1-transformer 目录下执行）：
    python viz_attention.py
    python viz_attention.py --layer -1 --head 0
输出：figures/attn_<name>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval          # noqa: E402
from src.tokenizer import CLS_TOKEN, PAD_ID   # noqa: E402
from src.console import setup_console         # noqa: E402

# 中文字体，避免热图刻度显示成方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SAMPLES = [
    ("positive", "这家酒店位置很好，房间干净整洁，服务态度也不错，下次还会再来。"),
    ("negative", "非常失望的一次入住体验，房间有异味，隔音差，前台态度也很冷漠。"),
    ("long", "第一次来这座城市出差，酒店离地铁站很近出行方便，"
             "早餐种类还算丰富，但是房间里的空调噪音有点大，晚上睡得不太好，"
             "总体感觉一般，性价比不算高。"),
    ("positive2", "故事情节很吸引人，演员演技在线，值得一看。"),
]


def plot_one(ids, weights, tokens, title, out_path, max_tokens=40):
    """画单个头的 (T, T) 注意力矩阵。"""
    w = weights[0].cpu().numpy()          # (T, T)
    n = min(len(tokens), max_tokens)
    w = w[:n, :n]
    labels = tokens[:n]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.32), max(5, n * 0.30)))
    im = ax.imshow(w, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key（被注意的位置）")
    ax.set_ylabel("Query（发起注意的位置）")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def head_entropy(weight_tensor) -> float:
    """一个头的平均注意力熵（越小说明越「聚焦」，越适合拿来做可视化）。"""
    p = weight_tensor.clamp_min(1e-9)
    ent = -(p * p.log()).sum(dim=-1).mean().item()
    return ent


def scan_heads(model, tokenize_fn, names):
    """遍历 (层, 头)，返回按平均熵升序排列的候选列表。"""
    table = {}
    for name, text in names:
        ids = tokenize_fn(text).unsqueeze(0)
        with torch.no_grad():
            _, weights = model(ids, return_attn=True)
        for layer_idx, w in enumerate(weights):
            for head_idx in range(w.size(1)):
                table.setdefault((layer_idx, head_idx), []).append(
                    head_entropy(w[0, head_idx]))
    ranked = sorted(table.items(), key=lambda kv: float(np.mean(kv[1])))
    return [(k, float(np.mean(v))) for k, v in ranked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(ROOT / "ckpt" / "best.pt"))
    parser.add_argument("--layer", type=int, default=-1, help="第几层（支持负数）")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--auto-head", action="store_true",
                        help="自动挑选平均注意力熵最小的 (层, 头)，热图更可读")
    parser.add_argument("--scan", action="store_true", help="只打印各头熵值排名")
    parser.add_argument("--out-dir", default=str(ROOT / "figures"))
    args = parser.parse_args()

    setup_console()
    if not Path(args.ckpt).exists():
        sys.exit(f"找不到 {args.ckpt}，请先运行 python train.py")

    model, tokenize_fn = load_for_eval(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"使用 checkpoint: {args.ckpt}")

    if args.auto_head or args.scan:
        ranked = scan_heads(model, tokenize_fn, SAMPLES)
        print("\n各 (层, 头) 平均注意力熵排名（越小越聚焦）：")
        for (layer_idx, head_idx), ent in ranked[:8]:
            print(f"    layer={layer_idx} head={head_idx}  mean_entropy={ent:.3f}")
        if args.scan:
            return
        args.layer, args.head = ranked[0][0]
        print(f"自动选择 layer={args.layer} head={args.head}（熵最小）\n")

    print(f"可视化 layer={args.layer} head={args.head}\n")

    for name, text in SAMPLES:
        ids = tokenize_fn(text).unsqueeze(0)                  # (1, T)
        with torch.no_grad():
            logits, weights = model(ids, return_attn=True)
        pred = int(logits.argmax(dim=-1).item())
        conf = torch.softmax(logits, dim=-1).max().item()
        layer_w = weights[args.layer][0, args.head]           # (T, T)

        tokens = [CLS_TOKEN] + list(text)
        out_path = out_dir / f"attn_{name}.png"
        title = (f"{name} | 预测={'正面' if pred == 1 else '负面'}"
                 f"(p={conf:.3f}) | layer={args.layer} head={args.head}")
        plot_one(ids, layer_w.unsqueeze(0), tokens, title, out_path)
        print(f"[{name}] 预测={'正面' if pred == 1 else '负面'} 置信度={conf:.3f} "
              f"-> {out_path.name}")

        # 顺带打印每个 query 最关注的 key，便于在报告里做定性分析
        top = layer_w.max(dim=-1)
        for i in range(1, min(len(tokens), 12)):
            j = int(top.indices[i].item())
            print(f"    “{tokens[i]}” 最关注 -> “{tokens[j]}” ({top.values[i].item():.3f})")


if __name__ == "__main__":
    main()
