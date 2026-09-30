"""节点分类主脚本：GCN / GAT / GraphSAGE / GIN，支持全图训练与子图采样训练。

用法（在 code 目录下执行）：
    python train_node_cls.py --dataset cora --model gcn --mode full
    python train_node_cls.py --dataset cora --model gcn --mode sample
    python train_node_cls.py --run-all
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets import describe, load_dataset          # noqa: E402
from models import build_model                       # noqa: E402

RESULT_DIR = ROOT.parent / "results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels, mask):
    pred = logits.argmax(dim=-1)
    return int((pred[mask] == labels[mask]).sum()), int(mask.sum())


def macro_f1(logits, labels, mask):
    pred = logits.argmax(dim=-1)[mask].cpu().numpy()
    gold = labels[mask].cpu().numpy()
    f1s = []
    for c in np.unique(gold):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(2 * prec * rec / max(1e-12, prec + rec))
    return float(np.mean(f1s))


def train_full(args, data, num_features, num_classes, device):
    model = build_model(args.model, num_features, args.hidden, num_classes,
                        args.layers, args.heads, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    best_val, best_state, best_epoch = -1.0, None, -1
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_correct, val_total = accuracy(logits, data.y, data.val_mask)
            val_acc = val_correct / max(1, val_total)
        history.append({"epoch": epoch, "loss": round(loss.item(), 4),
                        "val_acc": round(val_acc, 4)})
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.verbose:
            print(f"    epoch {epoch:3d} loss={loss.item():.4f} val_acc={val_acc:.4f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_correct, test_total = accuracy(logits, data.y, data.test_mask)
        f1 = macro_f1(logits, data.y, data.test_mask)
    return model, {
        "test_acc": round(test_correct / max(1, test_total), 4),
        "test_macro_f1": round(f1, 4),
        "best_val_acc": round(best_val, 4),
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 2),
        "history": history,
    }


def make_loader(data, args, input_nodes, shuffle):
    from torch_geometric.loader import NeighborLoader
    return NeighborLoader(
        data,
        num_neighbors=args.fan_out,
        batch_size=args.batch_size,
        input_nodes=input_nodes,
        shuffle=shuffle,
        num_workers=0,
    )


@torch.no_grad()
def eval_loader(model, loader, data, mask, device):
    """按批推理并汇总预测，再整体算指标（避免批间指标平均带来的偏差）。"""
    model.eval()
    logits_all = torch.zeros(data.num_nodes, data.y.max().item() + 1, device=device)
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        logits_all[batch.n_id[: batch.batch_size]] = out[: batch.batch_size]
    return accuracy(logits_all, data.y.to(device), mask.to(device)), logits_all


def train_sample(args, data, num_features, num_classes, device):
    model = build_model(args.model, num_features, args.hidden, num_classes,
                        args.layers, args.heads, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    train_loader = make_loader(data, args, data.train_mask, shuffle=True)
    val_loader = make_loader(data, args, data.val_mask, shuffle=False)
    test_loader = make_loader(data, args, data.test_mask, shuffle=False)

    n_train_batches = len(train_loader)
    best_val, best_state, best_epoch = -1.0, None, -1
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # 只用 seed 节点（batch 前 batch_size 个）算损失
            y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(out[: batch.batch_size], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        (val_correct, val_total), _ = eval_loader(model, val_loader, data, data.val_mask, device)
        val_acc = val_correct / max(1, val_total)
        history.append({"epoch": epoch, "loss": round(total_loss / max(1, steps), 4),
                        "val_acc": round(val_acc, 4)})
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.verbose:
            print(f"    epoch {epoch:3d} loss={total_loss/max(1,steps):.4f} "
                  f"val_acc={val_acc:.4f} batches={n_train_batches}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    (test_correct, test_total), logits_all = eval_loader(
        model, test_loader, data, data.test_mask, device)
    f1 = macro_f1(logits_all, data.y.to(device), data.test_mask.to(device))
    return model, {
        "test_acc": round(test_correct / max(1, test_total), 4),
        "test_macro_f1": round(f1, 4),
        "best_val_acc": round(best_val, 4),
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 2),
        "train_batches_per_epoch": n_train_batches,
        "history": history,
    }


def run_one(args, dataset, data, device) -> dict:
    set_seed(args.seed)
    data = data.to(device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    fn = train_full if args.mode == "full" else train_sample
    model, metrics = fn(args, data, dataset.num_features, dataset.num_classes, device)

    peak_mem = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == "cuda" else 0.0)
    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "dataset": args.dataset, "model": args.model, "mode": args.mode,
        "hidden": args.hidden, "layers": args.layers, "heads": args.heads,
        "lr": args.lr, "dropout": args.dropout, "seed": args.seed,
        "batch_size": args.batch_size if args.mode == "sample" else int(data.num_nodes),
        "fan_out": args.fan_out if args.mode == "sample" else None,
        "epochs": args.epochs,
        "params": int(n_params),
        "peak_gpu_mem_mb": round(peak_mem, 1),
        **{k: v for k, v in metrics.items()},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora", choices=["cora", "citeseer", "flickr"])
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "sage", "gin"])
    parser.add_argument("--mode", default="full", choices=["full", "sample"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fan-out", type=int, nargs=2, default=[10, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-all", action="store_true",
                        help="跑「4 模型 x 2 训练模式」共 8 组对比")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # noqa: BLE001
        pass
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    if args.device == "cpu":
        device = torch.device("cpu")

    dataset, data = load_dataset(args.dataset)
    info = describe(dataset, data)
    print("=" * 72)
    print(f"数据集: {info}")
    print(f"设备: {device}")
    print("=" * 72)

    combos = ([(m, mode) for m in ("gcn", "gat", "sage", "gin")
               for mode in ("full", "sample")] if args.run_all
              else [(args.model, args.mode)])

    results = []
    for model_name, mode in combos:
        args.model, args.mode = model_name, mode
        print(f"\n>>> {args.dataset} | {model_name} | {mode} | "
              f"layers={args.layers} hidden={args.hidden} lr={args.lr}")
        res = run_one(args, dataset, data, device)
        results.append(res)
        print(f"    test_acc={res['test_acc']:.4f}  macro_f1={res['test_macro_f1']:.4f}  "
              f"val_acc={res['best_val_acc']:.4f}  time={res['train_time_s']}s  "
              f"peak_mem={res['peak_gpu_mem_mb']}MB")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.dataset}"
    out = RESULT_DIR / f"node_cls_{tag}.json"
    payload = {"info": info, "results": results}
    key = ("dataset", "model", "mode", "lr", "layers", "hidden", "dropout", "epochs")
    if out.exists():   # 追加而不是覆盖；同配置（含超参）才覆盖
        old = json.loads(out.read_text(encoding="utf-8"))
        # 兼容早期版本写出的结果文件（缺少 epochs 等字段）：缺失字段不参与比较
        old["results"] = [r for r in old["results"]
                          if not any(all(r.get(k) == n.get(k) for k in key if k in r and k in n)
                                     for n in results)]
        old["results"].extend(results)
        payload = old
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
