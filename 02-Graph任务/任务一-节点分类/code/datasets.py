"""三类数据集的统一加载器：Cora / Citeseer / Flickr。"""
from pathlib import Path

from torch_geometric.datasets import Flickr, Planetoid

PLANETOID = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}


def load_dataset(name: str, root=None):
    """返回 (dataset, data)，data 为单张图（Planetoid/Flickr 都只含一张图）。"""
    name = name.lower()
    root = Path(root or (Path(__file__).resolve().parents[1] / "data" / name))
    root.mkdir(parents=True, exist_ok=True)

    if name in PLANETOID:
        dataset = Planetoid(root=str(root), name=PLANETOID[name])
    elif name == "flickr":
        dataset = Flickr(root=str(root))
    else:
        raise ValueError(f"暂不支持的数据集：{name}")

    data = dataset[0]
    if not hasattr(data, "train_mask") or data.train_mask is None:
        raise RuntimeError(f"{name} 缺少 train_mask")
    # Flickr 的 mask 是 (N, 1)，压成 (N,)
    for split in ("train_mask", "val_mask", "test_mask"):
        m = getattr(data, split)
        if m.dim() == 2:
            setattr(data, split, m[:, 0])
    return dataset, data


def describe(dataset, data) -> dict:
    return {
        "dataset": dataset.name if hasattr(dataset, "name") else "flickr",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
    }
