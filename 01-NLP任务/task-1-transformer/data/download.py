"""下载 ChnSentiCorp 中文情感分类数据集到当前目录，产出
    data/train.parquet  data/validation.parquet  data/test.parquet

注意（踩坑记录）：
  1. 官方脚本数据集 seamew/ChnSentiCorp 的原始数据托管在 Google Drive，
     国内网络无法访问，会 ConnectTimeout。
  2. datasets>=4.0 已移除「脚本型数据集」支持，加载 seamew 版本会直接报
     RuntimeError: Dataset scripts are no longer supported。
  所以这里改为直接拉取社区维护的 parquet 版本（lansinuote/ChnSentiCorp），
  再用 pandas 统一列名为 text / label。

用法：
    python data/download.py
    # 国内建议先设镜像（脚本内已默认设置）：
    # Windows PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent

# 社区维护的 parquet 版本，按优先级尝试
CANDIDATES = ["lansinuote/ChnSentiCorp", "ndiy/ChnSentiCorp", "Kerwin11/ChnSentiCorp"]
SPLITS = ["train", "validation", "test"]


def _set_mirror():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _download_from_hub(repo_id: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    mapping = {}
    for split in SPLITS:
        hits = [f for f in files
                if f.endswith(".parquet") and Path(f).name.startswith(split)]
        if not hits:
            raise RuntimeError(f"{repo_id} 中没有找到 {split} 分片")
        mapping[split] = hf_hub_download(repo_id, hits[0], repo_type="dataset")
    return mapping


def _normalize(path_or_repo, split: str) -> pd.DataFrame:
    """读进来并统一成 text / label 两列。"""
    if isinstance(path_or_repo, str) and path_or_repo.endswith(".parquet"):
        df = pd.read_parquet(path_or_repo)
    else:
        from datasets import load_dataset
        ds = load_dataset(path_or_repo, split=split)
        df = ds.to_pandas()

    rename = {}
    for col in df.columns:
        low = col.lower()
        if low in ("text", "sentence", "review", "content", "comment"):
            rename[col] = "text"
        elif low in ("label", "labels", "target", "sentiment"):
            rename[col] = "label"
    df = df.rename(columns=rename)
    if "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"无法识别列名：{list(df.columns)}")
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df


def main():
    _set_mirror()
    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")

    raw = None
    last_err = None
    for repo in CANDIDATES:
        try:
            print(f"尝试下载 {repo} ...")
            raw = _download_from_hub(repo)
            print(f"  [OK] {repo}")
            break
        except Exception as e:                       # noqa: BLE001
            print(f"  [失败] {repo}: {type(e).__name__}: {e}")
            last_err = e
    if raw is None:
        sys.exit(f"所有镜像都失败，最后一个错误：{last_err}")

    for split in SPLITS:
        df = _normalize(raw[split], split)
        out = DATA_DIR / f"{split}.parquet"
        df.to_parquet(out, index=False)
        print(f"  {split}: {len(df)} 条  标签分布={df['label'].value_counts().to_dict()}"
              f"  -> {out.name}")

    print(f"\n完成，数据保存在 {DATA_DIR}")


if __name__ == "__main__":
    main()
