# data 目录说明

本目录存放图数据集的缓存，**未提交到 Git**（体积约 900 MB）。首次运行训练脚本会自动下载：

| 子目录 | 数据集 | 来源 | 大小 |
|---|---|---|---|
| `cora/` | Cora（2708 节点 / 10556 边 / 1433 维 / 7 类） | `torch_geometric.datasets.Planetoid`，从 `github.com/kimiyoung/planetoid` 下载 | 约 12 MB |
| `citeseer/` | Citeseer（3327 节点 / 9104 边 / 3703 维 / 6 类） | 同上 | 约 20 MB |
| `flickr/` | Flickr（89250 节点 / 899756 边 / 500 维 / 7 类） | **需手动准备**，见下 | 约 390 MB |

## Flickr 数据的手动准备

PyG 自带的 `Flickr` 数据集通过 Google Drive 取数，国内网络无法访问
（`URLError: [WinError 10060]` 连接超时）。替代方案是从 DGL 官方站点下载同源数据
（GraphSAINT 版本），解压后放入 `flickr/raw/`：

```bash
# 1. 下载（约 25 MB）
curl -L -o flickr.zip https://data.dgl.ai/dataset/flickr.zip

# 2. 解压后应得到这 4 个文件，放入 data/flickr/raw/
#    adj_full.npz  feats.npy  class_map.json  role.json
```

目录结构应为：

```
data/flickr/raw/adj_full.npz
data/flickr/raw/feats.npy
data/flickr/raw/class_map.json
data/flickr/raw/role.json
```

只要这 4 个文件存在，PyG 就会跳过下载直接进入 `process` 阶段。
验证数据集是否就绪：

```bash
python -c "from datasets import load_dataset; ds, d = load_dataset('flickr'); print(d)"
```