# 任务一：节点分类（GCN / GAT / GraphSAGE / GIN）

对应 [graph_beginner](https://github.com/Maxioo/graph_beginner) 任务一。基于 **PyTorch Geometric**，
在 Cora / Citeseer / Flickr 三个数据集上对比四种主流 GNN，并比较**全图训练**与
**子图采样训练（NeighborLoader）**的性能与耗时。

## 目录结构

```
任务一-节点分类/
├── code/
│   ├── datasets.py          # 三个数据集的统一加载与统计
│   ├── models.py            # GCN/GAT/GraphSAGE/GIN 四合一模型
│   └── train_node_cls.py    # 训练 + 评测主脚本（含 --run-all 批量对比）
├── data/                    # 数据集缓存（首次运行自动下载）
│   ├── cora/  citeseer/     # Planetoid
│   ├── flickr/raw/          # 已手动放置 adj_full.npz / feats.npy / class_map.json / role.json
│   └── _downloads/          # 备用下载缓存
├── results/                 # 实验指标 JSON（node_cls_<dataset>.json）
└── README.md
```

## 环境

```
python 3.10 / torch 2.11.0+cu128 / torch_geometric 2.8.0 / pyg-lib 0.9.0
numpy 2.2.6 / scikit-learn 1.7.2
GPU: NVIDIA RTX 4070 Laptop (8GB)
```

安装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric scikit-learn numpy
pip install -f https://data.pyg.org/whl/torch-2.11.0+cu128.html pyg-lib
```

## 训练与测试脚本

在 `code/` 目录下执行：

```bash
# 单组实验（默认全图训练）
python train_node_cls.py --dataset cora --model gcn --mode full

# 单组实验（子图采样训练，fan-out = [10, 10]）
python train_node_cls.py --dataset cora --model gcn --mode sample

# 一键跑完「4 模型 x 2 训练模式」并写入 results/
python train_node_cls.py --dataset cora     --run-all --epochs 100 --tag cora
python train_node_cls.py --dataset citeseer --run-all --epochs 100 --tag citeseer
python train_node_cls.py --dataset flickr   --run-all --epochs 100 --tag flickr

# 常用可调参数
python train_node_cls.py --dataset cora --model gat --mode full \
    --hidden 128 --layers 3 --heads 8 --lr 0.005 --dropout 0.6 --epochs 200
```

主要参数：`--hidden`（隐藏维）、`--layers`（层数）、`--heads`（GAT 头数）、
`--lr`、`--dropout`、`--batch-size`（采样模式）、`--fan-out`（每层采样邻居数）。

## 数据集统计

| 数据集 | 节点数 | 边数 | 特征维 | 类别数 | 训练/验证/测试节点 |
|---|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 / 500 / 1000 |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 | 120 / 500 / 1000 |
| Flickr | 89,250 | 899,756 | 500 | 7 | 44,625 / 22,312 / 22,313 |

## 实验结果（100 epochs，seed=42）

### Cora

| 模型 | 训练方式 | test acc | macro F1 | 训练耗时(s) | 峰值显存(MB) |
|---|---|---|---|---|---|
| GCN | 全图 | 0.7930 | 0.7806 | 0.64 | 41.7 |
| GCN | 采样 | **0.8120** | **0.8037** | 2.40 | 43.9 |
| GAT | 全图 | **0.8210** | **0.8175** | 0.64 | 93.6 |
| GAT | 采样 | 0.7890 | 0.7746 | 3.21 | 52.9 |
| GraphSAGE | 全图 | 0.7910 | 0.7830 | 0.45 | 123.5 |
| GraphSAGE | 采样 | 0.7870 | 0.7776 | 2.12 | 59.8 |
| GIN | 全图 | 0.7940 | 0.7831 | 0.50 | 107.0 |
| GIN | 采样 | 0.7880 | 0.7837 | 2.11 | 54.9 |

### Citeseer

| 模型 | 训练方式 | test acc | macro F1 | 训练耗时(s) | 峰值显存(MB) |
|---|---|---|---|---|---|
| GCN | 全图 | 0.6800 | 0.6406 | 0.81 | 76.7 |
| GCN | 采样 | 0.6790 | 0.6423 | 2.39 | 94.0 |
| GAT | 全图 | **0.6890** | **0.6536** | 0.76 | 131.1 |
| GAT | 采样 | 0.6720 | 0.6407 | 2.94 | 107.5 |
| GraphSAGE | 全图 | 0.6750 | 0.6371 | 0.86 | 296.2 |
| GraphSAGE | 采样 | 0.6620 | 0.6180 | 1.75 | 121.7 |
| GIN | 全图 | 0.6330 | 0.6098 | 0.99 | 244.7 |
| GIN | 采样 | 0.6660 | 0.6291 | 1.82 | 112.6 |

### Flickr

| 模型 | 训练方式 | test acc | macro F1 | 训练耗时(s) | 峰值显存(MB) |
|---|---|---|---|---|---|
| GCN | 全图 | 0.4893 | 0.1919 | 3.6 | 751.6 |
| GCN | 采样 | 0.5004 | 0.1747 | 385.3 | 236.4 |
| GAT | 全图 | 0.4941 | 0.1619 | 14.3 | 4247.5 |
| GAT | 采样 | 0.4926 | 0.1618 | 478.1 | 252.0 |
| GraphSAGE | 全图 | 0.4583 | 0.1339 | 7.9 | 2263.0 |
| GraphSAGE | 采样 | 0.4935 | 0.1628 | 287.4 | 243.7 |
| GIN | 全图 | 0.4234 | 0.0850 | 8.5 | 2091.9 |
| GIN | 采样 | **0.5065** | **0.1684** | 308.6 | 241.7 |

## 观察与结论

1. **模型对比**：Cora 上 GAT 最优（0.8210），Citeseer 上同样是 GAT 最好（0.6890）。
   GAT 的注意力权重让中心节点能对不同邻居分配不同权重，在同配图（homophily）上更有效；
   GIN 在小图上受限于 MLP 参数量与较弱的归一化，表现偏弱。
2. **采样 vs 全图**：
   - 小图（Cora/Citeseer）上**全图训练更快**（0.5–1s vs 2–4s），因为采样器本身的
     开销远大于一次前向传播，且小图能整张放进显存。
   - 大图（Flickr）上采样的优势体现在**显存**：GAT 全图峰值 4247MB，采样仅 252MB（约 1/17），
     而精度基本持平。这正是 GraphSAINT/邻居采样类方法的动机：用显存换时间。
3. **层数/参数量**：GAT 参数量 369K，是 GCN（92K）的 4 倍，耗时却相近，说明小图上算力不是瓶颈。
4. **Flickr 精度偏低**：Flickr 是典型的**低同配**图，邻居标签与中心节点标签相关性弱，
   公开工作（GraphSAINT 等）报告的正确率同样在 0.50 左右，因此 0.42–0.51 属于正常范围；
   macro F1 只有 0.13–0.19 则反映**类别极不平衡**，模型倾向于预测多数类。
5. **调参影响**（GCN，Cora，详细见实验报告）：
   - 学习率 0.001 / 0.005 / 0.01 / 0.05 对应 test acc 0.7620 / 0.7770 / **0.7930** / 0.7880，
     0.01 左右最优；学习率过小（0.001）欠拟合，过大（0.05）开始震荡。
   - 层数 2→4（hidden=64）为 0.7930 → 0.7940 → **0.8050**，加深略有收益；
     但层数 4 + hidden=32 掉到 0.7610，说明**过深的网络在本文这种小图 + 小 hidden 下会过平滑**。
   - 隐藏维 32→128（2 层）为 0.7860 → 0.7930 → 0.7800，**并非越大越好**，
     显存从 36.9MB 涨到 50.7MB 而准确率反而略降。
