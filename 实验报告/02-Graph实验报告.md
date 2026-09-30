# 实验报告二：GNN 节点分类（全图训练 vs 子图采样训练）

> 对应任务：Graph 任务 —— <https://github.com/Maxioo/graph_beginner> 任务一「节点分类」
> 代码位置：`02-Graph任务/任务一-节点分类/`

---

## 一、实验环境与条件

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.10.20（conda 环境 `xinsheng`） |
| 深度学习框架 | PyTorch 2.11.0+cu128 |
| 图神经网络库 | PyTorch Geometric 2.8.0 + pyg-lib 0.9.0+pt211cu128 |
| 其他依赖 | numpy 2.2.6、scikit-learn 1.7.2 |
| GPU | NVIDIA GeForce RTX 4070 Laptop，8 GB 显存 |
| 数据集 | Cora、Citeseer（Planetoid）、Flickr（GraphSAINT） |

数据集规模：

| 数据集 | 节点数 | 边数 | 特征维 | 类别数 | 训练/验证/测试 |
|---|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 / 500 / 1000 |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 | 120 / 500 / 1000 |
| Flickr | 89,250 | 899,756 | 500 | 7 | 44,625 / 22,312 / 22,313 |

安装：

```bash
pip install torch_geometric scikit-learn numpy
pip install -f https://data.pyg.org/whl/torch-2.11.0+cu128.html pyg-lib
```

## 二、实验方法与过程

### 2.1 实现内容

| 文件 | 内容 |
|---|---|
| `code/datasets.py` | 统一加载 Cora / Citeseer / Flickr，把 Flickr 的 `(N,1)` mask 压成 `(N,)`，输出数据集统计 |
| `code/models.py` | 四合一 `GNN` 模型，按名字选用 `GCNConv` / `GATConv` / `SAGEConv` / `GINConv`，每层后接 `LayerNorm`，层间 ReLU + Dropout |
| `code/train_node_cls.py` | 训练 + 评测主脚本；`--run-all` 一键跑「4 模型 × 2 训练模式」 |

### 2.2 两种训练方式

1. **全图训练（full）**：整张图的 `x`、`edge_index` 一次性送进模型，只在训练节点掩码上算损失。
2. **子图采样训练（sample）**：用框架自带的 `torch_geometric.loader.NeighborLoader`，
   `batch_size=64`、两层 fan-out 均为 10（即每个种子节点每层采样 10 个邻居，共最多 1+10+100 个节点），
   只用每个 batch 的**种子节点**算损失。评测时按同样的 loader 分批推理，再把所有节点的
   logits 汇总后统一算指标（避免批间指标平均带来的偏差）。

### 2.3 操作步骤（可复现）

```bash
cd 02-Graph任务/任务一-节点分类/code
python train_node_cls.py --dataset cora     --run-all --epochs 100 --tag cora
python train_node_cls.py --dataset citeseer --run-all --epochs 100 --tag citeseer
python train_node_cls.py --dataset flickr   --run-all --epochs 100 --tag flickr
# 参数消融
python train_node_cls.py --dataset cora --model gcn --mode full --lr 0.001 --tag ablation
python train_node_cls.py --dataset cora --model gcn --mode full --layers 4 --hidden 64 --tag ablation
```

统一超参：`hidden=64, layers=2, heads=4(GAT), dropout=0.5, lr=0.01, weight_decay=5e-4, epochs=100, seed=42`。

## 三、实验结果及分析

### 3.1 Cora

| 模型 | 训练方式 | test acc | macro F1 | 训练耗时(s) | 峰值显存(MB) |
|---|---|---|---|---|---|
| GCN | 全图 | 0.7930 | 0.7806 | **0.64** | 41.7 |
| GCN | 采样 | **0.8120** | **0.8037** | 2.40 | 43.9 |
| GAT | 全图 | **0.8210** | **0.8175** | **0.64** | 93.6 |
| GAT | 采样 | 0.7890 | 0.7746 | 3.21 | 52.9 |
| GraphSAGE | 全图 | 0.7910 | 0.7830 | **0.45** | 123.5 |
| GraphSAGE | 采样 | 0.7870 | 0.7776 | 2.12 | 59.8 |
| GIN | 全图 | 0.7940 | 0.7831 | **0.50** | 107.0 |
| GIN | 采样 | 0.7880 | 0.7837 | 2.11 | 54.9 |

### 3.2 Citeseer

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

### 3.3 Flickr

| 模型 | 训练方式 | test acc | macro F1 | 训练耗时(s) | 峰值显存(MB) |
|---|---|---|---|---|---|
| GCN | 全图 | 0.4893 | 0.1919 | **3.6** | 751.6 |
| GCN | 采样 | 0.5004 | 0.1747 | 385.3 | 236.4 |
| GAT | 全图 | 0.4941 | 0.1619 | 14.3 | **4247.5** |
| GAT | 采样 | 0.4926 | 0.1618 | 478.1 | **252.0** |
| GraphSAGE | 全图 | 0.4583 | 0.1339 | 7.9 | 2263.0 |
| GraphSAGE | 采样 | 0.4935 | 0.1628 | 287.4 | 243.7 |
| GIN | 全图 | 0.4234 | 0.0850 | 8.5 | 2091.9 |
| GIN | 采样 | **0.5065** | **0.1684** | 308.6 | 241.7 |

### 3.4 学习率与网络层数（GCN / Cora / 全图）

| 学习率 | test acc | macro F1 |
|---|---|---|
| 0.001 | 0.7620 | 0.7506 |
| 0.005 | 0.7770 | 0.7691 |
| **0.01** | **0.7930** | **0.7806** |
| 0.05 | 0.7880 | 0.7768 |

| 层数 \ 隐藏维 | 32 | 64 | 128 |
|---|---|---|---|
| 2 | 0.7860 | 0.7930 | 0.7800 |
| 3 | 0.8060 | 0.7940 | 0.7990 |
| 4 | 0.7610 | **0.8050** | 0.7820 |

### 3.5 结果分析

1. **不同神经网络的影响**：Cora 与 Citeseer 上 **GAT 都是最优**（0.8210 / 0.6890）。
   GAT 通过注意力为不同邻居分配不同权重，在图书/论文这类**同配图**上能够压低噪声邻居的影响。
   GIN 在 Citeseer 上最差（0.6330）：GIN 的 MLP 逐层堆叠参数量大，而 Citeseer 只有 120 个训练节点，
   容易过拟合。GraphSAGE 表现居中且稳定。
2. **全图 vs 采样的性能**：
   - 精度上两者互有胜负，最大差距在 Cora 的 GAT（0.8210 vs 0.7890，约 3 个百分点）。
     采样引入了邻居覆盖不全带来的方差，小图上这种损失更明显。
   - 在 Flickr 上采样反而略微更好（GIN：0.4234 → 0.5065），因为全图训练在低同配图上
     容易被海量噪声邻居"淹没"，采样相当于一种正则化。
3. **全图 vs 采样的运行时间**：
   - **小图（Cora/Citeseer）上全图训练明显更快**（0.5–1 s vs 1.8–3.8 s，慢 3–5 倍），
     因为采样器本身（邻居采样 + 子图重建）的 CPU 开销远超一次小规模前向传播。
   - **大图（Flickr）上采样慢 10–100 倍**（GCN 3.6 s → 385 s），这是 CPU 采样成为瓶颈的典型表现：
     每个 epoch 要构造 697 个 batch，每个 batch 都要在 CPU 上做两层邻居采样。
   - 但采样换来的是**显存的大幅下降**：GAT 全图峰值 4247.5 MB → 采样 252.0 MB（约 1/17）。
     在单卡 8 GB 上，全图 GAT 已经逼近上限，采样则余量充足。这正是邻居采样方法的设计动机：
     **用时间换显存**，使得超大图（数十亿边）可以在单卡上训练。
4. **学习率影响**：0.001 欠拟合（0.7620），0.01 最优（0.7930），0.05 开始震荡（0.7880）。
Cora 这种小图必须配较小的学习率，否则梯度噪声会导致验证集准确率剧烈波动。
5. **层数与隐藏维影响**：
   - 层数 2→4（hidden=64）从 0.7930 提升到 0.8050，加深有一定收益；
   - 但 **层数 4 + hidden=32 反而掉到 0.7610**，说明在网络变深时如果每层容量太小，
     多层邻域平均会把节点表示"过平滑"（over-smoothing），所有节点趋于相同；
   - 隐藏维 32→64 有提升，64→128 反而略降（0.7930 → 0.7800），参数量增加带来过拟合，
     小图上**没必要一味加宽**。
6. **Flickr 精度偏低的原因**：Flickr 是典型的**低同配图**，邻居与中心节点的标签相关性弱，
   公开工作（GraphSAINT 等）报告的正确率同样在 0.50 附近，因此 0.42–0.51 属于正常水平；
   macro F1 只有 0.09–0.19 则反映**类别严重不平衡**，模型倾向预测多数类。

## 四、遇到的问题及处理情况

| # | 问题 | 现象 | 处理 |
|---|---|---|---|
| 1 | `NeighborLoader` 无法使用 | `ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'` | PyG 的邻居采样依赖 C++ 扩展，从官方 wheel 源安装适配本机 torch 的预编译包：`pip install -f https://data.pyg.org/whl/torch-2.11.0+cu128.html pyg-lib` |
| 2 | Citeseer 下载 502 | `raw.githubusercontent.com` 返回 502 Bad Gateway | 属于瞬时故障，重跑一次即成功 |
| 3 | **Flickr 无法下载** | PyG 的 `Flickr` 数据集通过 `download_google_url` 从 Google Drive 取数，国内网络 `URLError: [WinError 10060]` 连接超时 | 换源：从 DGL 官方站点下载 `https://data.dgl.ai/dataset/flickr.zip`（与 GraphSAINT 同源，含 `adj_full.npz`/`feats.npy`/`class_map.json`/`role.json`），手动放入 `data/flickr/raw/`，PyG 检测到原始文件存在即跳过下载 |
| 4 | 沙箱下 `torch_geometric` 导入失败 | `ModuleNotFoundError: No module named 'jinja2'` —— 该包只装在用户级 site-packages，受限环境下不可见 | 显式安装到实验环境：`pip install --ignore-installed jinja2 MarkupSafe` |
| 5 | 消融实验互相覆盖 | 结果按 (dataset, model, mode) 去重，不同超参的组会被覆盖 | 把 `lr/layers/hidden/dropout` 加入去重键 |

## 五、运行截图清单（提交时附上）

1. `python train_node_cls.py --dataset cora --run-all --epochs 100 --tag cora`
   → 8 组结果打印 + `结果写入 ...node_cls_cora.json`（建议截两屏）
2. `python train_node_cls.py --dataset citeseer --run-all --epochs 100 --tag citeseer`
3. `python train_node_cls.py --dataset flickr --run-all --epochs 100 --tag flickr`
4. 学习率 / 层数消融的两条命令输出
5. `results/node_cls_cora.json` 在 PyCharm 中打开的截图

## 六、结论

1. 基于 PyG 实现了 GCN / GAT / GraphSAGE / GIN 四种模型的节点分类，并提供了统一的多数据集对比脚本。
2. 在 Cora（0.8210）与 Citeseer（0.6890）上 **GAT 表现最好**；在低同配的 Flickr 上各模型差距缩小到 0.42–0.51。
3. 全图训练在**小图上更快**，子图采样在**大图上显著省显存**（GAT 4247.5 MB → 252.0 MB），
   但在 CPU 采样上付出了 10–100 倍的训练时间，说明采样方案需要配合 CPU 并行/GPU 采样才能真正发挥作用。
4. 学习率 0.01 左右最优；加深网络有收益，但过深 + 过窄会触发过平滑（0.7610）。
