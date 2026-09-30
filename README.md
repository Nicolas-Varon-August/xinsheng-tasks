# 新生任务实验代码与报告

本仓库为新生任务的完整提交材料，包含 3 个入门实验的可运行代码、运行截图与实验报告。

| 编号 | 任务 | 交付内容 | 状态 |
|---|---|---|---|
| 01 | NLP 任务（[nlp-beginner](https://github.com/FudanNLP/nlp-beginner)） | 从零手写 Transformer + ChnSentiCorp 情感分类，自检 3/3 通过，dev 准确率 **0.8417** | 完成 |
| 02 | Graph 任务（[graph_beginner](https://github.com/Maxioo/graph_beginner)） | GCN/GAT/GraphSAGE/GIN × 全图/采样 × Cora/Citeseer/Flickr，共 24 组对比 + 参数消融 | 完成 |
| 03 | PLM 任务（[transformers_tasks](https://github.com/HarderThenHarder/transformers_tasks)） | 中文评论八分类微调，bert-base-chinese **macro F1 0.9397** | 完成 |

实验报告统一放在 `实验报告/`，均包含「实验环境与条件、实验方法与过程、实验结果及分析、
遇到的问题及处理情况」四部分。此外还有一份 **Word 版实验报告**
`实验报告/新生任务实验报告.docx`，按学院模板排版（封面 + 六个章节 + 12 张运行截图）。
实际运行截图见 `screenshots/`。

---

## 一、环境准备（一次性）

所有实验共用同一个 conda 环境 `xinsheng`（Python 3.10.20 + PyTorch 2.11.0+cu128）：

```powershell
conda activate xinsheng

# 若环境不存在，可由已有 CUDA 环境克隆（避免重复下载数 GB 的 torch）：
# conda create -n xinsheng --clone lnp_gpu

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
    datasets==2.19.0 pyarrow scikit-learn mpmath==1.3.0 jinja2 MarkupSafe matplotlib
pip install torch_geometric
pip install -f https://data.pyg.org/whl/torch-2.11.0+cu128.html pyg-lib
```

国内访问 HuggingFace 不稳定时，先设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"     # hf-mirror 不支持 xet 协议
```

## 二、各任务的运行指令

> 下面所有命令都给出了 **工作目录**，请在 PyCharm 的 Terminal 或 Run 配置中先切到对应目录。

### 任务一：NLP（从零实现 Transformer）

工作目录：`01-NLP任务/task-1-transformer`

```powershell
python data/download.py                 # 下载 ChnSentiCorp（约 9600/1200/1200 条）
python train.py                         # 训练，约 50 s，产出 ckpt/best.pt
python eval/run.py                      # 自检 3 项，产出 eval/result.json
python viz_attention.py --auto-head     # 注意力热图 -> figures/*.png
python toy_lm.py                        # 因果掩码自检 + 唐诗 toy 语言模型
```

预期输出（关键行）：

```text
[epoch 6] train_loss=0.3526 train_acc=0.8556 dev_loss=0.4210 dev_acc=0.8417
最佳 dev 准确率 = 0.8417（epoch 6），总耗时 48.0s

[通过] attention_correctness: {"pass": true, "max_abs_diff": 9.5367431640625e-07}
[通过] causal_mask:           {"pass": true, "leaked_diff": 0.0}
[通过] classifier_accuracy:   {"pass": true, "accuracy": 0.8417, "baseline_reference": 0.85}
```

### 任务二：Graph（节点分类）

工作目录：`02-Graph任务/任务一-节点分类/code`

```powershell
# 单组实验
python train_node_cls.py --dataset cora --model gcn --mode full

# 一键跑完「4 模型 x 2 训练模式」
python train_node_cls.py --dataset cora     --run-all --epochs 100 --tag cora
python train_node_cls.py --dataset citeseer --run-all --epochs 100 --tag citeseer
python train_node_cls.py --dataset flickr   --run-all --epochs 100 --tag flickr

# 参数消融（学习率 / 层数 / 隐藏维）
python train_node_cls.py --dataset cora --model gcn --mode full --lr 0.001 --tag ablation
python train_node_cls.py --dataset cora --model gcn --mode full --layers 4 --hidden 64 --tag ablation
```

预期输出（节选）：

```text
>>> cora | gat | full | layers=2 hidden=64 lr=0.01
    test_acc=0.8210  macro_f1=0.8175  val_acc=0.7920  time=0.67s  peak_mem=93.6MB
```

> Flickr 的数据需要手动准备一次，见 `02-Graph任务/任务一-节点分类/data/README.md`。

### 任务三：PLM（预训练模型微调）

工作目录：`03-PLM任务/01-文本分类`

```powershell
python code/train_cls.py --model bert-base-chinese --tag bert      # 约 80 s
python code/train_cls.py --model hfl/rbt3         --tag rbt3      # 约 26 s
python code/train_cls.py --model hfl/rbt3 --loss focal --tag rbt3_focal
python code/inference_cls.py
```

预期输出（关键行）：

```text
[epoch  5] train_loss=0.1324 dev_acc=0.9206 P=0.9510 R=0.9327 macroF1=0.9397
最佳 macroF1 = 0.9397（epoch 5），耗时 81.8s

[水果] (p=0.999)  苹果很甜，个头也大
[书籍] (p=0.992)  这本书内容一般，纸张也薄
```

---

## 三、PyCharm 运行与截图指引

### 3.1 配置解释器（只需一次）

1. `File` → `Settings` → `Project: 新生任务` → `Python Interpreter`
2. 右上角齿轮 → `Add Interpreter` → `Add Local Interpreter` → `Conda Environment`
3. 选择 `Use existing environment`，在列表里选 `xinsheng`
4. 确认解释器路径类似 `E:\env\mea\xinsheng\python.exe`

> 若列表中没有 `xinsheng`，选 `Conda` 类型并手动指定 `E:\env\mea\xinsheng\python.exe`。

### 3.2 为每个任务建一个运行配置

以任务一为例：

1. `Run` → `Edit Configurations...` → `+` → `Python`
2. `Name` 填 `01-NLP-自检`
3. `Script path` 选 `01-NLP任务/task-1-transformer/eval/run.py`
4. **`Working directory` 一定要设为 `01-NLP任务/task-1-transformer`**（脚本靠相对路径找数据）
5. `Apply` → `OK`，然后点绿色三角运行

其余任务同理，工作目录分别为：

| 运行配置 | Script path | Working directory |
|---|---|---|
| 01-NLP 训练 | `train.py` | `01-NLP任务/task-1-transformer` |
| 01-NLP 自检 | `eval/run.py` | `01-NLP任务/task-1-transformer` |
| 01-NLP 热图 | `viz_attention.py` | `01-NLP任务/task-1-transformer` |
| 01-NLP toy LM | `toy_lm.py` | `01-NLP任务/task-1-transformer` |
| 02-Graph 对比 | `code/train_node_cls.py` | `02-Graph任务/任务一-节点分类/code` |
| 03-PLM 训练 | `code/train_cls.py` | `03-PLM任务/01-文本分类` |
| 03-PLM 推理 | `code/inference_cls.py` | `03-PLM任务/01-文本分类` |

**Parameters** 一栏填对应参数，例如任务二填：

```text
--dataset cora --run-all --epochs 100 --tag cora
```

### 3.3 运行截图清单

| # | 截图内容 | 来源 |
|---|---|---|
| 1 | PyCharm 解释器配置页（conda 环境 `xinsheng`） | Settings → Python Interpreter |
| 2 | 数据下载成功（`train: 9600 条`） | 运行 `data/download.py` |
| 3 | NLP 训练过程 + `最佳 dev 准确率 = 0.8417` | 运行 `train.py` |
| 4 | NLP 自检三个 `[通过]` | 运行 `eval/run.py` |
| 5 | 注意力热图脚本运行输出 | 运行 `viz_attention.py --auto-head` |
| 6 | 注意力热图 `attn_negative.png` | 打开 `figures/attn_negative.png` |
| 7 | toy LM 困惑度与生成样例 | 运行 `toy_lm.py` |
| 8 | Graph 8 组对比结果 | 运行 `train_node_cls.py --run-all` |
| 9 | Graph 结果 JSON | 打开 `results/node_cls_cora.json` |
| 10 | PLM 训练 + `最佳 macroF1 = 0.9397` | 运行 `code/train_cls.py` |
| 11 | PLM 训练曲线 | 打开 `figures/train_curve_bert.png` |
| 12 | PLM 推理预测结果 | 运行 `code/inference_cls.py` |

以上 12 张截图存于 `screenshots/`（索引见 `screenshots/README.md`），
并已全部嵌入 `实验报告/新生任务实验报告.docx`。

---

## 四、已提交到 GitHub

仓库地址：**https://github.com/Nicolas-Varon-August/xinsheng-tasks**

该仓库包含本文件、三个任务的完整代码与运行输出、
`screenshots/` 下 12 张 PyCharm 实际运行截图，以及 `实验报告/新生任务实验报告.docx`（Word 版，含全部截图）。

若之后有修改需要重新上传：

```powershell
cd "C:\Users\33592\OneDrive\文档\ChatGPT\新生任务"
git add .
git commit -m "修改说明"
git push
```

> `.gitignore` 已排除约 900 MB 的图数据集缓存与 147–390 MB 的 PLM 微调权重
> （可用第二节的命令重新生成）。

## 五、目录总览

```
新生任务/
├── README.md                     # 本文件：总览 + 运行指引 + 截图清单
├── .gitignore
├── requirements.txt              # 汇总依赖
├── 01-NLP任务/                    # 任务一：从零实现 Transformer
├── 02-Graph任务/                  # 任务二：GNN 节点分类
├── 03-PLM任务/                    # 任务三：预训练模型微调
├── 实验报告/                      # 实验报告（环境/方法/结果/问题）
└── screenshots/                  # 运行截图 12 张（PyCharm 实际运行，含索引 README）
```
