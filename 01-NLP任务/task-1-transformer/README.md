# 任务一：熟悉 Transformer（从零实现 + 中文情感分类）

> 上游任务说明见 [README.upstream.md](README.upstream.md)（来自
> [FudanNLP/nlp-beginner](https://github.com/FudanNLP/nlp-beginner) 的 `task-1-transformer`）。
> 本文件说明我自己的实现、环境与运行方式。

## 目录结构

```
task-1-transformer/
├── src/
│   ├── attention.py      # 手写 scaled_dot_product_attention + MultiHeadAttention + mask 工具
│   ├── block.py          # TransformerBlock = Attn + FFN + 2xResidual + 2xLayerNorm (Pre-LN)
│   ├── model.py          # TransformerClassifier + load_for_eval 工厂
│   ├── tokenizer.py      # 字符级分词器（词表从训练集统计）
│   └── console.py        # Windows 控制台 UTF-8 输出兼容
├── train.py              # 训练脚本（AdamW + warmup + 余弦衰减 + 早停保存）
├── toy_lm.py             # 因果掩码自检 + 唐诗字符级 toy 语言模型
├── viz_attention.py      # 注意力热图（自动挑选最聚焦的 层/头）
├── data/download.py      # 下载 ChnSentiCorp -> data/{train,validation,test}.parquet
├── eval/run.py           # 上游自检脚本（3 项测试）
├── ckpt/best.pt          # 训练好的权重（约 4 MB）
├── figures/              # 注意力热图
└── results/              # 训练日志 JSON
```

## 环境

```
Python 3.10.20 / PyTorch 2.11.0+cu128 / transformers 5.15.0 / datasets 2.19.0
pandas 2.3.3 / pyarrow 25.0.1 / matplotlib 3.10.9
GPU: NVIDIA RTX 4070 Laptop (8GB)，CPU 也可运行（慢约 10 倍）
```

```bash
pip install -r requirements.txt
```

## 运行指令

在 `task-1-transformer` 目录下依次执行：

```bash
python data/download.py                 # 1. 下载数据（约 9600/1200/1200 条）
python train.py                         # 2. 训练（约 50 s，产出 ckpt/best.pt）
python eval/run.py                      # 3. 自检（3 项，产出 eval/result.json）
python viz_attention.py --auto-head     # 4. 注意力热图（产出 figures/*.png）
python toy_lm.py                        # 5. 因果掩码自检 + toy 语言模型
```

`train.py` 常用参数：`--epochs --batch-size --lr --d-model --n-heads --n-layers
--d-ff --max-len --dropout --pooling {mean,cls} --tag`。

## 实验结果

### 自检（`eval/result.json`）

```text
[通过] attention_correctness: {"pass": true, "max_abs_diff": 9.5367431640625e-07}
[通过] causal_mask:           {"pass": true, "leaked_diff": 0.0}
[通过] classifier_accuracy:   {"pass": true, "accuracy": 0.8417, "baseline_reference": 0.85}
```

### 训练（d_model=128, n_heads=4, n_layers=4, d_ff=256, lr=3e-4, batch=64, 8 epochs）

| 指标 | 数值 |
|---|---|
| 词表大小 | 3659（字符级） |
| 参数量 | 1.00 M |
| 最佳 dev 准确率 | **0.8417**（epoch 6） |
| 训练总耗时 | 48.3 s |

### toy 语言模型（唐诗语料，400 步）

| 指标 | 数值 |
|---|---|
| 随机初始化困惑度 | 2937.2 |
| 训练后验证困惑度 | 636.64 |
| 因果掩码泄漏检测 | 0.000e+00（通过） |

生成样例（prompt = "床前明月光"）：

```text
床前明月光。岁公昔为曲，洋烛水在。从藿不不得，动，筹郁对涛。
所愧寐，吟机不见形。银伦
```

### 注意力热图

`figures/` 下 4 张：`attn_positive.png`、`attn_negative.png`、`attn_long.png`、`attn_positive2.png`。
`--auto-head` 会按平均注意力熵自动选出最聚焦的 `layer=2, head=0`（随机头熵 > 3.5，该头熵 2.874）。

## 踩坑记录

1. `datasets>=4.0` 移除了脚本型数据集支持，需用 `datasets==2.19.0`；
2. `seamew/ChnSentiCorp` 原始数据在 Google Drive，国内不可达，已改为拉取 parquet 镜像
   `lansinuote/ChnSentiCorp`；
3. 需要 `HF_ENDPOINT=https://hf-mirror.com`；
4. mask 必须用 `-inf` 而不是乘 0，否则被屏蔽位置仍有概率质量；
5. 缩放因子是 `sqrt(d_k)` 不是 `sqrt(d_model)`；
6. mean pooling 必须排除 PAD 位置。

详见 [../../实验报告/01-NLP实验报告.md](../../实验报告/01-NLP实验报告.md)。