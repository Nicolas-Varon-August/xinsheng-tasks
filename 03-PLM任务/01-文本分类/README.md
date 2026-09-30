# 任务三：预训练语言模型微调 —— 中文评论八分类

> 上游任务：<https://github.com/HarderThenHarder/transformers_tasks> 的 `text_classification` 子任务

## 目录结构

```
01-文本分类/
├── code/
│   ├── train_cls.py         # 训练主脚本（复刻上游流程，适配 transformers 5.x）
│   ├── inference_cls.py     # 加载微调模型做预测
│   ├── train.upstream.py    # 上游原文件（对照用）
│   ├── utils.upstream.py
│   └── train.upstream.sh
├── data/comment_classify/   # train.txt / dev.txt / label_mapping.txt（直接复用上游数据）
├── ckpt/                    # 微调后的权重（已 gitignore，体积 147–390 MB）
├── figures/                 # 训练曲线
├── results/                 # 指标 JSON
└── README.md
```

数据格式：每行 `label<TAB>text`，8 个类别（0 电脑 / 1 水果 / 2 平板 / 3 书籍 /
4 衣服 / 5 酒店 / 6 蒙牛 / 7 洗浴），训练 402 条、验证 63 条。

## 环境

```
Python 3.10.20 / PyTorch 2.11.0+cu128 / transformers 5.15.0 / tokenizers 0.22.2
scikit-learn 1.7.2 / numpy 2.2.6 / matplotlib 3.10.9
GPU: NVIDIA RTX 4070 Laptop (8GB)
```

```bash
pip install torch transformers tokenizers scikit-learn numpy matplotlib
# 国内下载模型：
#   Windows PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET = "1"
```

## 运行指令

在 `01-文本分类` 目录下执行：

```bash
python code/train_cls.py --model bert-base-chinese --tag bert        # 约 80 s
python code/train_cls.py --model hfl/rbt3         --tag rbt3        # 约 26 s
python code/train_cls.py --model hfl/rbt3 --loss focal --tag rbt3_focal
python code/inference_cls.py                                        # 交互预测
```

参数：`--model --tag --epochs --batch-size --lr --max-seq-len --loss {cross_entropy,focal} --device`。

## 结果（dev 集，seed=42，按 macro F1 保存最优）

| 模型 | 参数量 | loss | 最优 epoch | dev acc | macro P | macro R | macro F1 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| **bert-base-chinese** | 102 M | 交叉熵 | 5 | **0.9206** | 0.9510 | 0.9327 | **0.9397** | 79.3 s |
| hfl/rbt3 | 38.5 M | 交叉熵 | 14 | 0.8254 | 0.8689 | 0.9062 | 0.8713 | 25.5 s |
| hfl/rbt3 | 38.5 M | Focal | 8 | 0.8095 | 0.8575 | 0.8880 | 0.8574 | 25.8 s |

推理示例：

```text
[水果] (p=0.999)  苹果很甜，个头也大
[书籍] (p=0.992)  这本书内容一般，纸张也薄
```

## 注意

- 上游 `requirements.txt` 固定 `transformers==4.22.1`，与本机 torch 2.11 不兼容，
  因此这里按其设计复刻了等价训练流程（保留 Focal Loss 选项与逐类指标），
  上游原文件保留为 `*.upstream.py` 以便对照。
- `ckpt/` 已被 `.gitignore` 排除（单文件超过 GitHub 100 MB 限制），权重可用上面的命令复现。

详见 [../../实验报告/03-PLM实验报告.md](../../实验报告/03-PLM实验报告.md)。