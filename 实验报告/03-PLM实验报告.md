# 实验报告三：预训练语言模型微调——中文评论八分类

> 对应任务：PLM 任务 —— <https://github.com/HarderThenHarder/transformers_tasks>
> 选用其中的 `text_classification` 子任务
> 代码位置：`03-PLM任务/01-文本分类/`

---

## 一、实验环境与条件

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.10.20（conda 环境 `xinsheng`） |
| 深度学习框架 | PyTorch 2.11.0+cu128 |
| 预训练模型库 | transformers 5.15.0 + tokenizers 0.22.2 |
| 其他依赖 | scikit-learn 1.7.2、numpy 2.2.6、matplotlib 3.10.9 |
| GPU | NVIDIA GeForce RTX 4070 Laptop，8 GB 显存 |
| 预训练模型 | `hfl/rbt3`（3 层 RoBERTa，38.5 M 参数）与 `bert-base-chinese`（12 层 BERT，102 M 参数） |
| 下载通道 | `HF_ENDPOINT=https://hf-mirror.com`，并设 `HF_HUB_DISABLE_XET=1` 绕过 xet 协议 |

### 数据集

原仓库自带的中文商品评论数据集，8 个类别：

| label | 类别 | 训练数 | 验证数 |
|---|---|---|---|
| 0 | 电脑 | 22 | 2 |
| 1 | 水果 | 78 | 7 |
| 2 | 平板 | 67 | 12 |
| 3 | 书籍 | 29 | 3 |
| 4 | 衣服 | 70 | 16 |
| 5 | 酒店 | 56 | 16 |
| 6 | 蒙牛 | 13 | 1 |
| 7 | 洗浴 | 67 | 6 |
| — | 合计 | 402 | 63 |

文本长度：最短 5 字，中位数 34 字，最长 1552 字（截断到 128 token）。
数据集**规模极小且类别不平衡**（最多的「水果」78 条 vs 最少的「蒙牛」13 条，差 6 倍），
验证集里「蒙牛」只有 1 条 —— 这是后面结果波动大的根本原因。

## 二、实验方法与过程

### 2.1 为什么没有直接跑原仓库的 train.py

原仓库 `requirements.txt` 固定了 `transformers==4.22.1` / `datasets==2.4.0` / `evaluate==0.2.2`，
与本机 torch 2.11 所需的 transformers 5.x **不兼容**。
因此我按其 `train.py` 的设计**复刻了一套等价的训练流程**，保留了它的核心特性：

| 原仓库特性 | 复刻实现 |
|---|---|
| AutoTokenizer + AutoModelForSequenceClassification | 相同 |
| label\ttext 格式的数据文件 | 相同（直接复用原仓库数据） |
| AdamW + 线性 warmup 调度 | torch.optim.AdamW + get_scheduler("linear") |
| bias / LayerNorm 不做 weight decay | 参数分组，与原文一致 |
| 可切换 cross_entropy / focal_loss | --loss 参数，自实现 FocalLoss(alpha=0.25, gamma=2.0) |
| 每个类别的 P/R/F1 | 用 sklearn 逐类计算并写入结果 JSON |

### 2.2 实现内容

| 文件 | 内容 |
|---|---|
| code/train_cls.py | 训练主脚本：数据读取、tokenize、AdamW + warmup、训练/验证循环、按 macro F1 保存最优模型、输出指标 JSON 与训练曲线 |
| code/inference_cls.py | 加载微调后的模型做单句/批量推理，映射回中文类别名 |
| code/train.upstream.py 等 | 原仓库对应文件，便于对照 |

### 2.3 操作步骤（可复现）

```bash
cd 03-PLM任务/01-文本分类

python code/train_cls.py --model bert-base-chinese --tag bert
python code/train_cls.py --model hfl/rbt3         --tag rbt3
python code/train_cls.py --model hfl/rbt3 --loss focal --tag rbt3_focal

python code/inference_cls.py
```

统一超参：`max_seq_len=128, batch_size=16, lr=5e-5, weight_decay=0.01, warmup_ratio=0.06,
epochs=20, seed=42`；按 **dev macro F1** 保存最优 checkpoint。

## 三、实验结果及分析

### 3.1 总体结果

| 模型 | 参数量 | loss | 最优 epoch | dev acc | macro P | macro R | macro F1 | 训练耗时 |
|---|---|---|---|---|---|---|---|---|
| bert-base-chinese | 102 M | 交叉熵 | 5 | 0.9206 | 0.9510 | 0.9327 | **0.9397** | 79.3 s |
| hfl/rbt3 | 38.5 M | 交叉熵 | 14 | 0.8254 | 0.8689 | 0.9062 | 0.8713 | 25.5 s |
| hfl/rbt3 | 38.5 M | Focal Loss | 8 | 0.8095 | 0.8575 | 0.8880 | 0.8574 | 25.8 s |

### 3.2 逐类 F1

| 类别 | bert-base-chinese | rbt3（交叉熵） | rbt3（Focal） |
|---|---|---|---|
| 电脑 | 1.000 | 1.000 | 1.000 |
| 水果 | 0.9231 | 0.7778 | 0.8235 |
| 平板 | 0.9091 | 0.7826 | 0.8333 |
| 书籍 | 1.000 | 1.000 | 1.000 |
| 衣服 | 0.8824 | 0.6400 | 0.6400 |
| 酒店 | 0.9697 | 0.9697 | 0.9375 |
| 蒙牛 | 1.000 | 1.000 | 1.000 |
| 洗浴 | 0.8333 | 0.8000 | 0.6250 |

### 3.3 训练过程（bert-base-chinese）

| epoch | train loss | dev acc | macro F1 |
|---|---|---|---|
| 1 | 1.8993 | 0.7937 | 0.6338 |
| 2 | 0.8222 | 0.7302 | 0.5567 |
| 3 | 0.3619 | 0.8730 | 0.9044 |
| 4 | 0.1430 | 0.8571 | 0.9080 |
| 5 | 0.1324 | 0.9206 | 0.9397 |
| 9 - 13 | 约 0.002 | 0.8889 | 0.9198 |
| 14 - 20 | 约 0.001 | 0.8730 | 0.9063 |

### 3.4 结果分析

1. **模型规模的作用非常明显**：`bert-base-chinese`（102 M）的 macro F1 比 `hfl/rbt3`（38.5 M）
   高 **6.8 个百分点**（0.9397 vs 0.8713），dev 准确率高 9.5 个百分点。
   数据只有 402 条，小模型（rbt3 只有 3 层）容量不足以拟合这种细粒度的商品类别区分，
   而大模型凭借预训练阶段学到的语义先验，即使微调样本很少也能取得好效果。
2. **收敛极快且严重过拟合**：bert 在第 **5** 个 epoch 就达到最优（macro F1 0.9397），
   此后 train loss 一路降到 0.0012（几乎完全记住训练集），但 dev 指标反而下滑并稳定在 0.9063。
   这提醒我们在小数据集上必须**按验证集早停**，而不是固定 epoch 数训练到底 ——
   本项目按 macro F1 保存最优 checkpoint，最终提交的是第 5 轮的模型。
3. **Focal Loss 在本任务上没有收益**：rbt3+Focal 的 macro F1（0.8574）略低于交叉熵（0.8713）。
   原因分析：Focal Loss 通过 (1-p_t)^gamma 抑制「已经学好」的样本，理论上适合类别不平衡；
   但本任务验证集每类只有 1 - 16 条，`衣服`（F1 0.64）和`洗浴`（0.625 / 0.80）这两类本身
   样本少、语义又和「酒店/水果」接近，Focal 反而让模型对难样本过度敏感，
   在仅 63 条验证样本上的差异（0.014）很难说显著。要得到可靠结论需要多次随机种子重复实验。
4. **各类别难度**：`电脑`/`书籍`/`蒙牛` 三个类在所有模型上 F1 都是 1.000，
   因为它们的文本包含强烈的领域关键词（「电脑」「书」「蒙牛」）。
   最难的是 **`衣服`（0.64 - 0.88）**，从定性上看，服装评论里大量出现「质量」「不错」「性价比」
   这类跨类别的通用词，模型容易和「水果」「洗浴」混淆。
5. **小验证集的统计风险**：`蒙牛` 在 dev 里只有 1 条样本，预测对错会让该类的 F1 在 0 和 1 之间跳变，
   macro F1 因此波动很大（例如 rbt3 第 5 到第 6 个 epoch 从 0.8338 跳到 0.8671）。
   更严谨的做法是交叉验证或合并 dev/test 后再评估。

## 四、遇到的问题及处理情况

| # | 问题 | 现象 | 处理 |
|---|---|---|---|
| 1 | 原仓库依赖与本机 torch 不兼容 | transformers==4.22.1 无法在 torch 2.11 上运行 | 不降级 torch，改为按原仓库设计复刻一套等价训练流程（见 2.1），并用 --loss 保留 Focal Loss 选项 |
| 2 | 大模型下载超时 | 下载 bert-base-chinese 时 OSError: We couldn't connect to 'https://hf-mirror.com' / httpx.ReadTimeout | 先加大超时（HF_HUB_ETAG_TIMEOUT=60、HF_HUB_DOWNLOAD_TIMEOUT=300）；改用 huggingface_hub.snapshot_download 整仓库拉取 |
| 3 | hf-mirror 不支持 xet 传输协议 | HTTP 404 Not Found ... /xet-read-token/... | 设置 HF_HUB_DISABLE_XET=1 强制走普通 HTTP 下载，随即成功（10 个文件 3m38s） |
| 4 | 无网络时重复校验拖慢训练 | 每次 from_pretrained 都要联网校验 | 权重缓存到本地后即可离线加载（local_files_only=True 亦可），训练脚本默认复用缓存 |
| 5 | 结果文件互相覆盖 | 三组实验的指标写同一个文件名 | 输出文件名带 --tag：metrics_bert.json / metrics_rbt3.json / metrics_rbt3_focal.json |
| 6 | 微调模型体积大 | model.safetensors 达 147 MB / 390 MB，超过 GitHub 单文件 100 MB 限制 | 在 .gitignore 中排除 ckpt/，仓库只提交代码、数据、指标与曲线图，权重由训练脚本复现 |

## 五、运行截图清单（提交时附上）

1. `python code/train_cls.py --model bert-base-chinese --tag bert`
   → 显示 `[epoch 5] ... macroF1=0.9397` 与 `最佳 macroF1 = 0.9397` 的输出
2. `python code/train_cls.py --model hfl/rbt3 --tag rbt3`
3. `python code/train_cls.py --model hfl/rbt3 --loss focal --tag rbt3_focal`
4. `python code/inference_cls.py`
   → 显示 `[水果] (p=0.999) 苹果很甜，个头也大` 等预测
5. 打开 `figures/train_curve_bert.png` 的截图
6. `results/metrics_bert.json` 在 PyCharm 中打开的截图

## 六、结论

1. 复刻了 transformers_tasks/text_classification 的完整训练流程（含 Focal Loss 选项与逐类指标），
   并解决了原仓库依赖与本机环境不兼容的问题。
2. bert-base-chinese 在 402 条训练样本上取得 **dev acc 0.9206 / macro F1 0.9397**；
   hfl/rbt3 为 0.8254 / 0.8713。**模型规模是本任务最重要的影响因素**。
3. 小数据集上过拟合极快（第 5 轮最优，之后 train loss 降到 0.001 而 dev 指标下降），
   验证了「小样本微调必须早停」的经验。
4. Focal Loss 在该任务上未带来提升，说明类别不平衡的缓解手段需要结合具体的数据规模来评估。