# 实验报告一：从零实现 Transformer 并在中文情感分类上验证

> 对应任务：NLP 任务 —— <https://github.com/FudanNLP/nlp-beginner>（现为 LLM-Beginner 系列）
> 任务一《熟悉 Transformer》，详见 `01-NLP任务/task-1-transformer/README.upstream.md`

---

## 一、实验环境与条件

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.10.20（conda 环境 `xinsheng`，由 `lnp_gpu` 克隆而来） |
| 深度学习框架 | PyTorch 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 4070 Laptop，8 GB 显存（驱动 572.83 / CUDA 12.8） |
| 关键依赖 | transformers 5.15.0、datasets 2.19.0、pandas 2.3.3、pyarrow 25.0.1、matplotlib 3.10.9 |
| 数据集 | ChnSentiCorp（中文酒店评论情感二分类）：训练 9600 / 验证 1200 / 测试 1200 |
| 硬件 | 训练耗时 GPU 约 48 s（8 epoch） |

环境安装（一次性）：

```bash
conda create -n xinsheng --clone lnp_gpu      # 复用已有 CUDA PyTorch，避免重复下载数 GB
conda activate xinsheng
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets pandas pyarrow matplotlib
```

## 二、实验方法与过程

### 2.1 任务目标

**不调用 `nn.MultiheadAttention` 等高层封装、不加载任何预训练模型**，用约 300 行 PyTorch
从零写出 Transformer encoder，在 ChnSentiCorp 上做情感二分类，目标 dev 准确率 ≥ 0.80，
并输出注意力热图说明模型"在看什么"。

### 2.2 实现内容

| 文件 | 内容 |
|---|---|
| `src/attention.py` | `scaled_dot_product_attention(Q,K,V,mask)`（缩放点积 + mask + softmax + 加权求和）、`MultiHeadAttention`（分头 / QKV 投影 / 输出投影）、`build_causal_mask`、`build_padding_mask` |
| `src/block.py` | `TransformerBlock` = 多头注意力 + 前馈网络 + 两处残差 + 两处 LayerNorm（Pre-LN） |
| `src/model.py` | `TransformerClassifier`（词嵌入 + 正弦位置编码 + N 层 block + 池化 + 分类头）与 `load_for_eval` |
| `src/tokenizer.py` | 字符级分词器，词表从训练集统计（`<pad>/<unk>/<cls>` + 出现≥2次的字符） |
| `train.py` | AdamW + 线性 warmup + 余弦衰减、padding mask、梯度裁剪、按 dev 准确率保存最优 checkpoint |
| `toy_lm.py` | 给同一个注意力加因果掩码，用唐诗语料训练字符级 toy 语言模型（DoD 的 M4） |
| `viz_attention.py` | 注意力热图可视化，自动挑选"最聚焦"的 (层, 头) |

### 2.3 关键实现细节

1. **mask 用 `-inf` 而不是乘 0**：乘 0 之后 softmax 仍会把概率质量分配给被屏蔽位置，
   用 `masked_fill(mask, -inf)` 才能让 PAD / 未来词元的权重真正为 0。
   另外对整行被屏蔽的情况用 `torch.nan_to_num` 兜底，避免 NaN。
2. **缩放因子是 `sqrt(d_k)` 而不是 `sqrt(d_model)`**：多头里 `d_k = d_model / n_heads`。
3. **Pre-LN 结构**：`x = x + Attn(LN(x)); x = x + FFN(LN(x))`，比原论文的 Post-LN 更容易训稳，
   不加 warmup 也不发散。
4. **mean pooling 要排除 PAD 位置**：`sum(x * keep) / keep.sum()`，否则句长不同会引入偏差。

### 2.4 训练超参

```bash
python train.py --epochs 8 --batch-size 64 --lr 3e-4 --d-model 128 \
                --n-heads 4 --n-layers 4 --d-ff 256 --max-len 200 --pooling mean
```

模型参数量 **1.00 M**，词表 **3659**。

### 2.5 操作步骤（可复现）

```bash
cd 01-NLP任务/task-1-transformer
python data/download.py     # 下载 ChnSentiCorp -> data/{train,validation,test}.parquet
python train.py             # 训练，产出 ckpt/best.pt 与 results/train_log_base.json
python eval/run.py          # 自检，产出 eval/result.json
python viz_attention.py --auto-head   # 注意力热图 -> figures/*.png
python toy_lm.py            # 因果掩码 + toy 语言模型
```

## 三、实验结果及分析

### 3.1 自检结果（`eval/result.json`）

```text
[通过] attention_correctness: {"pass": true, "max_abs_diff": 9.5367431640625e-07}
[通过] causal_mask:           {"pass": true, "leaked_diff": 0.0}
[通过] classifier_accuracy:   {"pass": true, "accuracy": 0.8417, "baseline_reference": 0.85}
```

- **M1 注意力数值一致性**：与官方 `F.scaled_dot_product_attention` 的最大绝对误差 9.5e-07，
  远小于 1e-5 的阈值，说明手写实现与框架实现数值等价。
- **M4 因果掩码**：把最后一个位置的 V 改成 999 后，前面所有位置的输出**完全不变**（差异 0.0），
  证明未来词元没有泄漏。
- **M3 分类准确率**：dev 准确率 **0.8417**，超过 0.80 的通过线，接近 0.85 的参考基线。

### 3.2 训练过程

| epoch | train loss | train acc | dev loss | dev acc |
|---|---|---|---|---|
| 1 | 0.6701 | 0.5950 | 0.6426 | 0.6175 |
| 2 | 0.5996 | 0.6853 | 0.6504 | 0.7233 |
| 3 | 0.4638 | 0.7931 | 0.6002 | 0.7825 |
| 4 | 0.4229 | 0.8163 | 0.4450 | 0.8275 |
| 5 | 0.3783 | 0.8432 | 0.4566 | 0.8192 |
| **6** | 0.3526 | 0.8556 | 0.4210 | **0.8417**（最优） |
| 7 | 0.3307 | 0.8649 | 0.4416 | 0.8342 |
| 8 | 0.3234 | 0.8688 | 0.4135 | 0.8417 |

总耗时 48.3 s。

**分析**：
1. 前 3 个 epoch 准确率快速上升（0.62 → 0.78），第 4 个 epoch 就跨过 0.80 的及格线，
   说明字符级 Transformer 在 9.6K 中文句子上很快能学到情感词。
2. 第 5、7 epoch 出现 **dev 准确率回落而 train 准确率继续上升**的典型过拟合征兆
   （train acc 0.87 vs dev acc 0.83，gap 约 4 个点）。由于每轮都按 dev 准确率保存最优
   checkpoint，最终用第 6 / 第 8 epoch 的模型，不受回落影响。
3. 收敛后 train loss 仍在缓慢下降而 dev loss 已经走平（0.41~0.44 波动），
   继续训练收益有限；若追求 >0.88 需要更强正则（更大 dropout、weight decay）或更多数据。

### 3.3 注意力热图（M5）

`viz_attention.py` 会先遍历全部 (层, 头)，用**平均注意力熵**衡量"聚焦程度"，自动选出最聚焦的
`layer=2, head=0`（熵 2.874，而 layer=3 的若干头熵在 3.5 以上，接近均匀分布）。

以负面样本 *"非常失望的一次入住体验，房间有异味，隔音差，前台态度也很冷漠。"* 为例：

- 该句被正确预测为**负面**，置信度 0.950；
- 第 2 层第 0 头在 `失/望`、`前/台/态/度`、`冷/漠` 几列上出现明显高亮列，
  说明这一头学到的是"把整个句子对齐到评价性短语"的能力（该头最大权重约 0.2，而随机头仅 0.045）；
- 而 `<cls>` 行与 `非` 行的权重分散，说明浅层/其他位置承担的是句法聚合而非情感打分。

4 张热图见 `01-NLP任务/task-1-transformer/figures/`：
`attn_positive.png`、`attn_negative.png`、`attn_long.png`、`attn_positive2.png`。

### 3.4 toy 语言模型（M4 的另一半）

用 `poetryFromTang.txt`（16647 字符、2515 字表）训练 2 层 decoder-only 小模型（400 步，3.9 s）：

| 指标 | 数值 |
|---|---|
| 随机初始化验证困惑度 | 2937.2 |
| 训练 400 步后验证困惑度 | 636.64 |
| 因果掩码泄漏检测 | 差异 0.000e+00（通过） |

生成样例（prompt = "床前明月光"）：

```text
床前明月光。岁公昔为曲，洋烛水在。从藿不不得，动，筹郁对涛。
所愧寐，吟机不见形。银伦
```

**分析**：困惑度下降 4.6 倍，说明模型确实学到了字符级转移规律；但因为只有 1.6 万字语料、
400 步、2 层，生成结果仍是"字词搭配像诗、语义不通"的状态，这符合小模型 + 小语料的预期。

## 四、遇到的问题及处理情况

| # | 问题 | 现象 | 处理 |
|---|---|---|---|
| 1 | `datasets` 版本过高 | `load_dataset("seamew/ChnSentiCorp")` 报 `RuntimeError: Dataset scripts are no longer supported` | 新版 datasets（≥4.0）已移除脚本型数据集支持，降级为 `datasets==2.19.0` |
| 2 | 原始数据托管在 Google Drive | 脚本型数据集内部去 `drive.google.com` 取数，`ConnectTimeout` | 改为直接拉取社区维护的 parquet 版本 `lansinuote/ChnSentiCorp`，并重写 `data/download.py` 自动匹配 train/validation/test 分片、统一列名为 `text`/`label` |
| 3 | 国内访问 HuggingFace 不稳定 | 下载慢/超时 | 统一设置 `HF_ENDPOINT=https://hf-mirror.com`（脚本内已默认设置） |
| 4 | 克隆环境缺少 `mpmath` | `AdamW` 初始化时 `sympy` 导入失败 | `pip install --ignore-installed mpmath==1.3.0`（用户级 site-packages 里的版本该环境看不到） |
| 5 | Windows 控制台中文乱码 | 打印中文变成乱码 | 在脚本里 `sys.stdout.reconfigure(encoding="utf-8")`（`src/console.py`） |
| 6 | 注意力热图看不出结构 | 随机选 `layer=-1, head=0` 时权重几乎均匀（最大仅 0.045） | 增加按"平均注意力熵"自动选头，改用 `layer=2, head=0`（最大权重 0.24，结构清晰） |

## 五、运行截图清单（提交时附上）

在 PyCharm 中依次运行下列命令并截图（建议截 **Run 工具窗口的完整输出**）：

1. `python data/download.py` → 显示 `train: 9600 条 ...` 的截图
2. `python train.py` → 显示 `[epoch 1] ... [epoch 8]`、`最佳 dev 准确率 = 0.8417` 的截图
3. `python eval/run.py` → 显示三个 `[通过]` 的截图
4. `python viz_attention.py --auto-head` → 显示预测结果与 `figures/attn_*.png` 的截图
5. `python toy_lm.py` → 显示困惑度下降与生成样例的截图
6. 打开 `figures/attn_negative.png` 的截图

## 六、结论

1. 手写注意力与官方实现在数值上等价（误差 < 1e-6），因果掩码实现正确（无未来信息泄漏）。
2. 从零训练的字符级 Transformer 在 ChnSentiCorp 上取得 **0.8417** 的 dev 准确率，
   超过 0.80 的任务通过线。
3. 注意力可视化表明模型确实把权重集中到"失望""前台态度""冷漠"等评价性词元上，
   具有一定的可解释性。
4. DoD 五项（M1–M5）全部完成。
