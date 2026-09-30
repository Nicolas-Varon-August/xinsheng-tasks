# 运行截图

本目录存放实验运行截图，全部在 PyCharm Community Edition 2024.3.4 中运行得到，
解释器为 conda 环境 `xinsheng`（`E:\env\mea\xinsheng\python.exe`）。

| 文件名 | 对应内容 |
|---|---|
| `04-pycharm-interpreter.png` | PyCharm 解释器配置页（conda 环境 `xinsheng`） |
| `01-nlp-download.png` | `data/download.py` 下载 ChnSentiCorp（train 9600 / validation 1200 / test 1200） |
| `01-nlp-train.png` | `train.py` 训练过程与 `最佳 dev 准确率 = 0.8417` |
| `01-nlp-eval.png` | `eval/run.py` 三项自检全部 `[通过]` |
| `01-nlp-attention-run.png` | `viz_attention.py --auto-head` 的输出 |
| `01-nlp-attention.png` | 注意力热图 `figures/attn_negative.png` |
| `01-nlp-toylm.png` | `toy_lm.py` 的困惑度与生成样例 |
| `02-graph-cora.png` | `train_node_cls.py --dataset cora --run-all` 的 8 组对比 |
| `02-graph-result-json.png` | `results/node_cls_cora.json` 结果文件 |
| `03-plm-bert.png` | `train_cls.py --model bert-base-chinese --tag bert` |
| `03-plm-curve.png` | 训练曲线 `figures/train_curve_bert.png` |
| `03-plm-inference.png` | `inference_cls.py` 的预测结果 |

Word 版实验报告见 `实验报告/新生任务实验报告.docx`（含上述全部截图）。

具体操作见根目录 `README.md` 第三节「PyCharm 运行与截图指引」。