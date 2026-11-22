# 新基线批量训练与验证

聚合默认 2 层。每层从类间注意力更新后的 encoder feature 并行计算五路类内注意力。36×36 和 24×24 各使用普通窗口与移位窗口，12×12 使用一次全图窗口。五路更新量对齐到 36×36 后，取均值并加回 encoder feature。默认每层五路共享一套参数，`model.aggregation.share_window_weights_across_scales=False` 时每层五路各有一套独立参数。

## 训练

```bash
python tools/train_experiments.py configs/suites/paper_train.py --dry-run
python tools/train_experiments.py configs/suites/paper_train.py
```

Full 使用 batch 4、10000 步、LR 1e-4，RemoteCLIP 学习率乘 0.01。前 500 步预热，后 9500 步相对余弦衰减，W&B 使用离线模式。每 1000 步保存并在 LoveDA 七类上验证。实验目录为 `work_dirs/multiscale_windows_v2/full/seed_42/train`，实现标识为 `reca-sam3-multiscale-windows-v2`。

`--devices cuda:0 cuda:1` 为每个独立任务分配设备；种子由清单 `seeds=[42]` 控制。`--resume` 继续配置一致的新实验并跳过已完成任务；旧结构不能恢复到新结构。

## 手选权重验证

在 `configs/suites/paper_validate.py` 的 Full selection 中填写本次训练的权重路径：

```python
selections=[dict(variant="full", seed=42,
    checkpoint="work_dirs/multiscale_windows_v2/full/seed_42/train/best.pth")]
```

```bash
python tools/validate_experiments.py configs/suites/paper_validate.py --dry-run
python tools/validate_experiments.py configs/suites/paper_validate.py
```

`None` 保持待选，不自动选择 best/latest。验证从权重旁 `config.json` 恢复模型结构，核对身份、指纹和哈希；同一权重用于全部八数据集：DLRSD、iSAID、LoveDA、Potsdam、Vaihingen、UAVid、UDD、VDD。

两项原始 SAM3 对比（Semantic-only、Semantic × Presence）仅进入验证队列，使用原始预训练权重，保持 training-free 和 step 0。数据协议、日志与结果 schema 不变。独立验证按变体、种子、数据集保存，不覆盖训练记录。

CPU 合成检查不产生论文成绩；正式预训练权重的 GPU 训练和八域评测需在数据就绪的机器上执行。
