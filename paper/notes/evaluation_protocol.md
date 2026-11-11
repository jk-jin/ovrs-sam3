# 实验协议登记（定稿前必须补齐）

## 当前代码的开发与 checkpoint 选择

训练入口：`configs/experiments/main/isaid_loveda_full.py`。
训练划分：iSAID `img_dir/train` 与 `ann_dir/train`，15 个配置类别，标签 255 忽略。
选择划分：LoveDA `img_dir/val` 与 `ann_dir/val`。
选择指标：`semantic.miou`（LoveDA 七类，含 background），取最大值。
每 2000 步完整验证；`val_max_iters=None`。验证结果的标准 JSON 记录实际样本数、期望样本数、样本清单散列和 `is_full`；TODO：用真实数据核实清单散列。

现有 `configs/test/loveda.py` 也指向 LoveDA val，因此并非自动切换到独立正式测试集。该入口与训练入口都使用七类 `semantic.miou`；不再声明六类 metric group。

## 正式评测（与选择记录分开）

| 字段 | 当前状态 |
| --- | --- |
| 数据集与正式划分 | TODO，尚未确定 |
| 图像与标签路径/清单及散列 | TODO |
| 与选择划分是否独立 | TODO，需明确证据 |
| 标签可用性或官方评测服务 | TODO，未验证 |
| checkpoint 文件/散列/迭代 | TODO，当前无指定可报告权重 |
| 冻结配置与后处理日期 | TODO |
| checkpoint 选择依据与选择分数 | 单独记录，不替代正式分数 |
| 输入尺寸、同义提示、TTA、双门阈值 | TODO，正式运行前锁定 |
| iSAID 与目标类别语义重叠 | TODO，不能直接声称严格未见类 |

如果正式结果仍取自 LoveDA val 且已经用其选模型，应显式称开发/选择结果。若采用无目标标签选择策略，也须在实验前明确并记录，不能事后根据目标结果重选 checkpoint。

## LoveDA 指标口径

主表：7 类 mIoU，包含 background、building、road、water、barren、forest、agricultural。
保存源：标准结果 `results/<eval_id>.json` 的 `metrics.miou`、`metrics.per_class` 与 `evaluation` 元数据；本地 JSONL 保存同一份逐类指标。
不再另存六类前景 mIoU，也不把六类外部结果补成七类。七类口径需逐类并集非零；当前评测器排除零并集类别，并在标准结果中记录有效类别数。若类别缺失，记录有效类别数并披露，不能悄悄称其为七类平均。
内部评测值与标准结果 JSON 统一为 fraction；论文导出 CSV/LaTeX 时按 `display_unit="percent"` 只转换一次。逐类结果保留 intersection/union/target_pixels 便于复核；缺失值写 null，空白表示未运行。

LoveDA 数据配置使用 `reduce_zero_label=False` 与 `background_cfg.enabled=True, exclude_from_forward=True`：评测标签保留 0..6，前向删除背景并把前景映射为 0..5，评测后处理把前景预测 +1、低分回填背景。不得再叠加 `reduce_zero_label`，否则会二次删除背景并平移索引。`tests/test_config_effective.py` 已用构造标签覆盖该映射；TODO：仍须用真实标签样本核实源标签值域与无效值。

## 比较组与消融约束

无额外遥感训练组：SAM 3、SegEarth-OV3，具体版本、权重及原协议 TODO。“无额外”仅指本轮遥感分割训练，不否定已有遥感预训练。
同 iSAID 训练组：简单 SAM 3 adapter、其他待确认对照、ReCA-SAM3。简单 adapter 结构及参数/预算 TODO；所有变体记录可训练范围与额外数据。
消融：关闭跨类交互、通用 CLIP 替换 RemoteCLIP、Semantic–Detail Fusion 替代融合、关闭四方向聚合。逐项定义操作及初始化，锁定其余条件。通用 CLIP 对照需匹配模型规模、微调模式与训练预算。

结果登记时把 run_id、模型 commit 和工作区散列、配置、数据清单、seed、checkpoint、选择划分、正式划分及原始指标路径绑定；标准 JSON 中未知字段写 null，导出 CSV 中留空，不填零或猜测数值。
