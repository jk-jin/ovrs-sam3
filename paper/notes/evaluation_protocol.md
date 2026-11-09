# 实验协议登记（定稿前必须补齐）

## 当前代码的开发与 checkpoint 选择

训练入口：`configs/train/isaid_loveda_full.py`。
训练划分：iSAID `img_dir/train` 与 `ann_dir/train`，15 个配置类别，标签 255 忽略。
选择划分：LoveDA `img_dir/val` 与 `ann_dir/val`。
选择指标：`semantic.groups.loveda_no_background.miou`，类别编号 1–6，取最大值。
每 2000 步评测；`val_max_iters=2000` 是当前配置上限，不等于已核实样本数或完整验证覆盖率。TODO：保存实际评测图像清单和数量。

现有 `configs/test/loveda.py` 也指向 LoveDA val，因此并非自动切换到独立正式测试集。该评测入口不默认继承训练入口声明的六类 metric group；另存前景指标时须核对实际组配置或从逐类统计重算，不得假定键必然存在。

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
另存：6 类前景 mIoU，类别 1–6；背景 IoU，类别 0；全部逐类 IoU。
保存源：`semantic.miou`、`semantic.groups.loveda_no_background.miou`（须显式配置该组）、`semantic.iou_class_0`，以及 `semantic.iou_class_0` 至 `semantic.iou_class_6`。七类口径需保证七类并集均非零；当前评测器会排除零并集类别。若类别缺失，记录有效类别数并披露，不能悄悄称其为七类平均。
内部评测值为比例；论文和 results.csv 统一百分比，只转换一次。推荐保存混淆矩阵及并集/交集，便于复核；空白表示未运行。

数据代码先进行 reduce_zero_label，再为前向排除背景；评测标签保留背景。背景不作为文本前向类别，双门后的分数经 argmax，全部分数不超过评测阈值时回填背景。TODO：用真实标签样本核实源标签值域、无效值和映射，不仅检查配置。

## 比较组与消融约束

无额外遥感训练组：SAM 3、SegEarth-OV3，具体版本、权重及原协议 TODO。“无额外”仅指本轮遥感分割训练，不否定已有遥感预训练。
同 iSAID 训练组：简单 SAM 3 adapter、其他待确认对照、ReCA-SAM3。简单 adapter 结构及参数/预算 TODO；所有变体记录可训练范围与额外数据。
消融：关闭跨类交互、通用 CLIP 替换 RemoteCLIP、DIF 替代融合、DIF 关闭四方向聚合。逐项定义操作及初始化，锁定其余条件。通用 CLIP 对照需匹配模型规模、微调模式与训练预算。

结果登记时把 run_id、模型 commit 和工作区散列、配置、数据清单、seed、checkpoint、选择划分、正式划分及原始指标路径绑定；results.csv 中未知字段留空，不填零或猜测数值。
