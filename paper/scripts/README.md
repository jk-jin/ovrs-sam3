# 论文表格生成与填写

从仓库根目录执行：

```bash
python paper/scripts/test_build_experiment_tables.py
python paper/scripts/build_experiment_tables.py --draft
```

输入为 `paper/data/table_spec.json`、文献 CSV 和真实标准结果汇总 `paper/generated/data/results_long.csv`。生成两组八域主比较、LoveDA逐类、核心消融、编码器微调和效率五张表。原数据协议表不再生成。所有未测量的本地单元格为 `--`；文献选择与来源以comparison_audit_20261124.json为准；历史数值记录保留。

## 选择真实结果

- 主表在每个本地行的 `native_selection` 中按数据集填实际 eval_id、expected_seeds、checkpoint_sha256_by_seed、eval_protocol_id 和 eval_split。
- 核心消融与编码器微调表共用 `ablation_native_selection[variant_id][dataset]`。Full 的选择器必须与主表一致，不允许在不同表格或目标域重新挑权重。
- LoveDA 逐类表直接复用主表的 LoveDA 选择器；七类 IoU 与 mIoU 使用同一组种子和 checkpoint。
- `core_parameter_selection`、`freeze_parameter_selection` 和 `efficiency_selection` 绑定实际标准结果 JSON 中已有的测量。选择器包括 eval_id、checkpoint_sha256、json_path 和 display_scale；不臆造效率 schema，不自动填写零值。
- `Trainable (M)` 填每个变体实际可更新的参数元素数，单位为百万。计入聚合、score embedding、SDF 和未冻结的 RemoteCLIP 参数，排除冻结 SAM 3／教师；共享参数去重。当前 RemoteCLIP attention 微调只更新 Q/V 和位置参数，融合 QKV 张量的 K 部分被梯度屏蔽，统计时应排除该部分，不能简单把整个 `requires_grad=True` 张量都计入。此数值不是模型总参数量或显存占用。
- iSAID 本地选择键为 `iSAID-15-IgnBG` 与 `iSAID-16-BG`，每方法／种子同一权重、不同协议 ID；真实结果的 eval_dataset 仍为 isaid。
- `dataset_protocols` 是当前配置的人工核对快照，配置改变后须同步核对类别、提示数量与阈值。

所有训练变体及两个仅评测基线的 ID 见 `paper/data/experiment_plan.csv`。十个训练变体及对应模型路径均已实现；实际命令见 docs/experiment_suites.md。计划 CSV 不代表实验已经运行，只有真实标准结果可填数。

不带 `--draft` 时要求全部必需本地成绩、参数测量、效率和协议锁齐全。draft 只为缺失值提供占位，非法数值、身份冲突或协议冲突仍报错。单种子只显示数值，多种子显示均值与样本标准差。直接修改生成的 TeX 可用于排版草稿，但重新生成前应先将正式数值回写结果记录与选择器。

## 文献证据

`evidence_type`区分PDF、出版方HTML表格和出版方摘要。PDF必须有文件指纹、页码、表号；HTML表格必须有出版方URL和表号；摘要只能保留未解决协议，且不能附伪造PDF字段。`method_citekey`和`source_citekey`分别标识方法原文及实际取数来源。

## 排版约束

方法原文引用保留在行名后，实际转引来源在主表短注中分组引用；不生成来源字母、UAVid 协议上标或未解决标签标记。其余表注仅保留必要定义和单位，避免重复实验设置。核心消融和图文微调使用单栏 table，主对比与逐类表保留跨栏 table*。

主表每个数据行固定十一格，iSAID 必须有两个独立格。协议未知时两格为 --，由生成器将原报告值按方法名在表下注释仅列一次；数据行不能使用跨列居中。调整后重新生成并目检 PDF；详见[比较规则](../notes/comparison_policy.md)。已完成事项不在本文件累积，实际待办见[当前状态](../notes/manuscript_status.md)。

实验章节末尾的 `FloatBarrier` 用于阻止实验表进入结论和参考文献；跨栏主表在实验文件开头提前声明以减轻章末积压。不要将“减少留白”解释为删除全部章节边界。
