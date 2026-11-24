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
- `dataset_protocols` 是当前配置的人工核对快照，配置改变后须同步核对类别、提示数量与阈值。

所有训练变体及两个仅评测基线的 ID 见 `paper/data/experiment_plan.csv`。当前十个训练变体中，严格提示对角、无引导、无类内、单尺度仍需工程实现；CSV 不自动生成这些模型。三尺度独立权重、图文冻结组合及关闭蒸馏已有对应配置能力，仍需创建并验证正式实验入口。

不带 `--draft` 时要求全部必需本地成绩、参数测量、效率和协议锁齐全。draft 只为缺失值提供占位，非法数值、身份冲突或协议冲突仍报错。单种子只显示数值，多种子显示均值与样本标准差。直接修改生成的 TeX 可用于排版草稿，但重新生成前应先将正式数值回写结果记录与选择器。

## 文献证据

`evidence_type`区分PDF、出版方HTML表格和出版方摘要。PDF必须有文件指纹、页码、表号；HTML表格必须有出版方URL和表号；摘要只能保留未解决协议，且不能附伪造PDF字段。`method_citekey`和`source_citekey`分别标识方法原文及实际取数来源。
