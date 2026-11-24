# ReCA-SAM3 论文工作区

实验章节以回退后的 `master`（实现标识 `reca-sam3-multiscale-windows`）为源码基准，并采用用户确认的最终实验设计：两层聚合、三尺度普通窗口、主模型共享尺度间权重、唯一 RemoteCLIP score embedding 指导、单方向 SDF。中间层特征不参与模型；当前源码仍提取和缓存这些未使用特征，后续需清理，效率测量在清理后进行。

## 实验章节与表格

正文使用中文，表格、表题、表注和图题使用英文。`sections/experiments.tex` 已引用以下六张表，所有未测量的本地成绩及开销均为 `--`：

| 表格源文件 | 内容 |
| --- | --- |
| `tables/dataset_protocol.tex` | 八数据集角色、类别数、展开提示数、背景与过滤阈值 |
| `tables/main_rskt8.tex` | Semantic-only、Semantic × Presence、Full 的本地评测及原有文献参考行 |
| `tables/loveda_per_class.tex` | 三个本地方法的 LoveDA 七类 IoU 与 mIoU |
| `tables/ablation_core.tex` | Full、严格提示自交互、无引导、无类内、单尺度、三尺度独立权重、无蒸馏 |
| `tables/ablation_freeze.tex` | RemoteCLIP 图像／文本 attention 微调的四种组合 |
| `tables/efficiency.tex` | 两个原始基线、Full、单尺度及独立权重的参数与推理开销 |

核心消融与微调表共享 Full，合计十个训练变体。两个自跑比较基线均使用原始 SAM 3 权重，不需要 DLRSD 训练。不新增移位窗口或中间层特征消融。`data/experiment_plan.csv` 保存变体 ID、优先级和实现状态；这些计划行不会自动创建训练配置。

## 后续填写数据

表格布局由 `data/table_spec.json` 与 `scripts/build_experiment_tables.py` 管理。完成评测后，将真实标准结果汇总至 `generated/data/results_long.csv`，按 `table_spec.json` 的选择器填写实际 eval_id、种子、checkpoint SHA-256 和协议，再生成表格。主模型在主表、消融表、微调表及逐类表必须复用相同评测和权重。参数与效率选择器只读取实际测量记录，不估算参数量或时间。

也可直接替换 `tables/*.tex` 中的 `--` 进行排版；重新运行生成器会覆盖手工修改，因此正式成绩应同步保存到真实结果记录和选择器。保留现有文献 CSV 与来源，不用本地未跑成绩替换文献原报告值。

```bash
python paper/scripts/test_build_experiment_tables.py
python paper/scripts/build_experiment_tables.py --draft
cd paper
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

`export_config.json` 同时预留标准结果导出的本地比较、逐类和六个按数据集拆分的消融表，辅助保存 mIoU/mAcc。论文实际引用的布局由上述六个 `tables/*.tex` 文件提供。

## 核对记录

- `notes/evaluation_protocol.md`：训练源、开发域、目标域、选权重和统计规则。
- `notes/experiment_config_snapshot.json`：本轮核对的有效主配置与八个数据集设置。
- `notes/source_snapshot.json`：基准 commit、源码与论文文件哈希及验证记录。
- `notes/manuscript_status.md`：实现状态、待跑实验与待修改内容。
- `notes/review.md`：本轮主张与证据、自审记录。

本轮仅重构实验章节与表格工作流，没有修改模型、训练配置或训练任务。方法章节与架构图仍保留旧版本内容，需要另行同步；本轮不将整篇论文认定为与当前实现完全一致。Zotero 导出的 `references.bib` 保持原样。
