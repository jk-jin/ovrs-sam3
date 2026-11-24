# 2026-11-24 实验章节核对

范围：仅 paper/ 的实验章节、五张表、实验计划、导出选择器和协议快照。没有实现模型消融或修改训练配置。

## 主张—证据映射

| 文中设置／设计 | 证据 | 核对结论 |
| --- | --- | --- |
| Full 两层，36/24/12 普通窗口，共享权重与可学习Softmax融合 | models/aggregation_attention.py、models/feature_aggregation.py、configs/_base_/model/reca_sam3.py | 当前实现；实验正文已同步 |
| RemoteCLIP仅score embedding指导，无中间层指导 | feature_aggregation.py；用户确认的最终设计 | 中间层未参与聚合；提取／缓存清理仍待工程修改 |
| 全配置类别前向、同义提示合并、背景过滤 | data/dataset.py、data/class_prompts.py、semantic_adapter.py、engine/evaluator.py | 类别／提示数及阈值快照与当前配置核对 |
| 学习率1e-4、batch4、10k步、固定蒸馏0.5 | 有效配置，notes/experiment_config_snapshot.json | 当前默认设置，不表述为实测最优值 |
| 严格提示对角、无引导、无类内、单尺度 | 用户确认方案；data/experiment_plan.csv | 实验设计，尚需工程实现，不声称已经运行 |
| 三尺度独立权重、四种图文微调组合、无蒸馏 | 当前配置能力与用户确认方案 | 参数开关已支持，正式入口和实验仍待完成 |
| 仅自跑两个原始基线 | configs/baselines、models/experiment_variants.py | Semantic-only 与 Semantic × Presence；后者每提示乘presence一次，不融合实例掩码 |
| LoveDA逐类表和多表共享Full | 显式选择器与renderer校验 | 绑定同一权重／种子／评测；禁止逐目标挑分 |
| 文献原报告值 | literature_results.csv、既有comparison audit | 历史记录保留；新增149条来源记录，所选156个单元格均有独立出处；补充10条方法引用 |

## 五维自审

1. **事实与实现**：区分当前代码、用户确认的最终设计及待实现变体；不把十个变体写成已经完成的实验。
2. **实验公平性**：DLRSD单源训练；LoveDA为开发域；相同预算、提示、阈值和种子；每种子一份权重用于全部目标。
3. **数值与溯源**：本地成绩、参数与时间全部为 --；保留文献来源限制；单种子不构造标准差。
4. **表达与结构**：中文正文、英文表格；数据协议、主比较、逐类、核心消融、微调组合、效率依次组织；不新增公式。
5. **排版与可复现性**：五张表由同一规范生成并在实验章节引用；组表回归检查包括变体总数、Full复用、七类IoU、多种子统计和严格模式缺失拒绝。PDF构建与页面检查结果记录在source_snapshot。

## 范围外遗留

方法章节仍有四块／移位窗口及背景排除等旧描述，架构图也需另行同步。本轮未重绘图或改写方法公式。旧定性占位图不再插入实验章节，待真实预测图准备后再恢复。整篇论文尚不是已完成实验的投稿定稿。

## 本次比较与引用审阅

- 证据：最新references.bib 26条逐字节保留；重复ConceptBank DOI未双引。来源审计记录数值PDF指纹／页码，或明确出版方HTML表／摘要证据。
- 公平性：主表仅两组；本地N与文献来源分开标注，不做跨来源最优加粗或平均；DLRSD切换不伪造LoveDA与UAVid协议对齐。
- 表达：中文正文，英文表题／表注；GFMR/ConceptBank/PRS不再比较或引用；GLoP方法引用不替代ROSE转引来源。
- 排版：五张表编译成功，主比较页已目检，额外条件参考表已删除；完整结果和问题清单可直接审查。
- 范围：只修改paper/，不改模型／配置或用户Zotero导出库；原数据协议表删除，VLex／ConceptSeg成绩排除。

本轮依用户确认实施论文展示：iSAID双协议子列，同一权重的双评测选择键；UAVid单列7／8／u来源标记；未知iSAID值跨列一次显示。文献数值与Zotero导出库保持原样，外部版本排查不再阻塞引用。iSAID BG评测工程仍未修改或运行。

新增主张—证据：双协议与保留原值来自本轮用户确认；生成器校验同一权重及不同协议ID，回归检查覆盖双列位置、未知值不重复、UAVid标注与缺失结果。Zotero说明按官方集合文档与Better BibTeX自动导出文档核对，引用删除与文库删除分别处理。PDF编译和主表布局检查结果见source_snapshot.json。
