# 2026-11-24 实验章节核对

范围：仅 paper/ 的实验章节、六张表、实验计划、导出选择器和协议快照。没有实现模型消融或修改训练配置。

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
| 文献原报告值 | literature_results.csv、既有comparison audit | 文件与原表参考值保持不变，无新增数值或引用 |

## 五维自审

1. **事实与实现**：区分当前代码、用户确认的最终设计及待实现变体；不把十个变体写成已经完成的实验。
2. **实验公平性**：DLRSD单源训练；LoveDA为开发域；相同预算、提示、阈值和种子；每种子一份权重用于全部目标。
3. **数值与溯源**：本地成绩、参数与时间全部为 --；保留文献来源限制；单种子不构造标准差。
4. **表达与结构**：中文正文、英文表格；数据协议、主比较、逐类、核心消融、微调组合、效率依次组织；不新增公式。
5. **排版与可复现性**：六张表由同一规范生成并在实验章节引用；组表回归检查包括变体总数、Full复用、七类IoU、多种子统计和严格模式缺失拒绝。PDF构建与页面检查结果记录在source_snapshot。

## 范围外遗留

方法章节仍有四块／移位窗口及背景排除等旧描述，架构图也需另行同步。本轮未重绘图或改写方法公式。旧定性占位图不再插入实验章节，待真实预测图准备后再恢复。整篇论文尚不是已完成实验的投稿定稿。
