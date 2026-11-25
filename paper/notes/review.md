# 当前论文检查记录

## 主张与证据

| 内容 | 核验依据与边界 |
| --- | --- |
| 主表文献值 | data/literature_results.csv 与 data/comparison_audit_20261124.json；引用来源，不代表本地复现 |
| 双协议 iSAID、单列 UAVid | 用户确认方案与 comparison_policy.md；未知 iSAID 值留在独立表注，数据行不跨列 |
| 本地权重与结果复用 | 生成器检查种子、checkpoint、协议及多表 Full 选择；实际未运行结果保持 -- |
| 当前模型与配置 | notes/experiment_config_snapshot.json、notes/source_snapshot.json；当前默认值不称实测最优值 |
| 聚合正文与公式 | models/aggregation_attention.py 与 models/feature_aggregation.py：两个完整块、语义仅引导 Q/K、条件特征提供 V、块内共享三尺度普通窗口与三个标量 Softmax 融合 |
| SDF 与推理正文 | models/semantic_detail_fusion.py、models/adapters/semantic_adapter.py 与 engine/evaluator.py：单方向双支路、全配置类别前向、直接类别索引和数据集独立阈值 |
| Zotero | 两个 bib 数据库不随表格清理改写；正文引用决定参考文献，集合操作见 zotero_sync.md |

## 验收标准

- 事实：区分已有实现、确定的实验设计与实际测量，不编造本地成绩或宣称引用模型已复现。
- 可比性：训练资源和来源单列；不混算不同类别协议、划分及后处理的提升幅度或平均值。
- 数值：所选原报告值保持不变；未知 iSAID 分数在表注按方法名只出现一次，全部数据行有两个独立协议格。
- 文档：comparison_policy.md 保存长期规则，evaluation_protocol.md 保存本地实验协议，manuscript_status.md 只保留真正未完成事项；删除已被取代的日期式问题清单与重复手抄主表。
- 排版：重新生成、编译并目检主表；检查 iSAID 两列位置、后续列对齐、表注和未定义引用。最终结果记录于 source_snapshot.json。

## 正文更新自审

- 事实与证据：摘要、引言、相关工作、方法、实验和结论统一到当前实现；不把固定语义引导误写为冻结、不把尺度标量误写为图像动态权重。
- 逻辑：从模板分数生成引导，到类间特征更新、多尺度空间增量融合，再到 SDF 与冻结解码器，保持一条条件特征更新路径。
- 数学：解释新增块索引、尺度集合、插值算子、窗口增量与融合标量；保留普通窗口相对位置偏置，删除移位屏蔽及背景索引恢复。
- 表达：中文概述与英文公式说明采用同一结构定义，不保留双状态更新及独立类内／类间块的旧表述。
- 完整性：正文与图注已更新，draw.io/PDF 图内结构仍待重绘；实验表格、本地占位和文献数值保持原样，不将整篇论文视作投稿定稿。

架构图和实际实验的剩余工作只在 manuscript_status.md 维护。
