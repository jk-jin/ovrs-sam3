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
| 当前结构图 | figures/src/、figure_manifest.json 与对应 PDF：两个完整块、共享三尺度普通窗口、仅条件特征接受残差；组合图接入方法，定性占位不接入正文 |
| 动机图与引言 | motivation.drawio/PDF 的四个适配位置子图；ROSE 原文 https://arxiv.org/html/2611.28112v1（III-A 至 III-D，含竞争类别保护），ProC 官方说明 https://github.com/YanghuiSong/ProC-SAM3；区分提示构建与响应选择两种离线策略，保留参数冻结／监督适配差异，不声称已有方法没有类别关系 |
| overview 视觉设计 | 用户确认的白底三板块预览：原生梯形编码器、图标、特征叠片与五处占位；灰色训练栏；内部卡片与输出占位留有边距，类间特征沿深度分层，Block 2 顶部、右侧及底部轮廓独立可见。图像／文本输入、Q/K 引导、V 来源与代码核对，同名跨板块端口复用相同张量 |
| 统一绘图规范 | figures/STYLE_GUIDE.md 集中规定视觉风格与验收；以 7d43658 的 overview 为参考，paper/AGENTS.md 与两级 README 提供入口。动机图与 overview 已迁移；聚合图和定性模板仍保留原图稿 |
| Zotero | 两个 bib 数据库不随表格清理改写；正文引用决定参考文献，集合操作见 zotero_sync.md |

## 验收标准

- 事实：区分已有实现、确定的实验设计与实际测量，不编造本地成绩或宣称引用模型已复现。
- 可比性：训练资源和来源单列；不混算不同类别协议、划分及后处理的提升幅度或平均值。
- 数值：所选原报告值保持不变；未知 iSAID 分数在表注按方法名只出现一次，全部数据行有两个独立协议格。
- 文档：comparison_policy.md 保存长期规则，evaluation_protocol.md 保存本地实验协议，manuscript_status.md 只保留真正未完成事项；删除已被取代的日期式问题清单与重复手抄主表。
- 排版：重新生成、编译并目检主表；检查 iSAID 两列位置、后续列对齐、表注和未定义引用。最终结果记录于 source_snapshot.json。

## 正文更新自审

- 事实与证据：摘要、引言、相关工作、方法、实验和结论统一到当前实现；不把固定语义引导误写为冻结、不把尺度标量误写为图像动态权重。
- 逻辑：动机图从匹配、响应、提示适配递进到条件特征更新；承认 ROSE 已考虑类别证据。方法从模板分数生成引导，到类间特征更新、多尺度空间增量融合，再到 SDF 与冻结解码器，保持一条条件特征更新路径。
- 数学：解释新增块索引、尺度集合、插值算子、窗口增量与融合标量；保留普通窗口相对位置偏置，删除移位屏蔽及背景索引恢复。
- 表达：方法的机制、流程与段落标题采用中文，公式及符号说明保留英文；22 个编号公式保持原样，不保留双状态更新及独立类内／类间块的旧表述。
- 完整性：独立 SDF 图及其导出项已删除，模型融合模块与正文公式保留；动机、总体框架和聚合／多尺度组合图均已重绘并接入正文，原生 draw.io 与矢量 PDF 同步。定性模板列名已对齐两个实际基线，待真实结果替换占位后再接入正文。实验表格与文献数值保持原样，不将整篇论文视作投稿定稿。

架构图和实际实验的剩余工作只在 manuscript_status.md 维护。
