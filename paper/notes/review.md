# 当前论文检查记录

## 主张与证据

| 内容 | 核验依据与边界 |
| --- | --- |
| 主表文献值 | data/literature_results.csv 与 data/comparison_audit_20261124.json；引用来源，不代表本地复现 |
| 双协议 iSAID、单列 UAVid | 用户确认方案与 comparison_policy.md；未知 iSAID 值留在独立表注，数据行不跨列 |
| 本地权重与结果复用 | 生成器检查种子、checkpoint、协议及多表 Full 选择；实际未运行结果保持 -- |
| 当前模型与配置 | notes/experiment_config_snapshot.json、notes/source_snapshot.json；当前默认值不称实测最优值 |
| Zotero | 两个 bib 数据库不随表格清理改写；正文引用决定参考文献，集合操作见 zotero_sync.md |

## 验收标准

- 事实：区分已有实现、确定的实验设计与实际测量，不编造本地成绩或宣称引用模型已复现。
- 可比性：训练资源和来源单列；不混算不同类别协议、划分及后处理的提升幅度或平均值。
- 数值：所选原报告值保持不变；未知 iSAID 分数在表注按方法名只出现一次，全部数据行有两个独立协议格。
- 文档：comparison_policy.md 保存长期规则，evaluation_protocol.md 保存本地实验协议，manuscript_status.md 只保留真正未完成事项；删除已被取代的日期式问题清单与重复手抄主表。
- 排版：重新生成、编译并目检主表；检查 iSAID 两列位置、后续列对齐、表注和未定义引用。最终结果记录于 source_snapshot.json。

方法正文和架构图的剩余同步工作只在 manuscript_status.md 维护，不将整篇论文视作已完成实验的投稿定稿。
