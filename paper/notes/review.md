# 本轮核对

范围：新完整模型的聚合块和单方向融合，及旧消融设计清理。

| 主张 | 源码证据 | 检查 |
| --- | --- | --- |
| 每块完整执行类间、普通、移位注意力 | models/aggregation_attention.py | CPU 执行顺序、逐步残差与梯度 |
| score embedding 固定且通过 Q/K 可训练 | models/feature_aggregation.py | 张量身份、RemoteCLIP/embedding 梯度、checkpoint 一致性 |
| SDF 每支路执行原方向卷积一次 | models/semantic_detail_fusion.py | 支路次数、残差等价与三路输入梯度 |
| 旧消融不再排队或组表 | configs/suites、paper/data | 配置、验证任务、组表回归 |

结构说明、训练设置、实验身份、结果溯源与论文表格已按源码核对。没有正式训练/成绩证据。架构图仍待重绘，不能认定论文图文完全一致；本轮不对旧图作新设计验收。文献数据未改写，不编造实验结论。
