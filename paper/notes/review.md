# 当前修订审阅记录（2026-11-18）

基线：master@b953cc98e2f96012cc654a9c9f39f1884ac65f8f。当前计划与待补证据的唯一入口为[manuscript_status.md](manuscript_status.md)；历史修订过程通过Git历史查阅，不在本文件累计互相覆盖的旧设计。

## 修改提纲（编辑正文前记录）

1. 摘要：问题、相似度到类别指导特征、双流聚合、SDF与冻结参考约束；暂不写领先结论或详细实验设置。
2. 引言：遥感语义/空间挑战、CLIP代价聚合路线、SAM3逐类迁移路线、解码前特征交互缺口、本文方法及两项方法贡献。承认已有空间/类别聚合和遥感提示适配。
3. 相关工作：遥感图文适配、SAM3遥感迁移、代价与结构引导聚合。补GeoSeg-OV方法引用，不增加比较行；自动导出的references.bib不覆盖。
4. 方法：保留24个公式及其英文说明，补清similarity-derived class-specific guidance与独立SAM3特征流的作用；分别定义语义/特征更新中的Q/K与V来源。
5. 图片：引言动机、方法总体框架/双流/SDF，以及定性比较均保留英文占位；机制结果图不加入。定性占位从main输入，不改实验章节原有文字或结果表。
6. 结论：总结实现和设计目标，不写未经实验验证的收益。
7. 文档：删除旧iSAID空结果模板与被后续证据覆盖的旧文献汇总；保留CSV/JSON原始证据和仍被引用的补核文档；将进度、待实验主张、图计划与验证结论集中到当前状态文档。

## 主张—证据与验证

| 陈述 | 源码/原文 | 当前结论 |
| --- | --- | --- |
| 相似度投影并与dense图像融合，生成每提示指导特征 | models/remoteclip_guidance.py，template_projection/remoteclip_fusion/spatial_enhancement | 已核对；与24个原有公式一致 |
| 两状态共同构造Q/K，V来自更新目标流；先语义后特征 | models/aggregation_attention.py、models/feature_aggregation.py | 已核对；类内额外中间层指导与类别指导状态分别说明 |
| 最终更新SAM3特征经SDF接冻结解码器 | semantic_detail_fusion.py、sam3_image.py | 已核对；不从结构推断边界/精度收益 |
| RSKT/CAT已包含代价嵌入和空间/类别聚合 | [RSKT v3](https://arxiv.org/html/2509.12040v3)、[聚合代码](https://github.com/LiBingyu01/RSKT-Seg/blob/main/RSKT_Seg/modeling/transformer/EfficientAggregator/model_aggregator.py)、CAT-Seg CVPR2024 | 区别限定于交互表征、双流更新及冻结SAM3解码接口 |
| GSNet已有细节恢复和类别聚合 | [GSNet源码](https://github.com/yecy749/GSNet/blob/main/gs_net/modeling/transformer/FusionAgg.py)、其RIPD与原论文 | 不笼统称CLIP路线无边界/类别建模 |
| SAM3遥感免训练方法已有提示/输出适配 | [OV3代码](https://github.com/earth-insights/SegEarth-OV-3/blob/main/segearthov3_segmentor.py)、[ProC代码](https://github.com/YanghuiSong/ProC-SAM3/blob/main/sam3_segmentor_cached.py)及原论文 | 不称完全没有遥感设计；本文关注解码前特征交互 |
| GeoSeg-OV以冻结VFM结构指导代价聚合/解码 | [arXiv v1](https://arxiv.org/abs/2608.10426v1)、[官方实现](https://github.com/zzaiyan/GeoSeg-OV/blob/main/cat_seg/modeling/transformer/model.py) | 仅补背景引用；元数据来自arXiv，未增加成绩或实验行 |

五维自审：贡献聚焦已实现的双流指导和SDF适配，不把普通蒸馏独立包装；写作以更新对象与解码接口区分相近工作；实验强度仍待正式运行，具体主张映射集中到状态文档；数据角色与文献参考限制保持原样；可复现性由固定代码基线、未改实验文件的字节核对、20个唯一引用键及当前快照支持。

实际检查：8项组表测试通过；24个公式环境逐字与基线一致；五个图位、输入文件、交叉引用、引用键、方法英文说明、Markdown链接及git diff空白检查通过。实验章节、原BibTeX、全部data/tables及export_config共13个受保护文件逐字节一致。ASCII正文/英文数学及五图位的249pt双栏临时XeLaTeX检查生成6页XDV，第二轮无Overfull或未定义引用警告。完整latexmk仍因环境缺少xelatex.fmt及IEEEtran/CJK/字体依赖未成功，本轮没有新的IEEEtran中文PDF视觉验收，不把临时XDV当完整排版验证。
