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


## 表格与参考文献排版修复提纲（2026-11-18）

范围只含版式和生成器同步，不改实验数值、方法公式、引用条目或训练代码。先使用IEEEtran/CJK原模板复现表体/表注宽度不齐及文末浮动体位置；测量表格实际宽度，统一列宽与表注宽度并同步生成器；为参考文献建立一次文末浮动边界，避免残留表图穿入参考文献。完整编译、引用和PDF页图检查完成后记录实际结果。


排版验证结论：旧稿第7页开始参考文献，但表IV仍在第8页；现在四表全部在第7页，结论/参考文献在第8页。tabularx与同宽表注同步生成，长表头正常换行；单次文末FloatBarrier隔离所有未排图表，raggedbottom消除最后一页的过度竖直拉伸，浮动页内容顶部对齐。8项组表测试通过、数据行/两份BibTeX/方法及实验正文逐字不变。完整latexmk构建成功，已检查第7/8页PNG与全稿缩略图，无Overfull、未定义引用或缺字。保留IEEEtran双栏和字体尺寸，不改实验结论。复现与检查结果已更新当前状态及source_snapshot；本地依赖未纳入提交。


## 可编辑五图与连续分页修订提纲（2026-11-18）

将确认的浅色分区、虚线边框、特征堆叠和模块颜色落实为五份原生draw.io源文件及矢量导出，逐对象可编辑；输入影像/GT/预测使用同一可替换占位图。按代码纠正预览中原始72×72跳连、先上采样再分支、双流Q/K/V与训练蒸馏方向。正文仅替换图位/图题和浮动声明，不改结果或24个公式。检查并解决浮动体集中排出造成的正文末尾空白、参考文献提前换栏，采用自然连续双栏排版；通过全稿编译和页面检查确定尺寸与位置。


五图完成记录：五份原生draw.io源、五份SVG与五份矢量PDF已加入论文；原生对象/图片URI/端点检查通过，文字附属于模块、堆叠成组。原始72×72跳连、FPN跳连、双流目标V与SDF投影对照代码核对。定性图全部为占位，不写预测结论。分页通过全局raggedbottom、自然浮动选项与定性图提前声明修复，唯一屏障位于参考文献前。全稿9页，第6/7页无大段空白，文献自然从当前栏续至下一页左栏，不进行末页均分。数值、24个公式、BibTeX及训练实现不变；验证明细见当前状态与快照。


## 五图配色核对（2026-11-19）

仅调整填充层级：15个大区域使用浅色纵向渐变，内部模块使用较饱和纯色，保留语义色系并区分定性图列标题。五份draw.io剔除填充/渐变属性后与基线结构完全一致；SVG除新增渐变定义与填充外一致，导出PDF的页面尺寸和提取文字不变。五图已逐图渲染检查，全稿重新编译为9页并检查缩略页与方法图页；无Overfull、缺字或未定义引用。图题、LaTeX、实验数据、公式及训练实现均未修改。
