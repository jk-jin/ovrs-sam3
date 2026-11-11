# 提纲、证据与自审

## 方法概述段落角色与反向提纲

1. 开篇定位：对冻结 SAM 3 的遥感适配，串联 SAM 3、RemoteCLIP、特征聚合、Semantic–Detail Fusion。
2. 设计：局部图文相似度转分数嵌入，SAM 3 原始类别条件特征初始化另一流。
3. 设计：类间提示轴交互与类内空间关系的区别，以及双流更新和固定指导。
4. 边界：同义提示展开、共享标签、推理合并与全提示交互。
5. 设计：Semantic–Detail Fusion 三类输入、语义/细节支路、四方向当前实现及冻结解码路径。
6. 限定：冻结与可训练部分、梯度边界；机制本身不构成性能证据。

每段围绕一个问题，先解释输入与作用，再交代实现限制。后续小节分别展开任务定义、先验、交互、融合、训练与推理。

## 主张—证据映射

| 主张 | 证据 | 状态 |
| --- | --- | --- |
| 当前逐类路径在特征聚合前无显式跨提示交互 | sam3_image.py 的提示 pair 编码与分块缓存；aggregation_attention.py 的交互轴 | 实现支持，限制在本文路径；SAM 3 文献仍待核实 |
| RemoteCLIP 局部相似度参与条件特征更新 | remoteclip_guidance.py、feature_aggregation.py 与注意力投影 | 实现支持；是否优于通用 CLIP 待验证 |
| 类间沿展开提示，类内沿提示内部空间 | CrossClassAttention reshape、WindowSpatialAttention、CrossClassAggregationBlock/IntraClassAggregationBlock | 实现支持 |
| Semantic–Detail Fusion 连接三类表征并复用冻结 SAM 3 解码器 | semantic_detail_fusion.py、sam3_image.py、model_builder.py | 当前工作区支持；最终变体 TODO |
| 减少道路/裸地等类别误响应 | 尚无已绑定当前实现的正式实验与混淆分析 | 待验证，正文仅作动机和研究目标 |
| 改善边界 | 无已核实边界指标与固定容差 | 待验证 |
| 增强跨数据集迁移 | 无独立正式目标划分结果，当前 LoveDA val 用于选择 | 待验证 |
| 优于 SAM 3、SegEarth-OV3 或简单 adapter | 无已核实同协议结果 | 不作结论 |

本轮未将旧实验或不可追溯数字移入论文，也未生成训练结果。摘要与结论保持占位。

## 五维审阅

| 维度 | 问题与回答 | 状态及行动 |
| --- | --- | --- |
| 贡献 | 是否证明类别交互提供额外价值？尚未 | 需新实验：简单 adapter 与关闭交互 |
| 写作清晰度 | 交互轴、同义提示和梯度边界是否明确？概述已交代；公式和完整符号尚缺 | 需续写：任务定义与逐模块推导 |
| 实验强度 | 有无稳定收益和失败案例？未有可报告结果 | 需新实验及结果追踪，不能写效果结论 |
| 评价完整性 | 选择与正式评测是否独立？当前尚未 | 需确认划分、指标有效类别数、对照预算与文献 |
| 方法合理性 | 复杂度是否值得、对类别提示是否稳健？未知 | 需成本测量、Semantic–Detail Fusion/提示敏感性消融与目标域协议 |

## 下一轮需要确认

- ReCA-SAM3 名称的正式英文展开、标题及作者单位。
- 特征聚合最终块数、头数、窗口/移位、固定图像指导与无 FFN 设计。
- 四方向单尺度 Semantic–Detail Fusion 是否定稿，以及替代融合对照的定义。
- RemoteCLIP 权重、模板、图文部分微调范围，是否匹配通用 CLIP 对照。
- 主 BCE、蒸馏、忽略区域抑制、Dice 的最终组合、权重和边界外环。
- 推理双门阈值、背景策略、提示词/TTA 设置及其选择数据。
- checkpoint 选择数据与独立正式评测数据；是否可用 LoveDA 独立测试，或需另定协议。
- 简单 adapter、SegEarth-OV3 与同 iSAID 训练对照的版本、预算、可复现入口。
- 边界、误响应与迁移主张的预定指标及可追溯实验产物。

## 2026-11-11：实验章节整体重建

旧 LoveDA 单数据集正文、表格与定性占位已替换，旧标签 tab:main、tab:ablation、fig:qualitative 不再作为实验入口。保留用户已修改的 related_work.tex 与 references.bib，不清理或重写其他章节。方法仅做与当前源命名直接相关的 Refiner→特征聚合、DIF→SDF 术语替换；main.tex 结论只引一次，未改结论内容。

新提纲与段落角色：

1. 数据与协议（范围、限制）：固定RSKT八集合，iSAID源域，LoveDA当前选择泄漏，指标与固定平均集合。
2. 实现（复现依据）：当前配置及冻结/梯度范围，全部标当前起点而非已执行设置。
3. 主比较（证据、限制）：方法与实际数值来源分开，文献参考与自测占位分区，未知口径不排名。
4. 核心消融（因果问题）：T00—T09完整定义，一次共享Full，结构/损失重新训练。
5. 后处理（控制条件）：同权重P00/P10/P01/P11，不增加训练，None语义与reference解码成本。
6. 定性与成本（证据规划）：真实输出、统一硬件与计时条件，保留成功/失败案例。
7. 进一步分析（优先级）：Full/T02上下文提示诊断优先，其他分析紧凑规划，不声称已实现。

### 本轮主张—证据映射

| 主张 | 依据 | 状态 |
| --- | --- | --- |
| 采用RSKT-Seg八数据集集合 | 本地AAAI正式PDF p.3、p.5 Table1 | supported；同协议仍needs evidence，附录缺失 |
| 当前为iSAID适配、LoveDA val选择 | main入口及protocol/schedule/dataset合并配置 | supported as current code；独立正式目标评测needs revision |
| SAM3冻结不等于全系统冻结 | model_builder.py、models/reca_sam3.py、feature_aggregation.py、semantic_detail_fusion.py | supported；RemoteCLIP Q/V/位置参数训练 |
| 主表文献值确有来源 | 5篇本地PDF、渲染表格、61条逐项记录 | supported as reported numbers；0 matched，不能解释为公平排名 |
| LoveDA只显示七类含背景 | SegEarth-OV附录+本地标签配置；其他不明确来源阻止入列 | supported；未知来源不估算或补背景 |
| 聚合、初始化、SDF、监督有益 | T01—T09对应实验问题 | needs new experiment；没有提升结论 |
| 推理门改善质量或速度 | 同权重四组计划 | needs new experiment；成本尚缺 |

### 五维自审与修订结果

- 贡献：尚未证明聚合比简单适配有额外价值。已定义T01/T02，保留性能主张为TODO；需新实验。
- 清晰度：交互轴、T05仍保留CLIP指导、SDF替代结构及别名屏蔽边界已写清；T01/T06等未实现变体不假称可运行。
- 实验强度：全部本文/消融数字为--，单seed无伪标准差；尚无自身统计证据，需新实验。
- 评价完整性：八集合、固定均值、文献/原生身份、版本和再报告性质分开；缺RSKT附录、VDD全文、ISPRS原始条目及实际样本清单，需补证据。
- 方法合理性：类别上下文诊断与成本仍待实现/测量；不把源域成绩当未见目标，不把ignore当完整负例。

### 组表与工程检查

`build_experiment_tables.py` 不导入模型、不下载论文、不训练，只透视显式选定文献记录或native fraction长表。重复result_id/eval_id、非法数、选中记录身份不符、LoveDA非七类、同seed多checkpoint、未完整评测、未预定seed、权重哈希不符均失败；缺少计划项仅在draft生成--。同一Full在主表/T00/双门间按seed共享checkpoint。

5组边界测试覆盖真实0与空、非法数、单位重复转换、固定均值缺项、明确来源选择、七类/背景、同seed多checkpoint、完整性和哈希。测试数据只在内存及临时目录，不进入结果登记。严格模式在缺少必需结果及协议锁定时预期失败，不通过放宽正式导出器校验绕过。

`notes/results.csv` 原来只有4条空计划行、无真实成绩；本轮保留行，删除废弃的前景汇总字段。该文件不再是数字来源：文献以data/literature_results.csv为准，自测以标准JSON/长表为准。原先 source_snapshot.json 保留为历史快照，不冒充本轮运行快照。

### 下一轮所需证据

先锁定8项评测版本/标签/波段/裁块、原始场景独立性及不使用目标标签的选择策略；取得缺失PDF/附录后才能将参考行提升为matched。然后提供T00/T01/T02八项、T03—T09至少三个核心目标、同T00权重P00/P10/P01/P11的标准完整结果及计时；提供统一测量条件下SAM3/adapter/Full/T07成本与固定样本预测。正式seed清单应在运行前登记。细节、生成与编译结果见README本轮记录。


## 2026-11-11 语言与排版审阅

删除可见正文中的执行性注释与TODO，将现状集中到manuscript_status.md；保留必要协议限制。图表英文策略同步到生成脚本，跨栏表精简为主表，去除FloatBarrier。无性能结论被新增；原始文献CSV未改。验证结果及待办见同一状态文档。

用户明确授权GitHub发布：提交已完成验证的论文语言、英文表格、状态文档和布局修改；仅paper/文本源码入库，不加入编译输出或本机依赖。
