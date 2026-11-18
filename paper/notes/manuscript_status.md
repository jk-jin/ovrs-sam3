# 当前论文状态与后续工作

更新：2026-11-18。源码基线：master@b953cc98e2f96012cc654a9c9f39f1884ac65f8f。本文件是当前待办入口，历史设计和修订过程通过Git历史查阅，不以旧笔记覆盖当前代码。

## 1. 已写入正文的内容与界限

| 章节 | 当前内容 | 等待的证据/动作 |
| --- | --- | --- |
| 摘要 | 任务、相似度到类别指导特征、双流聚合、SDF及冻结参考约束 | 实验完成后增加最主要结果及有证据支持的结论；不罗列步数/学习率 |
| 引言 | CLIP代价聚合与SAM3逐类迁移两条路线、解码前特征交互缺口、两项方法贡献 | 第三项实证贡献暂不写入正文；不预设所有CLIP方法边界较差或SAM3处处更强 |
| 相关工作 | 领域图文适配、SAM3提示/输出适配、空间/类别/结构引导聚合 | 新增GeoSeg-OV背景引用；不增加实验行；保持对已有工作能力的准确描述 |
| 方法 | 24个公式及完整英文解释，补类别指导状态与独立SAM3特征流的作用 | 四个概念/结构图正式绘制；当前公式未经本轮改写 |
| 实验 | 原有章节、四表及全部占位数值保持原字节 | 完成真实训练/独立验证、协议核对、效率测量及定性预测后再改 |
| 结论 | 总结实现与适配目标 | 根据最终结果补效果、适用范围与实测限制 |

SAM3编码和解码参数冻结；聚合和SDF训练，RemoteCLIP图文分支只更新attention相关参数及位置参数，不是全量微调。软掩码蒸馏仅用于存在提示的有效GT像素，固定权重0.5；主BCE逐图按存在/缺席提示分组平均，默认0.8/0.2，再在batch上平均。没有边界环、蒸馏调度或Full推理时的教师/实例融合。

## 2. 机制表述与代码映射

1. `models/remoteclip_guidance.py`：64模板逐一保留为相似度通道；分数经1×1投影，与归一化dense RemoteCLIP特征融合，再将两路归一化特征拼接并经过两个3×3卷积。输出每提示256×36×36的`clip_guidance_36`，初始化`semantic_state_36`。这不是仅把标量相似度当注意力偏置。
2. `models/aggregation_attention.py`：每次更新以归一化语义/特征状态共同构造Q/K，V及残差目标来自被更新的那条流。跨提示沿同位置的提示轴交互，含同类别别名；类内采用普通/移位窗口。完整块先更新语义再更新特征。
3. `models/feature_aggregation.py`：类间/类内四块交替；RemoteCLIP中间层引导是另一路类内空间指导，不能与相似度产生的类别相关语义状态混为一谈。最终输出的是归一化后的SAM3特征状态。
4. `models/semantic_detail_fusion.py`：语义/细节两支路分别连接原始条件特征/FPN，四方向共享卷积后逆向对齐平均并残差，分支各自输出投影后求和，再最终投影，送入冻结分割路径。
5. 与CAT-Seg/RSKT-Seg的区别：相似度先转成类别指导特征，与独立SAM3特征流共同演化；最终复用SAM3解码器。不能声称其他方法没有相似度嵌入或类别交互，也不把结构差异自动当性能证据。

## 3. 待实验支持的主张

以下不是已完成结论，不据占位表格写“显著提升”“领先”或“保持边界”。不因本表新增消融。

| 待验证问题 | 已有实验/材料 | 能支持什么，以及不能支持什么 |
| --- | --- | --- |
| 相对原始SAM3的适配收益 | Semantic-only、Semantic×Presence、adapter、Full | 统一协议下报告差异；文献原报告区不作同条件排名 |
| 跨提示信息交换是否有益 | Diagonal cross-prompt attention | 检验非对角提示交互，包含被同时屏蔽的同类别别名 |
| 动态语义更新是否有益 | Encoder feature updates only | 检验语义更新及其专用RemoteCLIP中间层指导的联合贡献，不能单独归因某一个操作 |
| 提示内空间建模是否有益 | w/o intra-class aggregation | 检验类内更新路径的整体作用 |
| 初始相似度指导是否有益 | w/o initial semantic prior | 仅去掉初始化路径，仍保留RemoteCLIP中间层指导；不能称完全无RemoteCLIP，也不能证明RemoteCLIP优于通用CLIP |
| SDF与多方向处理是否有益 | Simple fusion、Single-orientation SDF | 分别检验融合结构/方向处理；不声称整模型任意角度旋转不变 |
| 参考蒸馏是否有益 | w/o distillation | mIoU差异支持整体效果；若无边界专门证据，不得据此声称已保留或改善边界 |
| RemoteCLIP适配范围的影响 | 图文四种冻结/attention微调组合 | 同时报告准确率与真实可训练参数量；不称全编码器微调 |
| 开销与精度取舍 | 既定效率表五种设置 | 统一GPU/精度/输入/batch/提示/chunk/缓存/预热/同步，测真实时间与峰值显存，不估算 |
| 相似类别、局部结构和失败模式 | 同样本定性对比、GT、逐类IoU | 根据真实预测描述；草地/耕地仅为任务示例，未证实前不写具体模型已解决该混淆 |

第三项实证贡献、摘要结果句、引言末结果概述、实验结果分析及结论效果句，须从最终锁定结果提炼。当前两项方法贡献分别为双流类别指导聚合，以及对接冻结解码器的SDF适配方案；普通软掩码BCE不包装为独立新蒸馏算法。

## 4. 实验完成后依次补什么

1. **锁定运行身份**：提交版本、最终config.json、GPU/精度、样本清单与预处理版本、预定seed集合、每项run及checkpoint路径/sha256、模型选择依据。默认42不是已运行或已锁定的多seed集合。
2. **按既定角色选权重**：当前DLRSD训练，LoveDA val用于开发选择；它不是独立目标域。每变体/seed选一份权重跨数据集复用。若以后改为源域选择，需另行确定并重做对应实验，不能只改文案。
3. **验证数据协议**：LoveDA七类含背景、标签reduce_zero与union；iSAID背景忽略；UAVid七类合并车辆；UDD/VDD版本与划分；ISPRS通道来源、clutter、原图/裁块独立性。转换脚本和最终配置优先于历史笔记。跨数据集不自动等于严格未见类别，需处理类别重叠与基础预训练覆盖。
4. **收集标准结果**：保存mIoU/mAcc/逐类IoU、原始metrics与config/权重关联；原始SAM3无适配行标training_dataset=none，不填DLRSD。开发日志不能自动作为独立验证成绩。
5. **共享结果与组表**：Full在主表、核心、冻结表复用同权重/seed；同变体跨域不换更优权重。明确selector后先draft校验，再满足protocol_locked及必需数据条件生成正式四表；单seed不报标准差，多seed用样本标准差。
6. **补效率和定性**：端到端presence基线含完整语义/实例流程；Full推理无教师。定性采用相同样本、统一图例和局部裁切，保留代表性失败案例。
7. **回写正文**：先写实验比较与消融，再根据证据补摘要、实证贡献及结论；不得从架构或单张图推断普遍收益。机制热图/特征分析不在当前计划，有具体科学问题后再决定。

操作入口为[../../docs/experiment_suites.md](../../docs/experiment_suites.md)和[../scripts/README.md](../scripts/README.md)。本轮不改实验章节、表格、数据选择器和配置。

## 5. 五个图位及下一轮绘制

| 图位 | 当前文件/位置 | 正式绘制要求 |
| --- | --- | --- |
| 动机 | ../figures/motivation.tex；引言 | 三种信息流：代价聚合、逐类SAM3、本文解码前特征交互；不虚构错误预测或领先结果 |
| 总体框架 | ../figures/overview.tex；方法概述 | 图像/提示、RemoteCLIP分数到指导特征、两状态、聚合/SDF/冻结解码；训练教师用虚线 |
| 双流细节 | ../figures/aggregation.tex；聚合节 | Q/K与目标流V来源、语义先更新、类间/类内轴、普通/移位窗口及额外空间指导 |
| SDF细节 | ../figures/sdf.tex；融合节 | 两分支、共享方向卷积、逆向对齐、残差、分支输出投影与最终投影 |
| 定性对比 | ../figures/qualitative_placeholder.tex；main在实验章节后输入 | 仅放真实预测；提供figures/qualitative_comparison.pdf时自动关闭占位，由实验章节现有条件块加载真实图 |

前四个图下一轮直接替换对应tex中的盒子，保留英文caption和label。定性图的列与caption须在结果阶段一起核对。占位图不作为实验或视觉效果证据，不加入机制结果图。

## 6. 文献与资源缺口

- 当前文献数值以data/literature_results.csv、literature_inventory.csv及comparison_audit_20261117.json为机器来源，仍保留reference_only/unresolved；逐格核对和PDF来源见comparison_audit_20261117.md及literature_fill_audit.md。不修改原始数值、哈希或状态。
- GSNet官方LD50K训练与RSKT Table1的iSAID/ViT-L报告分开。OV3原报告47.4与ProC复现41.6分开，不归因为未经证实的六/七类差异；SegEarth-OV跨论文转报差异不按列混拼。
- 四个LoveDA单源参考值获准展示仍不代表历史类别映射完全核实。参考区52项所选值不与本地结果混算平均或统一排名。公开checkpoint与论文实际出表权重不自动等同。
- 保留Zotero自动导出references.bib。新增GeoSeg-OV元数据与方法背景来自arXiv:2608.10426v1及官方代码，单列references_extra.bib；未新增它的实验行或成绩。
- DLRSD原始资源条目缺作者/年份、ISPRS官方资源引用及VDD全文仍需在正式投稿前补核；CAT-Seg/ProC等实际阅读PDF与BibTeX出版版本须区分。论文中的正式出版元数据修复优先在Zotero进行。
- 本轮删除过期literature_audit.md与旧iSAID空results.csv；必要事实已并入本节，PDF哈希和逐格来源继续保留在现有证据文件。旧审计及代码快照可在基线commit的Git历史查阅。

## 7. 实际验证

本轮8项现有组表测试通过；静态核对20个唯一引用键、全部论文输入/引用、五个图位、Markdown链接及英文方法说明。24个公式环境与基线逐字一致，13个受保护文件（实验章节、原BibTeX、data/tables与export_config）逐字节未改，模型与配置不在改动集合中。新增GeoSeg-OV使用独立BibTeX文件，不覆盖Zotero导出库。

完整`latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex`失败，首先被本环境缺少xelatex.fmt阻挡，IEEEtran/ctex/字体依赖也不完整。临时构建格式后的ASCII正文/英文数学/五图位检查采用article双栏、249pt列宽，生成6页XDV，第二轮无Overfull和未定义引用警告。此检查不覆盖中文、IEEEtran浮动体、真实参考文献排版或PDF视觉验收；正式绘图和最终交付前仍须在完整TeX环境编译审阅。未修改模板来规避依赖限制。

此前方法公式修订的14项语义路径测试及9组损失/梯度/分块数值核对属于基线证据，本轮未重跑训练代码，也未运行正式权重或用户数据；不能据此写实验收益。
