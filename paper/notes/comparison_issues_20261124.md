# 对比实验与引用：下一步需要解决的问题

日期：2026-11-24。检查基于用户最新 `1e4bb76` 文献提交。26条Zotero条目保持逐字节不变；缺少的10条比较方法引用写入 references_extra.bib。主表25行（22个文献方法、3个待跑本地方法），独立条件参考表3行。固定八数据集不变，无文献均值或同条件排名。

## 优先处理

| 优先级 | 问题 | 当前已采取的处理 | 下一步需要的证据／操作 |
| --- | --- | --- | --- |
| 高 | GLoP-SAM3 原文全文未读；现有分数只能核对到 ROSE-SAM3 Table I 的转引行 | 引用用户最新 GLoP 条目，同时引用 ROSE 数值来源；标成 GLoP-SAM3 (ROSE report)，report_type 为 requoted_by_source | 取得 GLoP 原文的结果表、类映射、划分与后处理说明，再确认整行是否一致；论文宣称代码地址本轮返回404 |
| 高 | PRS-Seg 只有出版方摘要；离线原型库是否使用标注、支持划分、评测类数不明确 | 三个分数48.1/53.8/59.7只进入独立参考表并附标记；证据类型 publisher_abstract，不伪造PDF、页码或表号；不放入无监督Training-free主组 | 取得全文方法与实验部分，明确原型库来源、是否使用GT、是否同域支持、LoveDA背景与Potsdam/Vaihingen类映射 |
| 高 | RSKT/Pi单源LoveDA历史标签映射尚未完全对齐；换成DLRSD训练并不能消除该问题 | 新DLRSD行取同一原表，LoveDA值保留unresolved与dagger；其余值只为reference_only | 作者逐类IoU/union日志或绑定出表版本的明确标签映射；既有证据见 literature_fill_audit.md |
| 高 | UAVid文献中有8类，本项目合并car后为7类；iSAID背景政策也可能不同 | 不混算均值、不标粗跨来源最优，明确只为文献参考；ConceptBank的iSAID明确为含背景16类 | 正式同条件对比需统一类别、背景、数据划分和后处理后复测；不通过把旧数字换标签来声称对齐 |
| 中 | ConceptBank重复导出：peiTamingSAM3Wild2026 与 peiTamingSAM3Wild2026a 的DOI相同 | 全文统一引用前者，不改Zotero导出文件；v1成绩明确是GT support版本 | 在Zotero合并重复条目后重新导出；若要用伪掩码支持版本，需核对其逐数据集成绩，不能沿用GT版本数值 |
| 中 | ActiveSAM目前官方方法/代码已有更新；遥感数字为ProC论文复现，缺少出表代码版本绑定 | 行名和表注都标明 ProC-SAM3 reproduction；不称最新官方遥感成绩 | 确认复现commit/设置，或在统一协议下复测当前官方版本 |
| 中 | ROSE-SAM3发布的iSAID PromptBank为25个查询，论文为26个，README说尚未重新评测 | 使用论文Table I报告值；记录其与发布代码的差别 | 获取论文对应26查询提示库，或记录25查询发布版实测结果 |
| 中 | 部分方法仍以预印本表格取数，但bib条目已是期刊版（例如ProC、RSGPNet） | 每个数值记录独立保存PDF版本、SHA-256、页码及表号；不把期刊引用等同于已核对期刊全文 | 拿到正式版后逐行核对是否改变，不按列拼接不同版本的数字 |
| 低 | DLRSD原始文献条目 PerformanceEvaluationSingleLabel 缺作者与年份 | 不从这个不完整条目取成绩；数据协议引用RSKT基准 | 在Zotero补全原始数据集文献元数据 |

## 本轮已明确区分的设置

- GFMR完整版本是每图一次GroupNorm参数更新的测试时适配，不进入冻结参数Training-free主组。
- ConceptBank v1使用带标注目标域支持，进入独立参考表。Potsdam原文为60.5；不采用检索镜像的54.5。
- PRS-Seg暂为原型监督条件待核，不因为SAM3/OpenCLIP冻结就宣称无标注。
- RSGPNet只用其自身四集结果47.8/59.7/61.1/71.0，不用ROSE复现填其余列。
- VLex-SAM3、ConceptSeg按用户要求排除，不引用其分数或将其加入正式表格。
- 旧iSAID训练值保留在历史CSV中，新主表明确选择新增DLRSD记录；170个所选文献单元格逐项登记，均不作为本地复现实验。
- 26条Zotero记录未改；10条手工新增引用以原文/出版方或arXiv BibTeX为依据，和Zotero条目没有重复key。

## 范围外仍需处理

方法章节和架构图保留了旧的聚合块/移位窗口等描述，仍与当前模型不完全一致。本轮只同步与比较方法有关的相关工作段落，没有重写方法与架构图。正式投稿前需要另行统一。
