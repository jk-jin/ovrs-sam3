# 本地全文与文献成绩核验（2026-11-11）

## 范围与方法

以本机最新、已被用户修改的 references.bib 为准，未覆盖或重新导入。当前19个条目，18个条目有可读取PDF，共19个附件（CLIP条目有两个不同哈希附件）。所有附件均检查 `%PDF-` 文件头并成功用 `pdftotext -layout` 提取；网页快照不算PDF。对12篇与当前数据/方法直接相关的文献阅读了相关设置、数据说明或表格段落；其余附件仅提取文本，不冒称已完成数值核验。实际录入数字来自5篇PDF，并渲染相关表格页目检。

共享清单：`data/literature_inventory.csv`，记录key、题名、BibTeX发表信息、实际PDF版本、文件名、哈希和阅读状态。本机绝对路径、提取文本及整页图像仅在被忽略的 `build/literature/`：`local_manifest.json` 可定位附件。未复制全文到可共享源码。

字段解析保留空格/中文/转义，不按空格切分；多个附件按未转义分号分隔，识别描述:path:application/pdf及storage相对语义。实际附件为普通绝对路径。不存在路径只在项目论文范围及本机Zotero storage检查，未扫描其他私人目录。VDD条目没有file字段，在上述范围未找到相关PDF；官方Potsdam/Vaihingen原始资源条目在本地bib缺失。

## 已录入来源及视觉核对

| 实际来源key / PDF版本 | 设置与页序 | 表格、录入范围与性质 |
| --- | --- | --- |
| liExploringEfficientOpenVocabulary2026 / AAAI-26正式版 | PDF p.3–4、6–8；印刷5984–5985、5987–5989；共10页 | PDF p.5/印刷5986，Table 1，完整阅读DLRSD/iSAID两panel；仅录iSAID/ViT-L的CAT-Seg、GSNet、RSKT-Seg三行，共24项mIoU。前两者为来源论文在OVRSISBench重评，RSKT为自身报告。原始列顺序按数据集名称重排。|
| liPerturbationRegularizedOpenVocabularyRemote2026 / arXiv:2604.15652v3，2026-10-04 | PDF p.5、11–14相关设置/附录；无印刷页码处留空 | PDF p.5 Table 1，只录iSAID/ViT-L Pi-Seg自身报告的7个目标数值，源域破折号不补；Table2多源结果不拼接。|
| liSegEarthOVTrainingFreeOpenVocabulary2025 / arXiv:2410.01768v2，2024-11-04 | PDF p.5–6；附录p.13/印刷1与p.14 Table7 | PDF p.6/印刷6，Table1 SegEarth-OV自身行，录7个固定集合内数值。LoveDA附录明确七类含背景，可入参考区；iSAID是16类含背景，与本地15类不同。|
| liSegEarthOV3ExploringSAM2026 / arXiv:2512.08730v2，2026-04-22 | PDF p.5–6实验设置与正文 | PDF p.6/印刷6，Table1完整SegEarth-OV3自身行7项；Table4 Semantic Only两项。后者是该来源对SAM3语义变体的实验，方法引用SAM3，数值引用OV3。|
| songPromptCalibratedSAM32026 / arXiv:2606.21863v2，2026-07-22 | PDF p.3–5设置、Table I表题/星号、相关正文 | PDF p.4/印刷4 TABLE I，Ours行7项为自身报告；SegEarth-OV3*行7项明确为复现（reproduced_by_source）。不是根据接近程度推断。|

共录入61个原始mIoU标量：53项 `reference_only`、8项 `unresolved`（LoveDA七类口径未确认），0项 `matched`。主表显式选择其中54项来源记录；7个所选LoveDA值被阻止显示，实际显示47个参考数值。额外7项OV3复现记录仅存档，不自动覆盖原报告。所有值单位为percent，保留原文小数位，未录入源论文平均。

每个标量均有本地PDF文件名、SHA-256、版本、PDF页序、可辨认印刷页码、表号、行列、方法引用与来源引用。印刷页码缺失不是造一个编号。未知划分/类别数/目标选择/TTA字段保持空白或TODO。未以BibTeX摘要、搜索结果、fixture、日志或模型记忆填写数字。

## 版本、冲突和限制

1. BibTeX将SegEarth-OV登记为CVPR2025，但实际附件为2024年arXiv v2；ProC-SAM3登记GRSL2026，附件为2026年7月arXiv v2；CAT-Seg登记CVPR2024，附件为arXiv v2。保留用户库，数值/阅读版本以PDF为准，不能声称已读其正式版。完整差异见inventory。
2. OV3自身Table1的LoveDA 47.4与ProC的星号复现41.6不同；iSAID27.6/26.0、Potsdam57.8/57.2、UDD5 71.7/71.6、VDD64.5/64.4也分别登记。两源都未明确本次所需LoveDA七类含背景，因此不把任一LoveDA数值放入七类主表。
3. SegEarth-OV实际arXiv v2 Table1的Potsdam47.1，而后续OV3 Table1的SegEarth-OV行是48.5，Vaihingen也为40.0而原附件为29.1。后者是否来自不同正式版本/设置/重评尚未查清，不猜报告性质，不用其覆盖原附件。主表只选原附件记录。
4. RSKT-Seg正式版缺附录C/D：无法确认详细样本、训练/评测划分、类别删除与波段。其表内iSAID成绩不能仅凭接近程度推断为与本地15类口径一致。
5. Pi-Seg Table1称保留单源OVRSISBench映射；Table2及附录属于多源新协议，不可互换。其附录Table13明确UDD5去掉Background/Clutter；不能把该多源定义套到本地5类或Table1，也不能混用多源平均。
6. SegEarth-OV附录UAVid正文说5前景+背景，但Table7实际列出7个提示名称且说明合并moving/static car，内部不一致；类别数留空，仍仅作参考。原UAVid论文有8类，本地配置也有8类，并不证明与该来源相同。
7. 文献未明确报告的seed、样本数、训练步数、目标标签选择或TTA均不补默认值。SAM3 Semantic Only只在OV3 Table4中有对应局部结果，不把其他方法成绩或完整双头成绩当它的结果。

## 原始方法和数据文献核对

- CAT-Seg本地全文：相关实验/训练/附录说明用于核实方法与通用CLIP资源；本轮不从其自然图像表搬遥感成绩，RS成绩实际来源为RSKT-Seg。
- GSNet AAAI原文：PDF p.5–6/印刷9440–9441的设置和Table2明确LandDiscover50K训练，另有COCO-Stuff初始化设置；RSKT重评iSAID行与其原始大规模训练分开，不把资源改名。
- DLRSD原始PDF（PerformanceEvaluationSingleLabel）：p.1与数据说明给出17类；用户BibTeX缺作者/年份等信息，本轮不覆盖自动导出库，建议后续在Zotero修正。
- iSAID、LoveDA、UAVid、Chen等UDD相关原始PDF均存在，阅读数据/划分相关段落。LoveDA原文强调地理分区独立性；UAVid有原始序列划分。原始说明不替代本地裁块清单，也不替代RSKT重处理协议。
- VDD：缺少可读取全文；本地7类来自配置，不来自其BibTeX摘要。Potsdam/Vaihingen：原始官方资源条目待补。
- RemoteCLIP、SAM3、CLIP、ConceptBank、PCA-Seg、SkySense-O：本轮其附件可提取，未从中新增成绩。PCA-Seg用于现有相关工作，SkySense-O为可选项，尚未完成对应资源/协议核验，不强加参考数字。TPOV-Seg未定位到对应bib/PDF；ROSS不恢复为必需项。

## 尚待解决

取得RSKT-Seg完整协议附录或相应明确文档；确认所有来源的LoveDA含背景口径；补VDD全文和ISPRS原始条目；核查iSAID背景、UAVid车辆合并/序列、UDD5重标、VDD test/val及全部波段/裁块。没有这些证据时，文献数值只留参考，不以形式上同名的数据集认定matched。
