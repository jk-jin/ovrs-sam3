# 对比结果全文补核与待确认清单

日期：2026-11-11。仓库基线：`c9f4c07e3cfa2c55a53a37183a6542ea445ada5d`。范围：已有主表的文献成绩、来源和口径核查；不改本方法实验结果、模型或训练配置。

## 最新填写决定

用户于2026-11-11明确要求填写CAT-Seg25.11、GSNet27.21、RSKT-Seg29.62和Pi-Seg31.45，保留SegEarth-OV3原报告47.4。四项已进入文献参考区；英文表注注明来源及类别映射未完全对齐。显示许可不等于补充了新的协议证据。

## 1. 本轮实际完成的工作

根据 `references.bib` 的 DOI、标题、arXiv 编号和已有来源登记，直接从 arXiv/CVF 下载原始 PDF，执行文本提取，并渲染相关表格页逐项目检。官方代码仅用来补充类别定义、标签忽略和评测范围证据，数字本身取自论文表格。

没有访问用户本机的 `/home/joker/Zotero/storage/`。OV3、ProC 和 Pi 的下载 PDF SHA-256 与仓库先前登记的附件一致，因而可确认阅读的是相同文件内容，而非凭题名推断版本。

本轮补入三个原本被阻止显示的 LoveDA 数值；将 SegEarth-OV 的七个已选数值切换到已验证的 CVPR 正式来源，数字不变。四项口径未完全确认的数字保持 `unresolved`；按用户本次明确决定，以带标记的原文参考值展示。所有本轮可显示数字仍是 `reference_only`，没有提升为与本方法完整协议一致的 `matched`。

## 2. LoveDA 主表填写情况（mIoU，%）

| 方法/行 | 原表状态 | 本轮处理 | 数值原文位置 | 类别证据与剩余限制 |
| --- | --- | --- | --- | --- |
| SAM 3 Semantic-only | -- | 填35.4 | SegEarth-OV3 v2，PDF p.6，Table4，Semantic Only行 | 同一论文的官方LoveDA配置/类表是七类含背景；它代表OV3语义头消融管线，不是本仓库裸模型自测 |
| SegEarth-OV | 36.9 | 保留并验证正式版 | CVPR2025，PDF p.6/印刷10550，Table1，Ours行 | 官方补充PDF p.1明确6前景+背景、val；p.2类表列七类 |
| SegEarth-OV3 | -- | 填47.4 | arXiv:2512.08730v2，PDF p.6，Table1/4，完整模型行 | 官方cfg_loveda/cls_loveda及全类IoUMetric佐证七类；阈值与数据处理保持原方法设置，不宣称统一复现 |
| ProC-SAM3 | -- | 填46.1 | arXiv:2606.21863v2，PDF p.4，TABLE I，Ours行 | 官方LoveDA注册为七类，IoUMetric对全类求平均；不是已经逐表核对GRSL正式版 |
| CAT-Seg / iSAID / ViT-L | -- | 填25.11（用户确认引用） | RSKT正式版Table1；v3 PDF p.6同值交叉验证 | 论文附录7类与官方注册8项冲突，见下节；以原文参考值标记展示 |
| GSNet / iSAID / ViT-L | -- | 填27.21（用户确认引用） | 同上 | 是RSKT重评，不是GSNet原论文的大规模训练结果；同样受注册冲突影响 |
| RSKT-Seg / iSAID / ViT-L | -- | 填29.62（用户确认引用） | 同上 | 不能仅以附录写7类就忽略官方代码no-data条目 |
| Pi-Seg / iSAID / ViT-L | -- | 填31.45（用户确认引用） | arXiv:2604.15652v3，PDF p.5，Table1下半部 | 当前注册是7类，但该表声明沿用原单源映射，而代码和数据处理更新与V2交织；缺少与该单源表直接绑定的逐类评测证据 |

“填入”表示论文报告值可在明确标注的参考区展示，不表示我们亲自复现了模型，也不表示作者当前代码与当年出表运行逐位等价。精确评测图片、裁片、输入尺寸和阈值仍有方法间差异。

## 3. 用户已确认的填写决定与来源限制

### A. SegEarth-OV3 的47.4与41.6

两个数字都真实存在于已读PDF：47.4是SegEarth-OV3原报告，41.6是ProC-SAM3 TABLE I中带星号的复现行。表注明确星号代表复现。两套官方LoveDA代码都含背景七类，**没有证据把差异归因为六类/七类切换**。两论文管线、提示、阈值及运行版本未被证明完全一致，不能断言某一个具体因素就是原因。

用户明确决定保留47.4；本轮按主表原本“原报告”行定义填写47.4；41.6仍保留为独立候选记录，没有覆盖47.4。建议继续保留原报告47.4；若希望展示重现差异，可以另外增加清楚标明“reproduced by ProC-SAM3”的行，不能合并、平均或只选更有利值。

### B. RSKT来源的三个LoveDA数字

作者README建议使用AAAI正式主文和最新arXiv补充内容，并说明旧附录部分描述/参数统计已更正。本轮取得最新v3（2026-08-18），其Table1中的三个数值与已有正式版登记一致；附录仍描述LoveDA为7类。

然而官方仓库固定commit `6d79d24b0452f19e8429bae1bcd74135173794a9` 的：

- `RSKT_Seg/data/datasets/register_LoveDA.py` 注册8项：no-data、Background、Building、Road、Water、Barren、Forest、Agriculture。
- `RSKT_Seg/data/datasets/LoveDA.json` 同样有8项。
- 元数据 `ignore_label=255`，使用普通 `SemSegEvaluator`，没有看到LoveDA专门跳过no-data的分支。

这不足以证明论文实际按8类统计，因为真实标签可能已经预处理、某些类别可能没有有效并集，且出表代码可能不同；也不足以证明它恰好是本稿的七类。需要原始标签取值、逐类IoU/union日志，或作者针对这三个表格值的明确口径确认。

**用户已明确决定引用25.11/27.21/29.62。** 三项均按RSKT-Seg正式版Table1的iSAID/ViT-L数值填写，方法原论文与数字来源分开引用，附dagger标记说明按来源原协议报告。记录仍保持unresolved，不伪造已核实七类的元数据，不参与同协议均值。

### C. Pi-Seg的31.45

官方 `0c836ba2dfa939b409b046ce6988a5ef8c1f3e34` 的 `cat_seg/data/datasets/register_LoveDA.py` 明确7类含Background，并读取 `masks_png2`。`tools_V1/eval_v1.sh` 使用同一LoveDA注册名，证明现有代码具备七类单源评测路径。

但论文Table1正文称沿用原单源OVRSISBench类别映射；该注册也同时被V2脚本使用。作者仓库此前有JSON/代码整理，尚未拿到与Table1 iSAID/ViT-L那次运行绑定的逐类日志，因此不能把当前七类注册无条件当作该历史单元格的完整证明。

**用户已明确决定引用31.45。** 数值按Pi-Seg v3 Table1的iSAID/ViT-L行填写并加相同来源协议标记。保留历史映射限制记录，不改用Table2多源的43.26，也不把不同训练源的39.00拼入iSAID行。

### D. SegEarth-OV的转报差异

CVPR正式原文Table1：Potsdam47.1、Vaihingen29.1。SegEarth-OV3 Table1转报为48.5和40.0；ProC TableI则列48.5和29.1。当前证据未解释这些差异。本轮保留正式原文自身行的整组数字，不从不同来源挑列混拼。

## 4. 其他缺项为什么不补

- SAM 3 Semantic-only：已核实Table4只给出固定八数据集中的LoveDA与UAVid；其余六项没有对应数值。不用完整双头结果填语义头行。
- SegEarth-OV、SegEarth-OV3、ProC-SAM3：所选原论文主表没有DLRSD。不能用OpenEarthMap代替DLRSD。
- Pi-Seg / iSAID：所选Table1源域iSAID为破折号。不能从别的训练源、多源表或其他论文挪一个源域数字填进去。
- 本方法和adapter行仍无确认native结果，不能引用其他方法代替。
- Avg-8与Target Avg继续不计算：参考协议未锁定且有真正未报告项，不能按每行可用项改变平均集合。
- SkySense-O虽然已在bib中，但不是当前主表的缺失行；本轮不为凑数新增不同大规模训练资源的方法。是否扩展另行决定。

## 5. 来源、页码与文件指纹

下载全文与临时渲染在本地 `research/`，不把整篇PDF加入Git仓库。下列链接均为论文原始发布方。

| 文件 | 来源URL | SHA-256 | 本轮查看页面 |
| --- | --- | --- | --- |
| SegEarth-OV3 v2 | https://arxiv.org/pdf/2512.08730v2 | b7cf3e68414a1eadd34709263f4ee137725b813766491c79bcf8645473bed930 | p.6 Table1/4 |
| ProC-SAM3 v2 | https://arxiv.org/pdf/2606.21863v2 | f21de49839da3a8c7b8e29bbd5be74686d6d5830b44359dba0908c82b89ee11d | p.4 TABLE I及表注 |
| Pi-Seg v3 | https://arxiv.org/pdf/2604.15652v3 | 83e679e4efedbcdebb371d339ec696a77562d108f2983a053f87b72b0653f85b | p.5 Table1/2及设置；附录不同协议分开 |
| RSKT-Seg v3 | https://arxiv.org/pdf/2509.12040v3 | 6fe6b20ccb7f9fc3d2303da2511792a1cb7d0da650adb2fbc188b6ebf6b28701 | p.6 Table1及p.9附录 |
| SegEarth-OV CVPR | https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf | 944eb077fe1c7fbd570b7abc1a5369b1b896698bf8577e57ecc8e6207d2ddc71 | PDF p.6/印刷10550 Table1 |
| SegEarth-OV CVPR supplement | https://openaccess.thecvf.com/content/CVPR2025/supplemental/Li_SegEarth-OV_Towards_Training-Free_CVPR_2025_supplemental.pdf | 520b521080ecff83a535da9f469b16b4dadf00559a6fa17e08e78280ae72b8ae | p.1数据定义；p.2类名表 |

官方代码佐证：

- OV3：[cfg_loveda](https://github.com/earth-insights/SegEarth-OV-3/blob/87e5cc18c8bc9c0b4713b975587281ac1c97c79a/configs/cfg_loveda.py)、[cls_loveda](https://github.com/earth-insights/SegEarth-OV-3/blob/87e5cc18c8bc9c0b4713b975587281ac1c97c79a/configs/cls_loveda.txt)、base_config中的全类IoUMetric。
- ProC：[LoveDA注册](https://github.com/YanghuiSong/ProC-SAM3/blob/00f9d07bf78048397b1ee20dd7044bac0c485d40/mmseg/datasets/loveda.py)、同commit的cfg_loveda与IoUMetric。
- RSKT：[LoveDA注册](https://github.com/LiBingyu01/RSKT-Seg/blob/6d79d24b0452f19e8429bae1bcd74135173794a9/RSKT_Seg/data/datasets/register_LoveDA.py)、同commit类表与train_net评测器。
- Pi：[LoveDA注册](https://github.com/LiBingyu01/Pi-Seg/blob/0c836ba2dfa939b409b046ce6988a5ef8c1f3e34/cat_seg/data/datasets/register_LoveDA.py)、[V1评测脚本](https://github.com/LiBingyu01/Pi-Seg/blob/0c836ba2dfa939b409b046ce6988a5ef8c1f3e34/tools_V1/eval_v1.sh)。

说明：这些是类别口径佐证，不是远程运行的评测日志；不将代码推断包装为论文明确写出的事实。

