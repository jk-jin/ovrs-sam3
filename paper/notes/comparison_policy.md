# 文献比较与表格约定

本文件记录已确认并持续适用的规则。未完成工作只写入[manuscript_status.md](manuscript_status.md)；已处理问题不反复列为待办，旧方案从 Git 历史查阅。

## 文献数值与引用

- 已有方法直接采用所选论文的报告值，不要求逐个复现、不以当前仓库与历史出表版本一致作为引用前提。外部代码仅能提供补充证据，不能替代原报告设置或让已接受的引用重新变成阻塞事项。
- 保存方法原文与实际取数来源。第三方复现、转引、原报告分别标明；不把引用值称作本地实测，不宣称读过未取得的全文。
- ActiveSAM 采用 ProC-SAM3 reproduction 行；ROSE-SAM3 采用自身报告；GLoP-SAM3 采用 ROSE-SAM3 转引，同时引用方法原论文与数值来源。无需先拿到 GLoP 全文或核清作者最新代码版本。
- 同一行固定来源、训练设置与模型变体，不跨论文逐列挑数。训练组支持 DLRSD 的方法选 DLRSD 设置；SkySense-O 单列 Sky-SA。SegEarth-OV3 原报告与他文复现成绩分别保留，不能互换。
- 数值、训练资源、划分和协议以已有证据登记；未知项明确标记，不根据“训练／免训练”分组推断类别协议。仅对已明确可比的设置计算提升幅度，不混算跨协议排名或跨数据集均值。
- PDF、出版方 HTML 表格、摘要分别注明证据类型；保存已有来源、表号、页码和指纹，不为了当前引用重复进行外部代码审计。数据索引见 data/comparison_audit_20261124.json，历史记录见 data/literature_results.csv。

## 主表布局与评测口径

| 项目 | 固定约定 |
| --- | --- |
| 分组 | Training on remote sensing segmentation data / Training-free；ReCA-SAM3 属于前者 |
| 本地比较方法 | ReCA-SAM3、SAM 3 Semantic-only、SAM 3 Semantic × Presence |
| iSAID-IgnBG | 忽略所有 GT 背景像素，对十五个前景类求 mIoU |
| iSAID-BG | 保留 GT 背景，背景参与十六类 mIoU |
| iSAID 表格 | 每行始终有两个独立单元格；禁止在数据行将一个分数跨两列居中。协议未知时两格均为 --，原报告值按方法名在表下注释中仅列一次，不硬塞到任一协议列 |
| iSAID 本地结果 | 三个本地方法均预留双协议，同一方法／种子复用同一权重；两个真实 eval_protocol_id 必须不同，不能换 checkpoint 或重复填写一个结果 |
| UAVid | 本地统一七类，moving/static car 合并为 car；文献七类／八类原值保留在同一列，以 7／8 标注，未知标 u，无需重跑全部旧方法 |
| LoveDA | 七类含背景；原始 0 为 no-data，转 255；原始 1..7 仅减 1 一次，禁止二次平移 |
| 缺失值 | -- 代表相应协议没有可填写的结果，不当零，不估算或换算原分数 |

iSAID 两种协议必须各自按完整规则验证：不能仅从十六类均值扣除背景 IoU 冒充忽略 GT 背景；背景预测规则不能读取 GT。UAVid 合并类别的分数也不能通过平均两个 car IoU 换算。含背景评测入口的实现状态见 manuscript_status.md。

## 生成与验收

1. 正式数值改 data/table_spec.json 的选择器和实际结果记录，再运行生成器；不只手改 tables/*.tex。
2. 主表全部二十五个数据行固定十一格；iSAID 的 IgnBG 与 BG 均保留显式占位。未分类参考分数在注释中不能丢失、重复或变成任何一个协议的成绩。
3. 生成后检查缺失值与来源，再编译并查看实际 PDF。重点检查 iSAID 表头与数据的横向对齐、参考分数与正式结果的分离、后续数据列是否移位及表注是否越界。编译成功不等于视觉验收通过。
4. 本地成绩、效率和消融未运行时保留 --；同一 Full 的选择器必须在多表复用。核验记录集中于 source_snapshot.json 和 review.md。

## 排除方法与 Zotero

GFMR、ConceptBank、PRS-Seg 不纳入比较或正文引用；VLex-SAM3、ConceptSeg 数据继续排除。删除论文引用无需删除 Zotero 条目，Git 拉取不会反向删除 Zotero 数据。references.bib 保持 Zotero 自动导出，人工补充保存在 references_extra.bib，避免重复 key；具体操作见[zotero_sync.md](zotero_sync.md)。
