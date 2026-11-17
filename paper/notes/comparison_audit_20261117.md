
# 2026-11-17 对比成绩来源核验

本轮逐项核对主表保留的七个文献行、52 个 mIoU 数值。机器可读逐格记录见 `../data/comparison_audit_20261117.json`。没有发现与所选来源表格的抄录差异；这不代表已在本仓库复现，也不代表与本方法的类别、划分、裁块、阈值和模型选择协议完全匹配。原 literature_results.csv 的数值、PDF 哈希及 protocol_status 均保持不变。

## GSNet：两个训练设置必须区分

GSNet 官方 [README](https://github.com/yecy749/GSNet) 的 Prepare datasets 明确写 Training Dataset: LD50K。其原论文为 AAAI 2025，不能把我们的 iSAID 行称作 GSNet 官方 LD50K 权重结果。

当前八项成绩来自 [RSKT-Seg Table 1](https://arxiv.org/html/2509.12040v3#S5.T1) 的 iSAID as Training Dataset / GSNet / ViT-L 行；历史原始出处是 AAAI-26 正式版 PDF p.5、印刷5986，本轮用同作者 arXiv v3 主表再次比对。按本稿列顺序是 DLRSD 31.50、iSAID 93.11、LoveDA 27.21、Potsdam 28.50、Vaihingen 25.10、UAVid 15.98、UDD5 32.24、VDD 32.07。RSKT 正文与表头支持“该来源的 iSAID 单源设置”；不据此推断完整初始化、是否继承其他预训练资源或某个公开 checkpoint 的具体身份。

主表改为 GSNet (RSKT report)，Adaptation data 写 iSAID；方法引用 GSNet，数值来源在表注引用 RSKT-Seg。这是第三方报告，不能称为我们复现或官方 LD50K 实验。没有将其训练数据改填 LD50K，因为那会把一组 iSAID 数字错误归入另一实验。

## 其他行的核对

| 行 | 直接核对来源 | 结论 |
| --- | --- | --- |
| CAT-Seg | RSKT-Seg v3 Table 1，iSAID/ViT-L，8项；原正式版记录保留 | 数值一致，是 RSKT 的报告，不是 CAT-Seg 原论文自然图像设置 |
| RSKT-Seg | 同表自身 iSAID/ViT-L 行，8项 | 数值一致；不混用 DLRSD 行或 README 新权重成绩 |
| Pi-Seg | arXiv:2604.15652v3 Table 1，自身 iSAID/ViT-L 行，7项 | 数值一致；源域 iSAID 为破折号，继续留空；不拼接多源 Table 2 |
| SegEarth-OV | CVPR 2025 原文 Table 1，自身行，7项 | 出版社索引表可核对；Potsdam47.1、Vaihingen29.1保留，不用后续他文转报48.5/40.0覆盖 |
| SegEarth-OV3 | arXiv:2512.08730v2 Table 1，自身行，7项 | 数值一致；LoveDA47.4是原报告 |
| ProC-SAM3 | arXiv:2606.21863v2 TABLE I，Ours行，7项 | 数值一致；该文SegEarth-OV3*的41.6明确属于复现，与47.4不互换 |

RSKT 官方出版页面可访问，但本轮完整出版社 PDF 下载被拒绝；CVPR 全文直连也受限，其索引表可检索。没有声称本轮重新下载或计算这些 PDF 的哈希。arXiv 正文通过官方摘要页的 HTML 链接读取。所有新核对只确认数值与来源，不把原 reference_only/unresolved 提升为 matched。

## 尚未消除的可比性限制

- CAT-Seg、GSNet、RSKT-Seg、Pi-Seg 的四个 LoveDA 值依用户之前授权保留并加 dagger；历史单源类别映射仍未完全对齐，继续 unresolved。
- iSAID 15类/16类含背景、UAVid车辆合并、UDD5背景及视频数据划分等沿用原协议限制；不计算混合平均或跨区排名。
- SegEarth-OV3 原文说明按数据集调整背景和实例阈值；这是原报告条件，不等于我们固定开发集阈值的条件。
- RSKT README 明确 DLRSD/ViT-L 原权重遗失后重训；Pi README 明确公开权重与论文不同。不能将直接下载的权重自动标为论文同一实验，也不能直接沿用未核对的旧参数统计。
- 本地 SAM 3 Semantic-only 与 Semantic × Presence 是新计划，没有实测分数；不把原来文献35.4/47.1挪入本地实测区。
