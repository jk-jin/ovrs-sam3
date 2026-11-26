# 文献比较与表格约定

本文件维护文献取数、展示和生成规则；本地训练与评测设置见 [evaluation_protocol.md](evaluation_protocol.md)，未完成工作只记录于 [manuscript_status.md](manuscript_status.md)。

## 文献数值与引用

- 文献成绩直接采用选定来源的报告值，保留方法原文与实际取数来源。无需逐仓库复现，也不将外部代码版本、标签处理或其他评估差异作为引用前置任务。
- 方法名后引用原论文；直接取自原论文的数值不另加来源字母。转引或第三方复现按方法分组，在主表短注中引用实际取数论文，不在正文重复展开来源清单。
- SCAN、SAN、SED、CAT-Seg、OVRS、GSNet 使用 RSKT-Seg 的 DLRSD 重评结果；SCLIP、ClearCLIP、ProxyCLIP、CorrCLIP 使用 SegEarth-OV3 报告值。ActiveSAM 使用 ProC-SAM3 复现结果，GLoP-SAM3 使用 ROSE-SAM3 转引值，同时保留方法原文引用。其余方法采用已选原报告。
- 同一行保留选定来源与模型设置，不跨论文逐列挑数；支持 DLRSD 的训练方法采用 DLRSD 设置，SkySense-O 保留 Sky-SA。数值、引用键及来源记录不因精简文字改写。
- 来源版本、表号、页码、指纹及已有评测元数据继续保存在 data/literature_results.csv 和现有审计文件。正文与表注只呈现关键引用，不重复解释外部评测差异。

## 主表展示

| 项目 | 约定 |
| --- | --- |
| 分组 | Training on remote sensing segmentation data / Training-free；ReCA-SAM3 属于前者 |
| 本地方法 | ReCA-SAM3、SAM 3 Semantic-only、SAM 3 Semantic × Presence |
| iSAID | 每行保留两个独立格：15 cls. / IgnBG 与 16 cls. / BG；表注仅解释背景是否计入 |
| 未分类 iSAID | 两格保留 --，原值按方法名在一条紧凑参考注中列一次，不跨列居中或猜测归类 |
| UAVid | 文献原值同列展示，不添加 7／8／u 协议上标；本地七类评测设置只在实验设置说明一次 |
| 其他标记 | 删除来源字母、未解决标签对齐的 † 和重复的 Source／Dev. 表头；来源与元数据仍保留在记录中 |
| 缺失值 | -- 表示未报告，不估算、换算或填零 |

本地 iSAID 双协议仍使用同一方法／种子的同一权重及不同真实 eval_protocol_id；不能把一个评测结果重复填入两格。本地验证规则由 evaluation_protocol.md 维护。

## 生成与排版

1. 修改 data/table_spec.json 与 scripts/build_experiment_tables.py 后重新生成，不能只手改 tables/*.tex。保持二十五个主表数据行、每行十一格及全部已有数值。
2. 表题交代比较对象、指标和单位；表注只保留必要缩写、缺失值和数据来源。训练预算、模型选择、权重复用等在实验设置统一交代，不在每张表重复。
3. 主对比与 LoveDA 逐类表使用跨栏 table*；Table 3 核心消融、Table 4 图文微调和效率表使用单栏 table。按单栏宽度换行表项，不整体缩放成难读的小字。
4. 宽主表在实验文件开头提前声明，让其与实验设置同页；实验章节末尾保留一次 FloatBarrier，确保全部实验表先于结论和参考文献。不要取消全部浮动边界，也不在每个小节或每张表后强制清空。使用 flushbottom 对齐栏底、flushend 平衡末页；编译后同时检查图表归属、单栏表、iSAID 对齐及留白。
5. 本地未测成绩和开销保持 --；Full 的跨表选择与实际来源校验继续保留。核验结果集中于 source_snapshot.json 和 review.md。

## 排除方法与 Zotero

GFMR、ConceptBank、PRS-Seg 不纳入比较或正文引用；VLex-SAM3、ConceptSeg 数据继续排除。references.bib 保持 Zotero 自动导出，references_extra.bib 保存人工补充；本轮不改写两份文献库。删除正文引用不要求删除 Zotero 条目，操作见 [zotero_sync.md](zotero_sync.md)。
