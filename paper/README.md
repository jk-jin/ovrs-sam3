# ReCA-SAM3 中文草稿

中文暂定标题：ReCA-SAM3：面向开放词汇遥感语义分割的 RemoteCLIP 引导跨类别特征聚合。

本轮创建时仓库不存在 paper/，也未发现已有 LaTeX 正文、文献库或仓库内 IEEE 模板。使用本机已安装的 IEEEtran.cls（V1.8b）journal 双栏结构，适合作为 IEEE TGRS 投稿前的中文工作框架；作者、正式投稿元信息与最终英文稿另行补齐。

## 来源与工作区状态

- 核对日期：2026-11-11；分支：master。
- 当前基线 commit：`07b2f5de15d6a8bec2de097022da1759548c9fa5`；工程整理后的本地工作区以 `DESIGN.md` 为准。
- 本轮工程整理覆盖模型命名、配置、日志、标准结果与导出工具；论文正文、引用和排版不在本轮重写范围。
- 本稿描述的是上述 commit 加当前本地修改的工作区，不是纯 commit 快照。文件散列及合并配置见 [source_snapshot.json](notes/source_snapshot.json)。
- 已读根 AGENTS.md、DESIGN.md；已核对 SAM 3 特征/解码调用、特征聚合、注意力轴、RemoteCLIP guidance、Semantic–Detail Fusion、冻结构建、提示合并、损失与评测逻辑，以及训练/评测配置及其继承项。细节入口见下表。

| 机制 | 已核对的源文件 | 本轮写作结论 |
| --- | --- | --- |
| SAM 3 编码及解码 | `models/sam3_image.py`, `models/maskformer_segmentation.py` | 编码与前置提示交叉注意力无梯度；学生经冻结解码器向融合模块回传梯度；教师复用同一头 |
| RemoteCLIP guidance | `models/remoteclip_guidance.py`, `models/openclip_image_encoder.py`, `models/openclip_text_encoder.py` | 局部模板相似度、投影及两次归一化拼接；可训练文本不跨步缓存计算图 |
| 特征聚合 | `models/feature_aggregation.py`, `models/aggregation_attention.py` | 类间沿展开提示轴；类内普通/移位窗口；固定 RemoteCLIP 中间层与 FPN 指导；无 FFN |
| Semantic–Detail Fusion | `models/semantic_detail_fusion.py` | 单尺度语义/细节支路，独立支路参数，各支路四方向共享卷积后逆旋转平均残差 |
| 冻结与部分微调 | `model_builder.py`, `engine/optimizer_builder.py` | SAM 3 冻结，`core.feature_aggregation` 全部可训练，RemoteCLIP 图文 Q/V 与位置参数部分微调；融合 QKV 的 K 梯度屏蔽及权重衰减处理 |
| 提示与损失 | `data/collate.py`, `models/adapters/semantic_adapter.py`, `losses/semantic_criterion.py` | 同义提示共享原始标签，推理最大值合并；主 BCE、蒸馏、忽略区域抑制及关闭的 Dice |
| 标签与指标 | `data/dataset.py`, `engine/evaluator.py` | 背景单独映射；全类、前景组、逐类指标分开保存；零并集类别从均值排除 |
| 训练/评测 | `configs/experiments/main/isaid_loveda_full.py`, `configs/test/loveda.py` 及继承配置 | iSAID train → LoveDA val；七类 `semantic.miou` 选择 checkpoint，正式入口完整验证；正式独立评测协议尚未确定 |

## 配置快照与待定边界

以下是当前训练入口的配置值，**均不代表论文最终版本或经过验证的最优参数**。最终实验必须记录完整合并配置、权重和数据清单。

| 项目 | 当前值 | 定稿状态 |
| --- | --- | --- |
| 特征聚合 | 4 个 cross/intra 交替块、8 头、窗口 12、移位 6、dropout 0.1；36×36 特征 | TODO：确认层数与窗口并做消融 |
| SAM 3 / RemoteCLIP | 编码网格 72×72 / dense 网格 36×36；通道 256 / 768；RemoteCLIP ViT-L-14 | TODO：固定最终骨干与输入设置 |
| 模板与相似度 | 64 模板；余弦分数乘 20；中间层索引 7、15 指导 score 类内 Q/K | TODO：固定模板、确认设计 |
| Semantic–Detail Fusion | 72×72，256 通道，语义/细节双支路，各自四方向共享卷积 | TODO：选择最终变体，验证成本和效果 |
| 冻结 | SAM 3 全冻结；core.feature_aggregation 全训练；RemoteCLIP 图文 attention 模式 | TODO：确认最终微调范围 |
| 损失 | 主 BCE 1.0；Dice 0；蒸馏 0.5，忽略边界外环宽 3；忽略区域抑制 0.01 | TODO：确认目标、权重和监督范围 |
| 推理 | SAM 3 门 0.1；学生门 0.5；评测 prob_thd 0；TTA 关闭；prompt chunk 4 | TODO：固定阈值选择数据和后处理 |
| 优化与预算 | AdamW 1e-4，RemoteCLIP 乘 0.01；批量 2；20k 步；1k warmup；每 2k 评测 | TODO：统一各对照预算 |
| 梯度裁剪 | 合并训练配置为 0.01 | DESIGN.md 写 0.1，与当前 schedule 不同；本稿以配置为准，最终值 TODO |
| 日志与追踪 | 本地 JSONL 完整；W&B 默认离线，只记录主要曲线字段 | TODO：正式实验锁定 group/tags |

## 本地编译

实际使用的命令（从仓库根目录执行）：

```bash
cd paper && latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

PDF：`paper/build/main.pdf`；日志：`paper/build/main.log`。使用 TeX Live 2023/Debian、latexmk 4.83、XeLaTeX；中文正文字体 Noto Serif CJK SC，中文无衬线 Noto Sans CJK SC，中文等宽 Noto Sans Mono CJK SC，拉丁正文 TeX Gyre Termes。字体均已在本机核实，无需下载。中文标题样式为中文工作稿设置，保留 IEEE 双栏布局。

本轮实际编译成功，退出码 0，输出 3 页 Letter PDF，已逐页检查渲染。最终日志无缺字、未定义引用或 Overfull/Underfull 提示。保留 IEEEtran 类加载阶段的 TU/ptm 字形替代提示，以及中文没有拉丁小型大写和斜体时回退至正常字形的提示；正文使用上述显式配置字体，不影响本轮中文草稿阅读。

## 文件与续写顺序

- `main.tex`：统一文档结构、字体、中文标题与术语占位宏。
- `sections/`：摘要、引言、相关工作、方法、实验、结论；摘要和结论保持中文占位，方法含完整概述，其余小节为中文 TODO。
- `references.bib`：空文献库，仅保留核实要求；当前未引用未核实文献。
- `AGENTS.md`：后续写作流程与术语约定。
- `notes/evaluation_protocol.md`：选择与正式评测协议、指标保存方式。
- `notes/results.csv`：结果登记表，结果和正式评测划分留空。
- `notes/review.md`：段落提纲、主张—证据、自审与下一轮确认项。
- `notes/source_snapshot.json`：关键来源 SHA-256 与当前合并配置，便于追踪带本地修改的实现。
- `.gitignore`：仅忽略本地 build/。

下一轮先确认最终设计及正式评测协议，再逐节补任务定义、模块公式、实验细节和已核实文献。待实验结果确认后补摘要和结论；全文翻译等待另行指令。
