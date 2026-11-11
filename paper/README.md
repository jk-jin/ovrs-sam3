# ReCA-SAM3 中文草稿

中文暂定标题：ReCA-SAM3：面向开放词汇遥感语义分割的 RemoteCLIP 引导跨类别特征聚合。

2026-11-09 初次创建时仓库不存在 paper/，也未发现已有 LaTeX 正文、文献库或仓库内 IEEE 模板。使用本机已安装的 IEEEtran.cls（V1.8b）journal 双栏结构，适合作为 IEEE TGRS 投稿前的中文工作框架；作者、正式投稿元信息与最终英文稿另行补齐。

## 来源与工作区状态

- 本轮重建核对日期：2026-11-11；分支：master。
- 当前基线 commit：`ba7054285e10bda9f033c2561e06179156279f5f`；开始时已有用户修改 `references.bib`、`sections/related_work.tex`，本轮保持其字节不变。
- 历史工程整理基线（2026-11-11早前）：`07b2f5de15d6a8bec2de097022da1759548c9fa5`；旧 source_snapshot.json 保留为历史追踪，不作为本轮实际运行证据。
- 本轮只重建论文实验章节、来源记录和组表；未训练、未修改模型或训练配置、未提交Git。
- 模型描述依据当前本地工作区；本轮核对配置见实现表与 evaluation_protocol.md，历史散列快照见 [source_snapshot.json](notes/source_snapshot.json)。
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
| 梯度裁剪 | 合并训练配置为 0.01，当前 DESIGN.md 已一致 | 最终值 TODO |
| 日志与追踪 | 本地 JSONL 完整；W&B 默认离线，只记录主要曲线字段 | TODO：正式实验锁定 group/tags |

## 本地编译

实际使用的命令（从仓库根目录执行）：

```bash
cd paper && latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

PDF：`paper/build/main.pdf`；日志：`paper/build/main.log`。使用 TeX Live 2023/Debian、latexmk 4.83、XeLaTeX；中文正文字体 Noto Serif CJK SC，中文无衬线 Noto Sans CJK SC，中文等宽 Noto Sans Mono CJK SC，拉丁正文 TeX Gyre Termes。字体均已在本机核实，无需下载。中文标题样式为中文工作稿设置，保留 IEEE 双栏布局。

2026-11-09 初稿曾编译为3页。2026-11-11实验重建稿已实际编译成功，退出码0，输出8页Letter PDF，已逐页检查。最终日志无缺字、未定义引用/交叉引用、重复label或Overfull；仅有参考文献条目的Underfull排版提示。保留 IEEEtran 类加载阶段的 TU/ptm 字形替代提示，以及中文没有拉丁小型大写和斜体时回退至正常字形的提示；正文使用上述显式配置字体，不影响本轮中文草稿阅读。

## 文件与续写顺序

- `main.tex`：统一文档结构、字体、中文标题与术语占位宏。
- `sections/`：保留既有中文稿；experiments.tex整体替换为八数据集实验框架，摘要/结论仍占位；method.tex仅统一聚合与SDF术语。
- `references.bib`：当前19条用户文献，包含本地附件信息；本轮未修改。实际数值来源版本与BibTeX发表信息可能不同，见literature_audit.md。
- `AGENTS.md`：后续写作流程与术语约定。
- `notes/evaluation_protocol.md`：选择与正式评测协议、指标保存方式。
- `notes/results.csv`：保留旧空计划行，删除废弃前景汇总字段；无真实记录被覆盖，不再用作数字源。
- `notes/review.md`：段落提纲、主张—证据、自审与下一轮确认项。
- `notes/source_snapshot.json`：关键来源 SHA-256 与当前合并配置，便于追踪带本地修改的实现。
- `.gitignore`：忽略本地 build/ 与 generated/，不保存整篇PDF或临时提取图。

下一轮先确认最终设计及正式评测协议，再逐节补任务定义、模块公式、实验细节和已核实文献。待实验结果确认后补摘要和结论；全文翻译等待另行指令。


## 2026-11-11 八数据集实验重建

实验入口只有 `sections/experiments.tex`，旧单LoveDA框架已替换，结论只引入一次。新结构依次为：数据与协议、实现与比较设置、文献主比较、T00—T09核心训练消融、同权重推理双门、定性与计算开销、紧凑进一步诊断。图表标题与说明按本轮指导书用中文，定性框内用英文。主指标LoveDA仅七类含背景及逐类IoU，不再生成旧前景汇总。

新增/更新文件：

- `tables/`：9张表（协议、处理、实现、八列主表、资源、核心结构、损失、双门、效率）；无不存在的外部图像依赖。
- `data/literature_results.csv`：文献数值唯一录入源；61个已按PDF行列核对的标量，53个仅参考、8个LoveDA口径未确认，0 matched。主表显示47个明确选定的参考数值，其余为--。
- `data/literature_inventory.csv`：19个可读PDF附件对应18个文献条目的文件名/哈希/版本/阅读状态。12篇阅读相关设置或数据段落，数字来自其中5篇；其余只提取文本，未声称完成全文核验。缺VDD全文、RSKT正式版附录和ISPRS原始资源条目。完整冲突见 `notes/literature_audit.md`。
- `data/experiment_plan.csv`：T00—T09与P00/P10/P01/P11全部planned，写清唯一改变、是否重训及证据需求。
- `data/table_spec.json`：八项固定顺序、显式来源ID、目标均值集合与占位规则。当前均值协议未锁定，不对参考数值排名。
- `scripts/build_experiment_tables.py`：不依赖模型/GPU/W&B的轻量组表脚本；`scripts/test_build_experiment_tables.py`为5组必要边界检查。
- `export_config.json`：仅保留既有schema支持的逐数据集原生mIoU/mAcc和LoveDA七类逐类导出，删除旧权重扫描示例。它不是八列文献横表schema；不通过放宽正式工具校验混入文献。

从仓库根目录生成计划表：

```bash
PYTHONDONTWRITEBYTECODE=1 python paper/scripts/build_experiment_tables.py \
  --literature paper/data/literature_results.csv \
  --native paper/generated/data/results_long.csv \
  --spec paper/data/table_spec.json --output paper/tables --draft
PYTHONDONTWRITEBYTECODE=1 python paper/scripts/test_build_experiment_tables.py
```

已执行：生成5张动态表（其余4张为手工维护的协议/资源表）；native长表当前缺失，draft正常生成--和 `tables/build_report.json` 缺失清单；5组边界检查通过。无--draft的严格构建预期退出2，因为真实计划结果和协议锁定尚缺，未伪装正式成功。效率与计时当前仍是占位，不从mIoU长表推导；后续按标准结果JSON的实际efficiency字段和统一计时条件补入。

未来标准结果收集沿用：

```bash
python tools/collect_experiments.py work_dirs/paper \
  --output paper/generated/data/results_long.csv
python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated --tables-only
```

上面两条为未来有真实结果时使用，本轮没有运行训练或导入fixture。新脚本读实际 `eval_dataset`/`miou_fraction` 等字段；显式选择eval_id、完整评测、每seed checkpoint哈希与预定seed集合；不选最高/最后记录，不静默平均同seed多个checkpoint，真实0保留，fraction只转换一次，单seed不显示标准差。新增数值选择示例见 `scripts/README.md`。

编译实际执行了常规命令，并用以下命令强制重建BibTeX与PDF，避免已有构建时间戳造成no-op：

```bash
cd paper && latexmk -g -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

八数据集集合已固定，但RSKT附录、类别/背景、划分/波段/裁块与原始大图或序列独立性尚未全部匹配；当前本地LoveDA val参与checkpoint选择。先确认正式选择与评测方案，再运行矩阵，不把本轮框架交付写成模型实验已完成。全文英译仍等待另行指令。


## 2026-11-11 投稿语言与版式整理

正文保留中文学术表达，图表及其说明使用英文。状态说明、证据缺口、投稿待办和本轮验证集中见 [manuscript_status.md](notes/manuscript_status.md)。表格生成入口和实验数字来源保持不变。本轮只调整paper/，未训练、未提交Git。

用户随后明确授权将上述修改提交到GitHub默认分支；提交范围仍限于paper/源码，不包含本地build/及依赖文件。

## 2026-11-11 文献缺项补核

新增3项LoveDA参考成绩，SegEarth-OV行切换为已核验CVPR正式来源。记录及待用户确认的4项候选见 [literature_fill_audit.md](notes/literature_fill_audit.md)。主表仍不对未对齐参考行排名或计算不完整均值。
