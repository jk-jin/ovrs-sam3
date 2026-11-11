# ReCA-SAM3 中文论文工作区

## 写作流程

- 摘要、正文、章节标题、图表标题和注释中文优先。用户另行发起全文英译后才进入英文阶段。
- 2026-11-11实验重建约定：正文、表题、图题与说明用中文；正式图内部标注用英文，当前图形可用可编译占位框。本条覆盖此前图注英文约定。
- 固定实验集合为 DLRSD、iSAID、LoveDA、Potsdam、Vaihingen、UAVid、UDD5、VDD；LoveDA仅七类含背景及逐类IoU。文献数值以本地真实PDF逐项登记，不覆盖Zotero自动导出库。
- 保持 IEEEtran journal 双栏结构，main.tex 通过 input 组织章节；使用 XeLaTeX 和本机中文字体。
- 写架构前先读仓库 AGENTS.md、DESIGN.md，再核对源实现和最终合并配置。源码优先；保留已有正文和用户本地改动。
- 只在用户授权范围内编辑。本轮范围限于 paper/；不修改模型、配置或根目录文档，不提交 Git。
- 当前实现、最终待确认设计、实验主张分别记录。当前默认值不等于定稿或最优设置，未确认处写中文 TODO。
- 不编造结果、尺寸、超参数、文献、协议或基线。待跑或未知表格数值统一写 --，不能当零。核实原始文献后才录入 references.bib。
- 引入公式时，在首次出现处解释每个符号、索引、作用轴及必要的尺寸；当前框架未写未核对的推导公式。
- 每节续写前先列提纲，再按段落单一主题写作；更新 notes/review.md 的主张—证据映射和五维自审。
- 架构或结果变更后更新 README 的核对记录、配置快照和实验协议；不能仅靠 commit 识别带本地改动的实现。

## 术语约定

| 用语 | 约定 |
| --- | --- |
| ReCA-SAM3 | 工作名称；正式英文全称 TODO，不自行扩写 |
| SAM 3 | 首次写 Segment Anything Model 3；全文正文保留 SAM 3 的空格 |
| RemoteCLIP | 面向遥感图像与文本对齐的 CLIP 模型；不推断未经核实的预训练细节 |
| 特征聚合 | RemoteCLIP-guided feature aggregation；维护语义状态流与特征状态流 |
| 类间注意力 | CrossClassAttention；同一图像同一位置沿展开提示轴交互，包含同类别同义提示 |
| 类内注意力 | WindowSpatialAttention；同一提示内部的空间关系，通过普通与移位窗口建模 |
| 提示条件特征 | `prompt_conditioned_features_72`；完整 SAM 3 编码器及前置提示交叉注意力后的输出，未被聚合模块更新 |
| Semantic–Detail Fusion | 语义—细节融合模块；`SemanticDetailFusion`，单尺度四方向为当前实现，最终变体 TODO |
| 冻结与适配 | 当前称“对冻结 SAM 3 的遥感适配”；RemoteCLIP 部分微调，整个系统不能称完全冻结 |
| mIoU / IoU | 平均交并比 / 交并比；LoveDA 只使用含 background 的七类 mIoU，逐类 IoU 单独保存 |
| BCE / FFN / FPN / TTA | 首次解释二元交叉熵 / 前馈网络 / 特征金字塔网络 / 测试时增强 |
| 迁移与开放词汇 | 跨数据集不能自动称严格未见类别；需核实训练与目标类别的语义重叠 |

## 编译与审阅

在 paper/ 执行 `latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex`。
检查缺字、未定义引用、表格越界和 PDF 双栏布局；build/ 为本地生成目录。
主结果与消融必须记录选择划分和正式评测划分，禁止混用为独立评测证据。
