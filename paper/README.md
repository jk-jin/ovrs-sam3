# ReCA-SAM3 论文工作区

入口为[main.tex](main.tex)，采用IEEEtran journal双栏。一般正文与标题为中文；方法公式及其说明、图题、图内文字和表格为英文。

## 现在读哪些文件

| 文件 | 用途 |
| --- | --- |
| [notes/manuscript_status.md](notes/manuscript_status.md) | 唯一的当前写作状态、待实验主张、图计划和后续操作入口 |
| [notes/evaluation_protocol.md](notes/evaluation_protocol.md) | 当前数据角色、模型选择、结果溯源及四表规则 |
| [notes/review.md](notes/review.md) | 本轮提纲、代码证据与实际检查结果 |
| [notes/comparison_audit_20261117.md](notes/comparison_audit_20261117.md) | 当前文献行的直接来源核验，尤其GSNet的两种训练设置 |
| [notes/literature_fill_audit.md](notes/literature_fill_audit.md) | 仍被文献CSV引用的原文补核证据、页码、PDF哈希及类别限制 |
| [notes/source_snapshot.json](notes/source_snapshot.json) | 本轮代码和论文文件指纹；旧快照通过Git历史查阅 |
| [scripts/README.md](scripts/README.md) | 真实结果产生后的四表生成与验证 |

## 当前论证主线

RemoteCLIP的每提示64模板图文相似度经投影，与稠密RemoteCLIP特征归一化融合，形成类别相关指导特征并初始化语义状态。语义状态与独立维护的SAM3条件特征共同构造注意力Q/K；更新特征流时V来自特征流，更新语义流时V来自语义流。聚合后的SAM3特征经SDF接入冻结解码器。与CAT-Seg/RSKT-Seg的区别是双流更新对象及预训练分割解码接口，不能写成首次引入相似度嵌入、空间/类别聚合或RemoteCLIP。

摘要、引言、相关工作和结论已按此主线整理。24个方法公式保持不变，补充英文机制解释。摘要不罗列训练超参数，不填写领先结果；引言先保留两项方法贡献。现有软掩码蒸馏是预测约束，不被表述为已证实的边界保持。

## 图与实验边界

[figures/](figures/)保留五个英文占位：动机、总体框架、双流聚合、SDF、定性比较。机制热图/特征结果图退出当前计划。正式绘图的输入、布局和替换方式见状态文档。

本轮不改[sections/experiments.tex](sections/experiments.tex)、四张结果表、实验CSV/JSON、模型或训练配置；本方法及两个原始SAM3基线均无正式得分，继续显示--。训练与手选权重验证方式见[批量实验说明](../docs/experiment_suites.md)。实验跑完后再补摘要结果句、实证贡献、结果分析和结论。

## 引用与检查

[references.bib](references.bib)保持Zotero导出原样；人工核实的新增条目放[references_extra.bib](references_extra.bib)，main同时加载两者，避免后续导出覆盖。GeoSeg-OV仅补方法背景，不新增对比实验或成绩。

从仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python paper/scripts/test_build_experiment_tables.py
cd paper
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

需要组表时另运行`python paper/scripts/build_experiment_tables.py --draft`；它会写入tables/，本次文字修订不重新生成表格。完整编译与PDF视觉检查的实际状态见manuscript_status，不用静态检查代替PDF验收。

本轮删除旧iSAID空结果模板`notes/results.csv`及已被后续补核覆盖的`notes/literature_audit.md`，其必要事实移入当前状态文档；旧版本保留于Git历史。原始文献CSV、inventory、逐格审计JSON和仍有引用的补核文档保留。
