# ReCA-SAM3 论文工作区

当前代码基线为 `reca-sam3-feature-blocks-v3`：RemoteCLIP score embedding 固定参与注意力 Q/K；每个完整块依次执行类间、普通窗口和移位窗口注意力，只更新 SAM3 encoder feature。SDF 使用单方向双支路残差卷积。默认四个完整块。

旧消融配置、实验计划行、权重选择与消融表已删除。当前仅保留完整模型、两个原始 SAM3 对比及外部文献行；实验数值未运行的部分仍为 `--`，不沿用旧 Full 或旧消融成绩。训练与手选权重验证见 [批量实验说明](../docs/experiment_suites.md)。

## 文件与检查

* `sections/`：中文正文，方法公式及说明使用英文，表格与图题使用英文。
* `data/`：文献原始结果、当前主实验计划和组表选择器。
* `notes/manuscript_status.md`：当前状态与后续工作。
* `notes/evaluation_protocol.md`：数据角色、选权重规则和结果溯源。
* `notes/source_snapshot.json`：本次实现及论文源文件指纹。
* `references.bib` 保持 Zotero 导出原样，人工新增条目放 `references_extra.bib`。

```bash
python paper/scripts/test_build_experiment_tables.py
python paper/scripts/build_experiment_tables.py --draft
cd paper
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

draw.io 源图及现有 PDF 保留原样，当前仍展示旧结构，需按新基线另行重绘。自动导出机制见 `figures/README.md`。本轮没有重绘架构图或补充真实预测影像；论文尚不能作为与新实现完全一致的定稿。
