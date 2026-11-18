
# ReCA-SAM3 中文论文工作区

一般正文和章节标题为中文，表格、表题、表注与图内文字为英文；方法章节的公式及其说明按用户要求直接使用英文，其余正文的英译等待指令。主入口paper/main.tex，IEEEtran journal双栏。当前无本方法正式成绩，所有计划项显示--。

## 当前表格结构（2026-11-17）

| 顺序 | 文件 | 内容 |
| --- | --- | --- |
| I | tables/main_rskt8.tex | 八数据集对比，两个原始SAM3基线、adapter、Full与文献参考分区；显示适配数据，不显示综合均值 |
| II | tables/ablation_core.tex | 聚合、融合和蒸馏合表；删除通用CLIP变体 |
| III | tables/ablation_freeze.tex | RemoteCLIP图像/文本四种冻结组合与可训练参数 |
| IV | tables/efficiency.tex | 五种模型的参数、端到端耗时与显存 |

删除数据集协议表与预测过滤表，不生成Table S1/S2；表内只显示名称，不显示T/P/B计划编号。其他旧的实现/资源表已删除。必要数据角色、类别和背景处理转为实验正文；完整协议保留notes/evaluation_protocol.md。

## 当前设计与执行边界

当前训练入口configs/experiments/main/dlrsd_loveda_full.py：DLRSD官方5601/1401划分，LoveDA val选择模型。LoveDA是开发域，不能标为独立目标。批量实验版本更新训练配置及消融路径，仍保留当前开发域选择角色。

完整模型36×36双流聚合、72×72 SDF；SAM3冻结，RemoteCLIP图文attention微调。主BCE按存在/不存在提示分组、默认0.8/0.2；固定蒸馏0.5仅存在提示有效像素，不使用边界环或蒸馏调度。DESIGN.md与实际代码优先于历史稿件笔记。

SAM3语义与语义乘presence基线及12个训练入口现已实现并通过CPU合成检查；训练/独立验证由configs/suites/驱动，公共预算为batch 4、10000步、LR 2e-6。未运行用户数据和正式预训练权重的GPU训练/评测；所有成绩仍为--，seed集合与正式结果选择未锁定。详见根目录docs/experiment_suites.md。

## 组表与检查

从仓库根目录运行：

```bash
python paper/scripts/build_experiment_tables.py --draft
PYTHONDONTWRITEBYTECODE=1 python paper/scripts/test_build_experiment_tables.py
cd paper && latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

四张表均由data/table_spec.json、data/literature_results.csv和真实结果选择器生成，不手改数值。规则见scripts/README.md。正式模式要求全部必需原生结果/效率、显式权重和种子及protocol_locked=true；不能用草稿占位发布成绩。

## 文献核验与记录

本轮核对七行52个所选mIoU值，未发现抄录差异；协议仍未完全对齐。重点：GSNet官方使用LandDiscover50K，但本稿引用的八个值是RSKT-Seg Table1的iSAID/ViT-L报告，主表标GSNet (RSKT report)，不是本地或官方权重复现。核对详情及访问限制见notes/comparison_audit_20261117.md和data/comparison_audit_20261117.json。

既有文献CSV、BibTeX和PDF来源指纹不覆盖；47.4与41.6分别是OV3原报告及他文复现。四个历史LoveDA单源值仍带未完全对齐标记。公开checkpoint可能不同于论文权重，不能自动继承论文分数。

notes/manuscript_status.md记录缺口与实际验证；notes/review.md记录提纲与证据自审；notes/source_snapshot.json保留历史工程快照并单独记录本轮变更。没有新训练结果或性能提升结论。

新增核心消融 `encoder_only_update`：只更新encoder feature，score embed初值仍作为可学习的Q/K指导；保留普通/移位特征窗口，跳过语义更新及其专用RemoteCLIP中间层指导。正式得分尚未运行。


## 方法公式补全（2026-11-18）

`sections/method.tex` 已按 master@792490eb4ef1e53c1cc41e61acfbabde44b85d20 的实现补齐24个公式：模板级RemoteCLIP先验、双流初始化、跨提示/窗口注意力及顺序残差、SDF方向聚合及完整输出投影、分组BCE与固定教师蒸馏、别名合并与背景标签恢复。公式说明全部为英文，逐项定义符号、索引、作用轴与必要尺寸。框架图仍为原有占位示意。

本轮不修改实验章节、表格、数值、代码或配置。14项现有语义训练路径测试及9组独立公式/损失/梯度/分块数值核对通过。静态引用检查通过；249pt双栏宽度的临时英文数学源码可生成XDV且无越界公式。完整IEEEtran中文PDF因本环境缺少模板、中文包与字体依赖未能编译，未做新PDF视觉验收；详见notes/manuscript_status.md。
