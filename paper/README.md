# ReCA-SAM3 论文工作区

正文暂用中文，图表、表题和表注用英文。表格源数据由 data/table_spec.json 与 scripts/build_experiment_tables.py 管理；本地未测量的成绩与开销为 --。

## 阅读入口

| 文件 | 用途 |
| --- | --- |
| [notes/comparison_policy.md](notes/comparison_policy.md) | 文献直接引用、双协议 iSAID、单列 UAVid、未知值展示与 Zotero 的长期约定 |
| [notes/evaluation_protocol.md](notes/evaluation_protocol.md) | DLRSD 训练、LoveDA 开发域、权重选择、十个消融变体与本地推理规则 |
| [notes/manuscript_status.md](notes/manuscript_status.md) | 只记录未完成的工程、实验、正文和图稿工作 |
| [notes/zotero_sync.md](notes/zotero_sync.md) | 拉取、重新编译、可选集合整理及自动导出 |
| [scripts/README.md](scripts/README.md) | 真实评测选择器、生成命令与证据字段 |
| [notes/review.md](notes/review.md) | 当前主张与证据、验收标准 |

## 五张实验表

| 表格 | 内容 |
| --- | --- |
| tables/main_rskt8.tex | 22 个文献方法和 3 个本地方法，按遥感标注训练／Training-free 分组；iSAID 两个独立协议格，UAVid 一列 |
| tables/loveda_per_class.tex | 本地三方法的 LoveDA 七类 IoU 与 mIoU |
| tables/ablation_core.tex | Full 与六个单因素变体 |
| tables/ablation_freeze.tex | RemoteCLIP 图像／文本微调四组合；共同微调复用 Full |
| tables/efficiency.tex | 两个原始基线、Full、单尺度及独立权重的参数与推理开销 |

```bash
python paper/scripts/test_build_experiment_tables.py
python paper/scripts/build_experiment_tables.py --draft
cd paper
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

正式成绩填写实际标准结果 generated/data/results_long.csv 与选择器，再生成表格。主模型跨表、跨域复用同一权重；iSAID 两种协议分别登记真实 eval_id 和 eval_protocol_id。协议未知的文献 iSAID 原值只在表注保留一次，不能跨两列居中或硬归类。禁止只手改生成 TeX 而不更新数据与生成器。

## 数据与核验索引

- data/literature_results.csv：原报告数值与历史记录；当前 156 项选择见 data/comparison_audit_20261124.json。
- notes/literature_fill_audit.md：来源 PDF／补充材料的页码与指纹；不再维护旧的填写清单。
- data/protocol_evidence_20261124.json：已有代码核验记录，不作为逐仓库复现要求。
- notes/experiment_config_snapshot.json：有效配置快照；notes/source_snapshot.json：源码、论文哈希及最新验证结果。
- references.bib 保持 Zotero 自动导出；references_extra.bib 保存人工补充。未引用条目不会进入参考文献。

消融与iSAID双协议入口已实现，运行方式见 ../docs/experiment_suites.md；实际未运行的实验与模型／图稿同步工作见 notes/manuscript_status.md。旧的已解决调查、日期式问题清单和重复手抄主表已清理；旧内容通过 Git 历史查阅。
