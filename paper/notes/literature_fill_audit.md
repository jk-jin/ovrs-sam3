# 文献补充来源索引

本文件仅保留已有论文与补充材料的来源、页码和指纹；当前选择与逐格来源以 data/table_spec.json 和 data/comparison_audit_20261124.json 为准。历史数值保留在 data/literature_results.csv，旧填写决定可从 Git 历史查阅，不作为当前任务。

| 文件 | 来源URL | SHA-256 | 本轮查看页面 |
| --- | --- | --- | --- |
| SegEarth-OV3 v2 | https://arxiv.org/pdf/2512.08730v2 | b7cf3e68414a1eadd34709263f4ee137725b813766491c79bcf8645473bed930 | p.6 Table1/4 |
| ProC-SAM3 v2 | https://arxiv.org/pdf/2606.21863v2 | f21de49839da3a8c7b8e29bbd5be74686d6d5830b44359dba0908c82b89ee11d | p.4 TABLE I及表注 |
| Pi-Seg v3 | https://arxiv.org/pdf/2604.15652v3 | 83e679e4efedbcdebb371d339ec696a77562d108f2983a053f87b72b0653f85b | p.5 Table1/2及设置；附录不同协议分开 |
| RSKT-Seg v3 | https://arxiv.org/pdf/2509.12040v3 | 6fe6b20ccb7f9fc3d2303da2511792a1cb7d0da650adb2fbc188b6ebf6b28701 | p.6 Table1及p.9附录 |
| SegEarth-OV CVPR | https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf | 944eb077fe1c7fbd570b7abc1a5369b1b896698bf8577e57ecc8e6207d2ddc71 | PDF p.6/印刷10550 Table1 |
| SegEarth-OV CVPR supplement | https://openaccess.thecvf.com/content/CVPR2025/supplemental/Li_SegEarth-OV_Towards_Training-Free_CVPR_2025_supplemental.pdf | 520b521080ecff83a535da9f469b16b4dadf00559a6fa17e08e78280ae72b8ae | p.1数据定义；p.2类名表 |

## 当前仍需区分的来源

- SegEarth-OV3 原报告 LoveDA 47.4 与 ProC-SAM3 第三方复现 41.6 是不同记录，不能替换、拼接或把差异无依据归因为六／七类切换。
- SegEarth-OV 自身 CVPR Table 1 的整组结果与后续转报行分开保存；不按列挑选更高分。
- 文献基线的数字不能挪用为本仓库 Semantic-only 或 Semantic × Presence 的本地测量值。
- 既有官方代码核验记录保存在 data/protocol_evidence_20261124.json，仅作补充证据，不作为当前引用的阻塞条件。

长期约定见[comparison_policy.md](comparison_policy.md)，真正未完成工作见[manuscript_status.md](manuscript_status.md)。
