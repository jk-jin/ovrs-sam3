# 八数据集评测与模型选择协议（2026-11-11 重建）

本轮替换旧 LoveDA 单数据集实验框架。以下区分已经核对的来源、当前代码和正式运行前待锁定协议；没有启动训练。

## 固定集合与来源

顺序只能是 DLRSD、iSAID、LoveDA、Potsdam、Vaihingen、UAVid、UDD5、VDD。
集合依据：`liExploringEfficientOpenVocabulary2026` 本地 AAAI-26 正式版，PDF p.3/印刷5984、p.5/印刷5986 Table 1。原表顺序为 DLRSD、iSAID、LoveDA、Potsdam、UAVid、UDD5、Vaihingen、VDD，组表按名称重排。

原文说明两个独立单源 panel：DLRSD 与 iSAID。并非联合训练。PDF 将数据与划分详情指向附录 C、指标细节指向附录 D，本地正式版共10页，不包含这些附录。当前只能说采用同一八数据集集合，不能说全部划分/类别/裁块/波段已一致。不得用 Pi-Seg 多源附录替代 RSKT-Seg 单源缺失附录。

## 本文当前开发设置

- 实际入口：`configs/experiments/main/isaid_loveda_full.py`，继承 `configs/protocols/isaid_loveda_semantic.py`。
- 训练：iSAID train，源域监督适配；iSAID val 用于源域保持诊断。正式大图独立性仍需真实清单确认。
- 选择：LoveDA val，`semantic.miou` 七类含背景，取最大；每2000步完整验证配置，`val_max_iters=None`。实际 `is_full` 必须由样本统计确认，不能仅凭配置断言。
- 候选正式目标集合：DLRSD、LoveDA、Potsdam、Vaihingen、UAVid、UDD5、VDD。LoveDA 当前参与选择，不能标为独立目标结果。
- 正式模型选择 TODO：优先考虑源域留出验证或预先固定步数，不使用目标标签。当前没有实施，不修改训练配置。
- 可选 DLRSD 单源：不承诺已做，若启用另列 panel。

## 本地数据与原始数据/文献定义的区别

| 数据集 | 本地 eval 配置类别数与划分 | 本地背景与标签规则 | 仍须核对 |
| --- | --- | --- | --- |
| DLRSD | 17 / val | 无 background；不减零；ignore255 | 基准划分、裁块与原始图像独立性 |
| iSAID | 15 / val | reduce_zero_label=True，原始零值删除 | RSKT-Seg 类别数/切块；SegEarth-OV 本地版本是16类含背景，不能直接匹配 |
| LoveDA | 7 / val | reduce_zero_label=False；背景0保留评测、前向排除 | 基准实际类别与正式选择/评测独立性 |
| Potsdam | 6 / val | clutter5保留评测、前向排除 | 官方原始条目、波段、MMSeg版本、基准裁块 |
| Vaihingen | 6 / val | clutter5保留评测且参与前向 | 同上；道路同义提示映射 |
| UAVid | 8 / val | 背景0保留评测、前向排除 | 车辆合并、序列划分；其他文献包含不同合并版本 |
| UDD5 | 5 / val | 背景4保留评测、前向排除 | 原版/重标版与类别映射；不套用多源附录的删背景版本 |
| VDD | 7 / val | 背景0保留评测、前向排除 | 原始论文全文缺失；基准是否test而本地val |

所有当前图像经 `data/dataset.py` 转为 RGB，本地 eval 统一 resize1008且不保比例。这不能证明原始产品只有RGB，也不能证明与基准使用相同波段。真实样本数、大图/序列、样本清单散列、裁块尺寸及重叠均 TODO。原始数据论文只支持数据介绍，不能替代重处理基准记录。

LoveDA 只保存七类 mIoU、mAcc 和逐类 IoU（含背景）。标准 JSON 使用 fraction；论文 percent 转换一次。必须检查逐类 intersection/union、有效类别数、忽略规则，零并集不能悄悄从七类主指标删除。不得估算背景或拼接不同版本成绩。

## 平均集合与排序

- `Avg-8`：固定全部八项，包含源域，不叫跨域平均。
- `Target Avg`：固定七项，即除iSAID外所有数据集；当前含被选择过的LoveDA，必须注明非全独立。
- 若继续采用当前选择方式，`Independent Target Avg` 固定 DLRSD、Potsdam、Vaihingen、UAVid、UDD5、VDD；不能逐行按可用项平均。
- `table_spec.json` 当前 `averages_locked=false`；所有平均保留--。协议未核实的参考成绩不计算本项目平均，也不复制源论文 Mean 字段。
- 当前没有 matched 文献记录。53条可读数值仅 reference_only；8条LoveDA口径未明是 unresolved，不进入七类列。文献区不排名或加粗。

## 训练矩阵与推理矩阵

T00—T09 以 `data/experiment_plan.csv` 的稳定variant_id对齐。T01残差adapter和T06拼接融合结构由指导书定义，尚未宣称代码支持；T02沿展开提示对角屏蔽，同时去除同类别名交互；T05只清零初始语义状态，仍有CLIP中间层指导；T07保持256通道而非旧128通道。T08/T09关闭对应损失需重新训练。所有变体保持其余基准不变。

最低共同消融目标：LoveDA七类、Potsdam、UAVid；T00/T01/T02优先完整八项。正式单seed清单待预先固定，三者优先补三个预定seed，运行时默认42不等于已实施种子计划。单seed无标准差。

P00/P10/P01/P11只切换reference/prediction门，共用T00同一权重，不重训。关用None；开阈值由开发集预先固定。需记录reference_prob_threshold、prediction_prob_threshold、背景回填阈值、TTA、权重哈希、提示集合、缓存与端到端时间。原始reference解码成本纳入时间。Full/T02未过滤输出是待补诊断，不预写结论。

## 原生结果接口与正式运行前清单

保留每次运行有效config.json、metrics.jsonl、results/<eval_id>.json、对应best.pth、固定样本预测。复用根目录现有collect/export工具，不新增数据库或修改正式校验。

长表实际字段为 eval_id、method_id、variant_id、seed、training_dataset、eval_dataset、eval_split、eval_protocol_id、is_full、checkpoint_sha256、miou_fraction、macc_fraction、per_class、source、result_path。组表必须显式选定eval_id、每seed的checkpoint哈希和完整评测，不能自动取最后或最高成绩。

待确认：八项基准版本/原始类别/标签值域/波段/裁块、原始图像或序列独立性、正式选择方案、正式seed、checkpoint、提示词、后处理、硬件与完整评测清单。先解决这些协议项，再补成绩，不能把当前表格交付写成实验已完成。
