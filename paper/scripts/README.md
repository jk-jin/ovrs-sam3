# 论文组表脚本

从仓库根目录运行 `python paper/scripts/build_experiment_tables.py --draft`。仅用标准库，无GPU、torch、W&B或全文下载。文献只从data/literature_results.csv显式选result_id；本方法只从现有collect工具生成的fraction长表选eval_id，不接受示例或fixture。脚本生成5张动态表及tables/build_report.json；另4张协议/实现/资源表人工维护。

## 显式选择原生结果

在table_spec.json相应行的native_selection中，以真实数据集名选完整记录。必须填写：`eval_ids`、`expected_seeds`、`checkpoint_sha256_by_seed`、`eval_protocol_id`、`eval_split`。所有值均取自已完成实验；当前为空，不提供可误认成实际结果的示例分数/权重/种子。

- 主表：main_rows中的Full/adapter行。
- 核心与损失：ablation_native_selection，以T00—T08为键，再按数据集选择；T00跨表共享同一映射。
- 学生预测过滤：gate_native_selection，以P00/P01为键，再按数据集选择。variant_id仍为full；阈值开关必须与行定义一致，两组checkpoint相同。
- 正式seed集合须预先确定；不能自动取最大、最新或把同seed多checkpoint平均。单seed不显示±0.00，多seed为样本标准差ddof=1。
- 原生LoveDA需完整七个class_id/name及非零union；禁止把其他口径补成七类。
- 若正式模型选择改变，先更新固定目标均值集合及其独立性说明，再设averages_locked=true；缺任一必需项仍为--。参考记录不会自动进入matched区或平均。

## 计时与开销（直接读取既有标准JSON）

标准JSON预留efficiency块，但当前没有确认的实际字段，本轮不猜其结构或伪造数值。`gate_timing_selection`按P组选择；`efficiency_selection`按semantic_only/simple_adapter/full/single_orientation选择各成本列。每个选择项要求：真实`eval_id`、明确的`checkpoint_sha256`、已存在JSON中的`json_path`（必须位于efficiency下）、明确`display_scale`（从原始单位转为表内单位）。

成本列键：total_parameters_m、trainable_parameters_m、time_ms、peak_memory_gb；它们只是论文列选择键，不是第二套结果格式。数值直接取长表result_path指向的标准JSON，不录入新的结果CSV。原始JSON须与所选eval_id和checkpoint匹配。计时的实际数据集/样本、硬件/精度/batch/提示/chunk/缓存必须在协议中注明；不同条件不能汇为同一成本行。当前这些selector为空，全部--；正式模式拒绝未完成的计时和开销。

## 构建模式与检查

- draft：计划缺项生成--及明确缺失清单，不放宽非法数值或冲突校验。
- 不带--draft：全部必需自测/消融/预测过滤/计时/开销和协议锁定齐备才通过；参考方法缺列允许，但不生成不完整平均。
- 测试：`PYTHONDONTWRITEBYTECODE=1 python paper/scripts/test_build_experiment_tables.py`，合成数据仅存内存和临时目录。
- 原生逐数据集严格导出仍用根tools/export_paper_results.py与paper/export_config.json。该schema不接收八列文献横表，不修改根工具的校验。


## Display policy

All generated table captions, notes, headers, switches, and variant names use English. Main-text discussion remains Chinese during review. The eight-dataset table spans both columns; ablation, gate, and efficiency tables use one column and allow top/bottom placement. Source records and numerical validation are unchanged. Manuscript status and pending work are maintained in `../notes/manuscript_status.md`.

## Explicitly approved reported values

`table_spec.json` may list specific `approved_reported_result_ids`. An unresolved record in this list is displayed only in the literature reference panel with a dagger and source-protocol note. It remains unresolved in the source CSV and cannot contribute to a matched-protocol average. Incompatible records remain blocked. This permits explicit editorial selection without changing scientific provenance.
