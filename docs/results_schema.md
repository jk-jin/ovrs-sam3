# ReCA-SAM3 标准实验结果格式

本格式用于训练目录、外部方法和论文导出之间的稳定交换。

* 只使用 UTF-8 JSON；汇总表使用 UTF-8 CSV；训练曲线使用 JSONL。
* 导出器只读普通文件，不依赖 W&B、GPU、torch、SAM3 或本仓库模型包。
* `schema_version=1`。未知主版本直接拒绝；未知可选字段允许导入。
* 指标统一保存为 fraction（0～1）。percent 只在显式声明时转换一次。

## 1. 文件位置

```text
work_dirs/<experiment>/
  config.json
  metrics.jsonl
  results/<eval_id>.json
  best.pth
  latest.pth
  iter_XXXXXXX.pth
```

* `results/<eval_id>.json` 是一次完成评测的标准记录。
* 验证与独立评测都写标准记录；训练过程验证默认不进入论文正式表。
* 同一权重多次评测使用不同 `eval_id`，文件名包含 UUID 短码。
* 只有完整完成并通过 schema 校验才写 `status="completed"`；显式截断的验证写 `status="partial"`。
* 使用临时文件 + 原子替换写入。

## 2. 单次实测记录

结构与字段说明见模板 `docs/results_template.json`。核心块：

| 块 | 内容 |
| --- | --- |
| `record_type` | `single_run` 或 `reported_summary` |
| `status` | `planned` / `running` / `completed` / `failed` / `partial` |
| `role` | `validation` / `evaluation` |
| `experiment` | run、method、variant、implementation、kind、seed、factors |
| `training` | regime、dataset、split、protocol_id |
| `checkpoint` | path、step、sha256、selection |
| `evaluation` | 数据集、划分、协议、类别、样本、几何、阈值、TTA |
| `metrics` | unit、miou、macc、pixel_acc、per_class、可选 std/n_seeds |
| `provenance` | source、git、config、原始文件/引用 |

### 2.1 单位与缺失

* `metrics.unit` 必须显式为 `fraction` 或 `percent`。
* 规范化结果保存 `unit="fraction"`。
* 真实 0 保留为 0，不能用 `if value` 判断缺失。
* 缺失值写 `null`；NaN / Infinity 直接拒绝。
* 逐类 `iou` / `accuracy` 无定义时写 `null`，同时保留 `intersection`、`union`、`target_pixels`。
* 汇总仍使用当前 evaluator 政策：零并集类别从 mIoU 均值中排除；记录 `valid_iou_classes`。
* 六类 LoveDA 结果不能填进七类主 `miou` 字段，也不能估算背景。

### 2.2 类别对齐

```json
{
  "class_ids": [0, 1, 2, 3, 4, 5, 6],
  "class_names": ["background", "building", "road", "water", "barren", "forest", "agricultural"],
  "metric_class_ids": [0, 1, 2, 3, 4, 5, 6]
}
```

`per_class` 按 `class_id` 对齐并同时校验 `class_name`。不能依赖列表偶然顺序。

### 2.3 checkpoint 与选择

* `checkpoint.path` 相对结果文件解析。
* `checkpoint.sha256` 在保存或首次独立评测时计算并复用。
* 训练内部验证尚无对应落盘权重时允许 `sha256=null`，但必须记录真实 step。
* `checkpoint.selection` 是模型选择划分与指标，不替代最终评测证据。
* 若最终评价仍是 LoveDA val，结果和论文必须如实说明它同时参与过 checkpoint 选择。

### 2.4 completed 记录要求

* 至少一个整体指标非 null；
* `evaluation.class_ids` 与 `class_names` 非空且等长；
* `metrics.per_class` 的 `class_id` 和 `class_name` 必须与评测类别对齐。

native completed 额外要求：

`provenance.source="native"` 且 `status="completed"` 时：

* `experiment.seed` 为 int；
* `checkpoint.step` 为 int；
* `evaluation.expected_samples` 与 `evaluated_samples` 为 int；
* `evaluation.is_full` 明确；
* 类别集合与 `metric_class_ids` 非空；
* 至少一个整体指标非 null；
* 协议、阈值、TTA、样本计数完整。

`is_full` 由实际处理样本数与数据集划分长度比较得到，不能仅凭 `val_max_iters=None` 推断。

## 3. 文献汇总记录

`record_type="reported_summary"`：

* `experiment.seed=null`；无信息时 `checkpoint=null`。
* `metrics.miou/macc/pixel_acc` 保存已报告均值。
* `metrics.std` 是同名指标到标准差的字典。
* `metrics.n_seeds` 保存报告种子数；未知为 null。
* `metrics.std_definition` 为 `sample` / `population` / `unspecified`。
* 不与同方法的原始 `single_run` 再平均；导出器会拒绝混用。

## 4. 外部导入

```bash
python tools/import_results.py --input external/baseline.csv \
  --mapping external/mapping.json --protocol external/protocol.json \
  --input-unit percent --output-dir work_dirs/paper/imported

python tools/import_results.py --input external/standard_result.json \
  --output-dir work_dirs/paper/imported
```

* CSV 输入必须提供 `--input-unit` 与 mapping；JSON 已声明 `metrics.unit` 时可不重复指定。
* mapping 的 `columns` 是“目标字段路径 → 源列/字段路径”；`constants` 是固定元数据。
* protocol JSON 是与标准记录同结构的公共元数据片段（例如 `training` 与 `evaluation`）。
* CSV 中 `factor.<name>` 列自动收集为 `experiment.factors.<name>`。
* 同一事实在 protocol、constants、映射列中冲突时直接报错。
* 无法查证的 seed、step、样本数、背景规则、TTA 保留 null；不补成 seed42、step0 或“无 TTA”。
* 来源写 `external_run` 或 `reported`；文献记录保留引用、表格/页码和评测口径。

## 5. 汇总与论文导出

```bash
python tools/collect_experiments.py work_dirs/paper \
  --output paper/generated/data/results_long.csv

python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated

python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated --tables-only
```

* `collect_experiments.py` 默认只收 `completed`、`role=evaluation`、`is_full=true` 的记录。
* `export_paper_results.py` 严格模式是默认值；selector 未匹配、协议不一致、缺少期望种子或完整样本数时非零退出。
* `--allow-incomplete` 只生成明确标记的 draft，不得作为正式论文表。
* 多 seed 先对每个 seed 求数据集级指标，再跨 seed 求均值和样本标准差（`ddof=1`）。
* 单 seed 的 std 为 null，不显示 `±0.00`。
* 表格按 `allowed_varying_fields` 显式声明允许差异；其它协议差异报错。训练配置指纹排除输出目录、设备、日志、保存/验证间隔、workers 和数据路径。
* 正式导出还会检查原生记录的 `valid_iou_classes` 是否等于 `metric_class_ids` 数量；缺失类别只能生成 draft。
* `paper/generated/` 下输出：
  * `data/results_long.csv`
  * `data/<table>.csv`
  * `tables/<table>.tex`
  * `figures/<curve>.pdf` / `.png`
  * `export_report.json`

## 6. 测试

合成 fixtures 位于 `tests/fixtures/`，明确标注为测试用途，不是真实成绩：

```bash
python -m unittest discover -s tests -v
```

覆盖：单位和 0/null、多种子均值/样本标准差、导入字段映射、重复/冲突 eval_id、同 seed 多 checkpoint、协议不一致、六类/七类混用、报告型与原始 seed 混用、缺失期望种子、LaTeX 转义、无 torch 导入、曲线 attempt 链。
