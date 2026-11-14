# ReCA-SAM3

ReCA-SAM3：RemoteCLIP-guided cross-class feature aggregation for open-vocabulary remote sensing semantic segmentation.

当前实现以 DLRSD 作为训练集、LoveDA 七类验证集，只支持 semantic 模式。SAM3 backbone、encoder、Pixel Decoder 和 semantic head 冻结；RemoteCLIP 图文分支默认做 attention 微调；`core.feature_aggregation` 完整训练。

## 环境与权重

依赖见 `requirements.txt`。仓库不包含数据集和预训练权重。默认路径：

* SAM3：`weights/sam3.pt`
* RemoteCLIP：`weights/RemoteCLIP-ViT-L-14.pt`
* BPE：`assets/bpe_simple_vocab_16e6.txt.gz`

机器本地路径可复制 `configs/paths.local.example.py` 为 `configs/paths.local.py` 后覆盖，不要把真实凭据或机器绝对路径提交。

## DLRSD 数据准备

将官方 OVRSISBench 的 `DLRSD.zip` 放在 `data/datasets/DLRSD/`，在项目根目录运行：

```bash
python tools/datasets_converters/dlrsd.py data/datasets/DLRSD --overwrite
```

保留压缩包内的官方划分：5601 张训练、1401 张验证；生成 `img_dir/{train,val,all}`、`ann_dir/{train,val,all}`、`splits/*.txt` 和 `processing_metadata.json`。读取 `D2masks`，直接保留 0..16 标签（0 是 airplane），不再减 1，不重新划分。原始 JPEG 字节与压缩包保留。

`--overwrite` 仅替换脚本生成的目录和元数据，用于更新之前的自定义划分；转换失败时原输出保留。`all` 为 7002 张训练与验证样本的并集，不是独立测试集。压缩包也可使用绝对路径，输出目录默认与压缩包所在目录一致；`-o` 可指定输出目录。

## 训练

```bash
python tools/train.py configs/experiments/main/dlrsd_loveda_full.py
```

查看最终有效配置（包含默认值、CLI 覆盖、seed、路径、阈值）：

```bash
python tools/train.py configs/experiments/main/dlrsd_loveda_full.py --print-config
```

继续训练必须使用原实验目录和原 `config.json`：

```bash
python tools/train.py configs/experiments/main/dlrsd_loveda_full.py \
  --resume-from work_dirs/main/dlrsd_loveda_full/latest.pth
```

短调试入口 `configs/experiments/dev/dlrsd_loveda_4k.py` 的验证是截断的（`val_max_iters=500`），只能产生 partial/draft 结果。

验证与推理仅解码学生路径，每个提示块执行一次分割头。原始 SAM3 教师路径仅在训练需要蒸馏监督时执行。

## 独立评测

```bash
python tools/train.py configs/test/loveda.py --eval-only \
  --load-model-from work_dirs/main/dlrsd_loveda_full/best.pth
```

独立评测写出 `results/<eval_id>.json`，不会重新打开或截断训练 `metrics.jsonl`。原生评测记录需要 checkpoint step；若权重文件没有训练元数据，显式传 `--eval-iter`。

## 实验记录

每个训练目录包含：

```text
config.json
metrics.jsonl
results/<eval_id>.json
best.pth
latest.pth
iter_XXXXXXX.pth
visualizations/   # 可选
wandb/            # 可选离线文件
```

* W&B 默认离线：`logging.wandb.enabled=True, mode="offline"`；失败只警告并关闭本次 W&B。
* 本地 JSONL 是权威记录，W&B 只记录主要损失、加权贡献、lr 范围、显存、耗时和整体验证指标。
* LoveDA 只使用七类口径（含 background）；`monitor="semantic.miou"`，正式入口完整验证。
* 标准结果格式见 [docs/results_schema.md](docs/results_schema.md)，模板见 [docs/results_template.json](docs/results_template.json)。

## 结果汇总与论文导出

```bash
# 标准结果收集
python tools/collect_experiments.py work_dirs/paper \
  --output paper/generated/data/results_long.csv

# 外部 CSV 导入（input-unit 必填；mapping / protocol 显式提供）
python tools/import_results.py --input external/baseline.csv \
  --mapping external/mapping.json --protocol external/protocol.json \
  --input-unit percent --output-dir work_dirs/paper/imported

# 严格导出论文表格和曲线
python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated

# 只导出表格，不依赖 matplotlib
python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated --tables-only
```

导出器只依赖 Python 标准库；只有曲线导出才导入 matplotlib。正式模式要求完整评测与完整协议元数据；`--allow-incomplete` 只生成明确标记的 draft。

## 配置结构

```text
configs/_base_/model/reca_sam3.py
configs/_base_/loss/semantic.py
configs/_base_/optimizer/reca_sam3_adamw.py
configs/_base_/schedule/
configs/_base_/runtime.py
configs/_base_/logging.py
configs/_base_/visualization.py
configs/datasets/
configs/protocols/dlrsd_loveda_semantic.py
configs/experiments/main/dlrsd_loveda_full.py
configs/experiments/dev/dlrsd_loveda_4k.py
configs/experiments/ablation/
configs/test/
configs/paths.local.example.py
configs/sweeps/
```

64 条 RemoteCLIP 模板集中在 `data/prompt_templates.py`。配置指纹由规范化有效配置生成，排除输出目录、设备、日志、可视化、保存/验证间隔、workers 和数据路径等运行字段。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用 `tests/fixtures/` 中的合成数据，明确不是真实成绩。覆盖有效配置、LoveDA 七类标签空间、schema、导入导出、多种子统计、结果写入与曲线 attempt 链。不运行模型前向或 GPU 训练。

## 命名迁移

工程整理后的模块、配置、张量、日志和入口命名见 [docs/migration_reca_sam3.md](docs/migration_reca_sam3.md)。旧训练权重不能直接恢复或加载，新实验使用新目录；不提供旧参数映射或兼容别名。
