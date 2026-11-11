# ReCA-SAM3 设计说明

适用分支：`master`
项目仓库：`jk-jin/ovrs-sam3`
当前任务：开放词汇遥感语义分割

> 本文只描述当前实现。代码与本文不一致时以代码为准；旧实验设计不在此保留。

## 1. 项目目标

ReCA-SAM3 接收遥感图像与当前数据集的类别名称，输出每个类别的像素级分割 logits。系统组合三类能力：

* SAM3 提供多尺度图像特征、文本提示编码、类条件 transformer encoder 和冻结分割解码器。
* RemoteCLIP 提供面向遥感场景的局部图文对齐与固定文本模板。
* RemoteCLIP-guided feature aggregation 在 36×36 上聚合提示特征，再由 Semantic–Detail Fusion 生成 72×72 解码输入。

当前只实现 semantic 模式，不支持实例分割、hybrid 模式或非空几何提示训练。

整体流程：

```text
图像与类别名称
  ├─ SAM3 backbone → FPN 288/144/72
  ├─ SAM3 文本编码器 + 6 层 transformer encoder
  │    → prompt cross-attention
  │    → prompt_conditioned_features_72 [B, P, 256, 72, 72]
  └─ RemoteCLIP
       ├─ 504×504 图像 → 36×36 dense feature
       └─ 每提示 64 个模板 → template text feature

prompt_conditioned_features_72
  → 双线性下采样 → initial_feature_36
  → 作为 feature_state_36 初值

RemoteCLIP template scores + dense feature
  → RemoteCLIPGuidanceEmbedding
  → clip_guidance_36 [B, P, 256, 36, 36]
  → 作为 semantic_state_36 初值

feature_state_36 + semantic_state_36
  → 4 个聚合块（cross → intra → cross → intra）
  → feature_output_norm
  → aggregated_features_36 [B, P, 256, 36, 36]

按 prompt_chunk_size 逐提示块：
  原始 SAM3 reference path（训练蒸馏或 reference gate 启用时，no_grad）
    prompt_conditioned_features_72
      → 冻结 Pixel Decoder + semantic head
      → sam3_teacher_logits / sam3_reference_logits（detached）

  学生：
    aggregated_features_36_chunk
      + prompt_conditioned_features_72_chunk
      + SAM3 FPN72
      → SemanticDetailFusion（语义/细节双支路、四方向共享卷积）
      → decoder_features_72 [B×P_chunk, 256, 72, 72]
      → 替换 Pixel Decoder 最后一层 FPN 输入
      → 冻结 Pixel Decoder + semantic head（梯度开启）
      → final_logits_chunk
```

## 2. 张量约定

| 记号 | 含义 |
| --- | --- |
| `B` | batch 中的图像数 |
| `P` | 逗号展开后的提示数 |
| `P_chunk` | 当前提示块中的提示数 |
| `M` | 合并后的语义类别数 |
| `K` | 每提示 RemoteCLIP 模板数，固定 64 |
| `D` | 特征通道数，固定 256 |
| `D_clip` | RemoteCLIP 投影维度，ViT-L/14 为 768 |

注意力沿提示轴 `P` 聚合，`P` 不是语义类别数 `M`。同义提示共享原始标签，推理时按像素最大值合并回 `M`。

主要张量：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `backbone_fpn` | `[B, 256, 288/144/72, ·]` | 顺序固定为 288、144、72 |
| `prompt_conditioned_features_72` | `[B, P, 256, 72, 72]` | 完整 encoder + 前置 prompt cross-attention |
| `remoteclip_feat_map` | `[B, 768, 36, 36]` | RemoteCLIP dense image feature |
| `clip_template_scores_36` | `[B, P, 64, 36, 36]` | 局部图文相似度图 |
| `clip_guidance_36` | `[B, P, 256, 36, 36]` | 纯 RemoteCLIP guidance，作为语义流初值 |
| `initial_feature_36` | `[B, P, 256, 36, 36]` | 下采样 encoder 特征，作为特征流初值 |
| `semantic_state_36` | `[B, P, 256, 36, 36]` | 聚合中的语义流 |
| `feature_state_36` | `[B, P, 256, 36, 36]` | 聚合中的特征流 |
| `aggregated_features_36` | `[B, P, 256, 36, 36]` | 最终特征流 |
| `decoder_features_72` | `[B×P_chunk, 256, 72, 72]` | SemanticDetailFusion 输出 |
| `final_logits` | `[B, P_chunk, 288, 288]` | 学生 logits |
| `sam3_teacher_logits` | `[B, P_chunk, 288, 288]` | 训练蒸馏 teacher，detached |
| `sam3_reference_logits` | `[B, P, 288, 288]` | 推理时原始 SAM3 reference 路径 |
| `raw_reference_prompt_score_map` | `[B, P, H, W]` | reference logits 的 sigmoid |
| `raw_reference_score_map` | `[B, M, H, W]` | reference 概率按类别合并 |

## 3. SAM3 分支

* 输入 1008×1008，ViT patch 14，token grid 72×72。
* SimpleFPN 产生 288/144/72/36 四级，`scalp=1` 丢弃 36 级，主路径保留前三尺度。
* backbone 在训练中冻结并 `eval()`；图像特征在 `no_grad()` 中计算并 detach。
* 类别名称按逗号拆分为提示；展开后的提示按 `prompt_chunk_size` 分块。
* 冻结的文本编码器与 6 层 transformer encoder 在 `no_grad()` 中为每个图像-提示对生成特征。
* 完整 encoder 后执行一次 prompt cross-attention，仍在 `no_grad()` 中。
* 所有提示块按原顺序拼接为 `prompt_conditioned_features_72`。

## 4. RemoteCLIP guidance

`RemoteCLIPGuidanceEmbedding` 生成 36×36 语义引导：

1. 64 个模板文本特征与 dense 图像特征分别沿通道 L2 归一化，逐像素余弦相似度乘 20。
2. 64 通道模板分数经 1×1 Conv、GroupNorm、GELU 得到 `feature_dim=256` 中间特征。
3. 中间特征与 dense RemoteCLIP 特征分别逐像素 L2 归一化，拼接后经 1024→256 1×1 Conv、GroupNorm、GELU。
4. 两路 256 通道特征再次分别 L2 归一化，拼接为 512 通道，经两个 3×3 Conv（256→256），每层后接 GroupNorm 和 GELU。
5. 输出 `clip_guidance_36`，直接作为 `semantic_state_36`，不注入 SAM3 FPN。

文本缓存规则：

* RemoteCLIP 文本分支完全冻结时可缓存 detach 特征。
* 文本分支可训练且全局梯度开启时，每个训练 step 重新编码并保留计算图。
* 验证在 `no_grad()` 中可复用当前权重缓存。
* 不以模块 `training` 属性判断是否需要梯度。

## 5. RemoteCLIP-guided feature aggregation

`RemoteCLIPGuidedAggregation` 在 36×36 上同时维护特征流与语义流。

### 5.1 初始化与指导

* `initial_feature_36 = bilinear(prompt_conditioned_features_72)`。
* `semantic_state_36 = clip_guidance_36`。
* 固定图像指导：
  * score 类内：RemoteCLIP 中间层 `[7, 15]`（第 8、16 层）沿通道拼接，经共享 1×1 Conv 到 256 通道，每个前向只算一次，再广播到提示维。
  * feature 类内：`backbone_fpn[2]`（72×72）双线性下采样到 36×36，保留 256 通道，广播到提示维。
* 两种指导都是必需输入；缺少索引 7/15、原生通道不符或空间尺寸不符时直接报错。

### 5.2 块顺序

`num_blocks=4`，按 cross-class → intra-class 交替，从 cross-class 开始；奇数块按该顺序截断。块之间不共享参数。

| 块 | 执行顺序 |
| --- | --- |
| cross-class | semantic 更新 → feature 更新 |
| intra-class | semantic regular → semantic shifted → feature regular → feature shifted |

* 每步独立 pre-norm、Q/K/V 投影、注意力和直接残差。
* cross-class attention 的 Q/K 拼接 `[semantic_state, feature_state]`（512 通道）。
* intra-class attention 的 Q/K 拼接 `[semantic_state, feature_state, guidance]`（768 通道）。
* Value 只来自目标流；图像指导不进入 Value，也不做残差注入。
* 类内使用 12×12 窗口、6 像素移位，保留窗口相对位置偏置与 shift mask。
* 块末尾无 FFN。全部块结束后仅对 feature 流执行一次通道 LayerNorm。

### 5.3 Semantic–Detail Fusion

`SemanticDetailFusion` 在 72×72 执行一次：

* 输入：`aggregated_features_36`、`prompt_conditioned_features_72`、`sam3_fpn_72`。
* 特征流先双线性插值到 72×72。
* 四路独立 1×1 Conv + GroupNorm(8,256) 投影，分别用于 semantic 聚合特征、detail 聚合特征、encoder 特征和 SAM3 FPN。
* `semantic_input = semantic_proj(aggregated_72) + encoder_proj(prompt_conditioned_72)`。
* `detail_input = detail_proj(aggregated_72) + broadcast(fpn_proj(sam3_fpn_72))`。
* 每条支路使用独立的 256 通道卷积块：3×3 Conv → GN → GELU → 1×1 Conv → GN。
* `aggregate_orientations`：同一块在 0°/90°/180°/270° 下调用，逆旋转回输入坐标后等权平均，`output = input + 0.25 * Σ delta`。
* 两支路等权相加，经 256→256 `fusion_out_proj` 输出 `decoder_features_72`。
* 无融合系数、门控、零初始化、输出归一化或额外 encoder 残差。整个模块使用一次 non-reentrant checkpoint。

## 6. 冻结与梯度边界

* Pixel Decoder 与 semantic head 参数冻结并保持 `eval()`；学生和教师复用同一对象。
* 教师路径：`prompt_conditioned_features_72` → Pixel Decoder → semantic head，全程 `no_grad()`，只保留 detached logits。
* 学生路径：`decoder_features_72` → 同一 Pixel Decoder → semantic head，不使用 `no_grad()`、`detach()` 或 `enable_grad()`。
* 冻结参数不更新，但梯度可穿过冻结卷积回传到 `SemanticDetailFusion` 和聚合模块。
* `core.feature_aggregation` 完整训练；RemoteCLIP 图文默认 `attention` 微调并保持 `eval()`。

## 7. 训练设计

### 7.1 默认超参数

有效基线以 `configs/experiments/main/isaid_loveda_full.py` 合并结果为准：

* AdamW 基础学习率 `1e-4`；RemoteCLIP 图文乘 `0.01`；normalization 无 weight decay。
* 梯度裁剪 `0.01`；AMP 开启。
* warmup 1000 步（0.1 → 1.0），余弦衰减 19000 步，总计 20000 步。
* 每 2000 步保存与验证；`val_max_iters=None` 表示完整验证。
* batch size 2；训练 workers 8；验证 workers 2。
* 64 个固定 RemoteCLIP 文本模板。
* dropout 0.1；8 注意力头；窗口 12；移位 6。

### 7.2 数据增强

训练：短边确定缩放 1008 → 随机裁剪 1008×1008 → SSD 颜色增强 → 0.5 概率水平翻转 → Normalize(0.5,0.5)。
验证：直接 Resize 1008×1008，`keep_ratio=False`。TTA 默认关闭。

### 7.3 损失

```python
total_loss = (
    seg_bce_weight * loss_seg_bce
    + seg_dice_weight * loss_seg_dice
    + reference_distill_weight * loss_reference_distill_bce
    + present_ignore_suppression_weight
      * loss_present_ignore_suppression_bce
)
```

默认 `seg_bce_weight=1.0`、`seg_dice_weight=0.0`、`reference_distill_weight=0.5`、`reference_boundary_width=3`、`present_ignore_suppression_weight=0.01`。

* 主 BCE：所有 `label != 255` 的像素等权，全局均值。
* Dice：只对存在的提示计算；默认关闭。
* Reference 蒸馏：冻结 SAM3 head 的 soft probability；只监督存在提示的全部有效像素，并额外包含 GT 外侧宽度 `reference_boundary_width` 的 ignore 边界环。
* 存在类别忽略区域抑制：只对学生未过滤 logits、目标为 0，在存在类别的 ignore 区域计算软加损失；同义提示共享类别权重。
* 日志键：`loss_seg_bce`、`loss_seg_dice`、`loss_reference_distill_bce`、`loss_reference_distill_weighted`、`loss_present_ignore_suppression_bce`、`loss_present_ignore_suppression_weighted`、`total_loss`。

### 7.4 显存设计

* 聚合在全部提示上执行一次。
* 高分辨率按提示块计算；每块立即 loss/backward。
* detached leaf proxy 累积高分辨率梯度，全部块完成后一次回传聚合图。
* optimizer/scaler/scheduler 每 batch 只更新一次；不使用 `retain_graph=True`。

## 8. 推理与评测

推理输出：

* `prompt_logits`：`[B, P, H, W]` 学生提示 logits。
* `raw_prompt_score_map`：未过滤学生提示概率。
* `raw_final_score_map`：未过滤学生类别概率。
* `final_score_map`：双门过滤后的学生类别概率。
* `final_pred`：`final_score_map.argmax(dim=1)`。
* `final_logits`：未过滤模型 logits。
* `sam3_reference_logits`、`raw_reference_prompt_score_map`、`raw_reference_score_map`：按需解码的原始 SAM3 reference 路径。

双门在原始类别空间执行：

| 参数 | 判定数据 | 低于阈值 | 关闭 |
| --- | --- | --- | --- |
| `reference_prob_threshold` | 原始 SAM3 reference 概率 | 学生最终分数置 0 | `None` |
| `prediction_prob_threshold` | 学生原始 sigmoid 概率 | 学生最终分数置 0 | `None` |

保留条件是两道已启用的门都通过；等于阈值保留。所有提示先合并到原始类别，再统一过滤。

### 8.1 LoveDA 七类口径

LoveDA 评测类别顺序固定为 background、building、road、water、barren、forest、agricultural。

* `configs/datasets/eval/loveda.py` 使用 `reduce_zero_label=False`、`background_cfg.enabled=True`、`exclude_from_forward=True`。
* 标签空间只执行一次变换：原始 0 保留为评测背景；模型前向删除背景并把前景 1..6 映射为 0..5；评测后处理把前景预测 +1，低分位置回填背景 0。
* 不再叠加 `reduce_zero_label`，否则会二次删除背景并平移类别索引。
* `train_cfg.monitor="semantic.miou"`、`monitor_mode="max"`；训练选择与独立评测同一七类口径。
* 完整评测 `val_max_iters=None`。显式截断的调试结果为 partial，不进入正式表格。
* `evaluation.metric_groups=[]`；不再生成六类前景组。逐类指标包含背景。
* 缺失类：IoU 分母为 0 时该逐类指标记为 `null`；汇总仍按当前 evaluator 的零并集排除政策，并记录有效类别数。

### 8.2 TTA

TTA 只支持 scale=1.0 与空间翻转。多个视图分别生成学生与所需的 reference 提示概率，逆翻转并对齐后分别平均 `raw_prompt_score_map` 与 `raw_reference_prompt_score_map`，再各自合并到原始类别，最后统一执行一次双门过滤。禁止平均已过滤结果。

## 9. Checkpoint、日志与标准结果

### 9.1 训练目录

```text
work_dirs/<experiment>/
  config.json              # 有效配置 + 运行元数据
  code.patch               # 可选，tracked diff
  metrics.jsonl            # 完整本地标量事件
  results/<eval_id>.json   # 每次完成验证/评测的标准结果
  iter_XXXXXXX.pth
  latest.pth
  best.pth
  visualizations/          # 可选
  wandb/                   # 可选离线文件
```

* `config.json` 包含 git commit、dirty 标记、未跟踪代码标记、实际 seed、环境版本和展开后的有效配置。
* 新训练要求新目录或空目录；已有 checkpoint 时拒绝混跑。
* 恢复训练必须使用原目录和原 `config.json`，并校验语义配置指纹一致。路径、日志、可视化等非语义差异写入 JSONL meta 事件。
* 每次进程使用新的 `attempt_id`。checkpoint `extra` 保存 attempt id；恢复时记录 `resume_from_step` 与父 attempt。
* checkpoint 容器 schema 仍为 4；W&B 状态不写入 checkpoint。

### 9.2 日志

本地 JSONL 与 W&B 消费同一份规范化标量事件：

* JSONL 保存完整训练损失、加权贡献、各参数组学习率、步数、耗时、显存、验证整体与逐类指标。
* W&B 默认 `enabled=True, mode="offline"`；默认只记录主要损失、加权贡献、lr 范围、显存、耗时和整体验证指标。
* W&B 逐类与额外指标由 `log_per_class` / `log_extra` 控制，默认关闭。
* W&B 的 import/init/log/define_metric/finish 普通异常只警告一次并关闭本次 W&B；不吞模型或本地写盘错误，不捕获 `KeyboardInterrupt`。
* 日志与 W&B 初始化在保存/恢复 Python、NumPy、Torch CPU/CUDA RNG 状态的隔离范围内执行；恢复训练仍最后恢复 checkpoint RNG。
* 每次进程新建离线 run，`resume="never"`；run 名来自 `experiment.name` 并附 seed，group 来自配置组或 method id。
* 独立评测不打开或截断训练 `metrics.jsonl`，只写 `results/<eval_id>.json`。

### 9.3 标准结果

标准结果格式独立于 checkpoint，`schema_version=1`，详见 `docs/results_schema.md` 与 `docs/results_template.json`。

* 每次完整验证写 `role="validation"`；独立评测写 `role="evaluation"`。
* `best.pth` 只在 monitor 严格改善时更新；checkpoint `extra` 记录 monitor、mode、value、选择数据集/划分和是否 best。
* 结果文件包含实际 checkpoint step、sha256、model selection 划分与指标、完整评测协议、类别和样本计数。
* `is_full` 由实际处理样本数与数据集长度比较得到，不能仅凭 `val_max_iters=None` 推断。
* 原子写入；只有完整完成并通过 schema 校验才写 `status="completed"`。
* 同一权重多次评测生成不同 eval_id，不覆盖。
* 汇总与论文导出工具只读标准 JSON 和 JSONL，不依赖 W&B、GPU、torch、SAM3 或模型包。

## 10. 配置与入口

### 10.1 目录职责

| 路径 | 职责 |
| --- | --- |
| `configs/_base_/model/reca_sam3.py` | SAM3/RemoteCLIP/聚合/冻结 |
| `configs/_base_/loss/semantic.py` | 损失权重与边界 |
| `configs/_base_/optimizer/` | 优化器与参数组 |
| `configs/_base_/schedule/` | 20K 与 4K 迭代计划 |
| `configs/_base_/runtime.py` | seed、device、AMP、work_dir |
| `configs/_base_/logging.py` | 控制台、JSONL、W&B |
| `configs/_base_/visualization.py` | 固定样本可视化 |
| `configs/datasets/` | 数据路径、类别、标签空间 |
| `configs/protocols/isaid_loveda_semantic.py` | 训练/评测完整协议 |
| `configs/experiments/main/isaid_loveda_full.py` | 论文主实验入口 |
| `configs/experiments/dev/isaid_loveda_4k.py` | 短调试入口（partial 验证） |
| `configs/experiments/ablation/` | 已支持的参数消融入口 |
| `configs/test/` | 各数据集独立评测 |
| `configs/paths.local.example.py` | 机器路径覆盖示例 |
| `configs/sweeps/` | 外部 sweep 入口 |

根配置职责：`experiment`、`training`、`model`、`loss`、`train_dataloader`/`val_dataloader`、`optimizer`/`param_scheduler`/`train_cfg`、`evaluation`、`runtime`、`logging`、`visualization`。

* `runtime` 是 seed/device/AMP/work_dir 的唯一权威来源；model 与 train_cfg 不保存可编辑副本。
* 64 条模板集中在 `data/prompt_templates.py`；最终 `config.json` 展开实际字符串。
* `evaluation.background_prob_threshold`、`evaluation.postprocess.reference_prob_threshold`、`evaluation.postprocess.prediction_prob_threshold` 语义不同，不能合并；关闭用 `None`。
* 解析顺序：继承 → `--cfg-options` → 专用 CLI（`--seed`、`--device`、`--work-dir`）→ 默认值 → 校验 → 打印/保存/构建。专用 CLI 优先于同名 `--cfg-options`。
* `--print-config` 输出最终有效配置，包含实际 seed、路径、默认值和阈值。
* 配置指纹由规范化有效配置生成；排除输出目录、设备、日志、可视化、显示名称、保存/验证间隔、workers 和数据路径等运行字段。batch size、模型、损失、优化器、调度、训练/评测协议和阈值参与指纹。
* 未知字段、类型/范围错误、继承循环、scheduler 与 `max_iters` 不一致均直接报错。

### 10.2 命令

```bash
python tools/train.py configs/experiments/main/isaid_loveda_full.py
python tools/train.py configs/experiments/main/isaid_loveda_full.py --print-config
python tools/train.py configs/experiments/main/isaid_loveda_full.py \
  --resume-from work_dirs/main/isaid_loveda_full/latest.pth
python tools/train.py configs/test/loveda.py --eval-only \
  --load-model-from work_dirs/main/isaid_loveda_full/best.pth

python tools/collect_experiments.py work_dirs/paper \
  --output paper/generated/data/results_long.csv
python tools/import_results.py --input external/baseline.csv \
  --mapping external/mapping.json --protocol external/protocol.json \
  --input-unit percent --output-dir work_dirs/paper/imported
python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated
python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated --tables-only
```

## 11. 主要实现文件

| 文件 | 职责 |
| --- | --- |
| `models/reca_sam3.py` | ReCASAM3 外层 wrapper |
| `models/sam3_image.py` | 提示分块、缓存、encoder、逐 chunk 解码 |
| `models/feature_aggregation.py` | 36×36 聚合、最终 feature norm、72×72 融合接口 |
| `models/aggregation_attention.py` | cross/intra 块与单 Value 注意力 |
| `models/remoteclip_guidance.py` | 64 模板分数、两次 L2 融合、空间增强 |
| `models/semantic_detail_fusion.py` | 语义—细节双支路四方向融合 |
| `models/maskformer_segmentation.py` | SAM3 prompt attention、Pixel Decoder、semantic head |
| `models/adapters/semantic_adapter.py` | 双门、提示合并、推理输出 |
| `losses/semantic_criterion.py` | 主 BCE、Dice、reference 蒸馏、忽略区域抑制 |
| `engine/config.py` | 配置加载、默认值、校验、指纹 |
| `engine/trainer.py` | 流式 chunk loss/backward、验证、结果写入 |
| `engine/experiment_hooks.py` | 统一事件、JSONL、离线 W&B |
| `engine/result_writer.py` | config.json 与标准结果 JSON |
| `engine/evaluator.py` | 七类指标、逐类统计、TTA |
| `engine/checkpoint.py` | 原子 checkpoint、best 选择、恢复校验 |
| `utils/experiment_results.py` | 无 torch 依赖的 schema、单位、统计与转义 |
| `tools/import_results.py` | 外部 CSV/JSON 导入 |
| `tools/export_paper_results.py` | 论文表、曲线与导出报告 |

## 12. 不变量与限制

1. 提示块覆盖完整、无重复、按原顺序拼接。
2. SAM3 encoder / aggregation / RemoteCLIP 网格固定 72×72 / 36×36 / 36×36。
3. 特征通道固定 256；模板固定 64；聚合默认 4 块。
4. `clip_guidance_36` 完全由 RemoteCLIP 模板分数与 dense 特征生成，不注入 SAM3 FPN。
5. 两次拼接前分别执行逐像素通道 L2 归一化。
6. 固定保留 score 类内 RemoteCLIP 中间层指导与 feature 类内 FPN36 指导；缺失即报错。
7. 聚合先在全部提示上执行，再按 chunk 高分辨率解码；聚合不能放进 chunk 循环。
8. 教师使用 `prompt_conditioned_features_72`，必须 `no_grad()` 并 detach；学生不得使用 `no_grad()`/`detach()`/`enable_grad()`。
9. SemanticDetailFusion 只在 72×72 执行一次；每条支路四方向共享同一卷积块；输出替换 Pixel Decoder 最后一层 FPN 输入。
10. 每个 batch 只执行一次 optimizer/scaler/scheduler 更新；不使用 `retain_graph=True`。
11. `backbone_fpn` 顺序固定为 `[288, 144, 72]`；FPN 按图像投影后按提示广播。
12. 最终 logits 由冻结 SAM3 `semantic_seg_head` 生成。
13. LoveDA 标签只执行一次背景删除/索引平移；不得叠加 `reduce_zero_label` 与 `background_cfg` 的删除。
14. 配置指纹与恢复校验覆盖语义设置；日志、路径、可视化差异不改变指纹。
15. 标准结果缺少必需字段、未知主版本、单位不明或指标越界时拒绝导入/导出。

当前限制：

* 只支持 semantic 模式；不支持非空几何提示。
* 不支持动态空间尺寸或多尺度 TTA。
* 旧命名训练权重不能用于 `--resume-from` 或直接 `--load-model-from`；新实验使用新目录。
* 导出器正式模式要求完整评测与完整协议元数据；不完整数据只能生成 draft。
* 本次只做静态与合成数据验证；未验证模型前向、跨 GPU 数值等价或正式训练。不同 GPU 内核之间不承诺逐位一致。

## 13. 本次命名重构的非兼容变更

* `SAM3Segmentor` → `ReCASAM3`；`core.encoder_refiner` → `core.feature_aggregation`。
* `ClipScoreEmbedding` → `RemoteCLIPGuidanceEmbedding`。
* `ClassConditionedEncoderRefiner` → `RemoteCLIPGuidedAggregation`。
* `RefinerInterBlock` → `CrossClassAggregationBlock`；`RefinerIntraBlock` → `IntraClassAggregationBlock`。
* `ClassScoreAttention` → `CrossClassAttention`；`WindowScoreAttention` → `WindowSpatialAttention`。
* `DecoderInputFusion` → `SemanticDetailFusion`；四方向聚合函数 → `aggregate_orientations`。
* 配置：`encoder_refiner_cfg` → `aggregation`；`openclip_cfg` → `remoteclip`；`freeze_cfg` → `freeze`；`criterion_cfg` → 根 `loss`。
* 参数路径整体变化，旧 checkpoint 不能严格恢复。checkpoint 容器 schema 仍为 4。
* 不提供旧参数映射、自动转换或兼容别名；迁移对照见 `docs/migration_reca_sam3.md`。

CPU 回归测试：`python -m unittest discover -s tests -v`。测试覆盖有效配置、标签空间、schema、导入导出、多种子统计、结果写入与曲线 attempt 链；不运行模型前向或 GPU 训练。
