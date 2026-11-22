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
  → 2 个聚合块（每块先做类间注意力，再并行计算三个尺度的普通窗口注意力）
  → feature_output_norm
  → aggregated_features_36 [B, P, 256, 36, 36]

按 prompt_chunk_size 逐提示块：
  原始 SAM3 teacher path（仅训练需要蒸馏监督时，no_grad）
    prompt_conditioned_features_72
      → 冻结 Pixel Decoder + semantic head
      → sam3_teacher_logits（detached）

  学生：
    aggregated_features_36_chunk
      + prompt_conditioned_features_72_chunk
      + SAM3 FPN72
      → SemanticDetailFusion（语义/细节双支路、单方向卷积）
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
| `clip_guidance_36` | `[B, P, 256, 36, 36]` | 纯 RemoteCLIP guidance，固定参与每步 Q/K |
| `initial_feature_36` | `[B, P, 256, 36, 36]` | 下采样 encoder 特征，作为特征流初值 |
| `semantic_state_36` | `[B, P, 256, 36, 36]` | 聚合中保持初值的语义引导 |
| `feature_state_36` | `[B, P, 256, 36, 36]` | 聚合中的特征流 |
| `aggregated_features_36` | `[B, P, 256, 36, 36]` | 最终特征流 |
| `decoder_features_72` | `[B×P_chunk, 256, 72, 72]` | SemanticDetailFusion 输出 |
| `final_logits` | `[B, P_chunk, 288, 288]` | 学生 logits |
| `sam3_teacher_logits` | `[B, P_chunk, 288, 288]` | 训练蒸馏 teacher，detached |

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

`RemoteCLIPGuidedAggregation` 在 36×36 上更新 encoder feature，RemoteCLIP score embedding 作为固定语义引导。

### 5.1 初始化与指导

* `initial_feature_36 = bilinear(prompt_conditioned_features_72)`，作为 feature state 初值。
* `semantic_state_36 = clip_guidance_36`，聚合全过程保持相同张量，不执行任何 score embed 注意力更新；不 detach、不冻结，梯度通过 Q/K 回传到 embedding 与 RemoteCLIP。
* SAM3 类内指导：`backbone_fpn[2]`（72×72）双线性下采样到 36×36，保留 256 通道，广播到提示维；缺失或尺寸不符直接报错。
* RemoteCLIP 中间层 `[7, 15]` 的提取、缓存和调试输出保留，但不传入 aggregation，不构建对应投影或指导更新分支。

### 5.2 类间与多尺度类内注意力

默认 `num_blocks=2`。每块先在 36×36 网格的每个位置沿完整提示轴计算类间注意力。将更新量乘以本层的类间残差系数，再加到 encoder feature 上。三个尺度的类内注意力都从这份更新后的特征出发。

| 特征尺度 | 并行分支 | 每个分支的 12×12 窗口数 |
| --- | --- | --- |
| 36×36 | 普通窗口 | 9 |
| 24×24 | 普通窗口 | 4 |
| 12×12 | 全图窗口 | 1 |

* 保留 36×36 特征，从中分别双线性插值得到 24×24 和 12×12。每个尺度各做一次普通窗口注意力。12×12 的全图窗口只计算一次，全部 144 个位置可以相互关注。
* 保留 36×36 分支的更新量，将 24×24 和 12×12 两路的更新量分别上采样到 36×36。三路各乘以本尺度的可学习标量，逐元素求和，再一次性加回类间注意力更新后的 encoder feature。所有双线性插值均使用 `align_corners=False`。
* 每层独立学习四个残差系数，分别对应类间注意力和三个尺度。`model.aggregation.residual_weight_init` 统一设置初始值，默认 `0.1`，接受有限数值。主路径的 encoder feature 系数固定为 1。恢复权重时加载已训练的系数值。
* 类间注意力用 score embedding 和当前 encoder feature 构造查询与键。类内注意力还拼接对应尺度的 SAM3 FPN 引导。所有注意力的值都取自 encoder feature。两份引导的多尺度特征在一次聚合前向中生成一次，由各层复用。
* `model.aggregation.share_window_weights_across_scales=True`（默认）时，同一层三个尺度共用一套预归一化、查询/键/值投影、输出投影和相对位置偏置。设为 `False` 时，每个分支独立学习上述全部参数，每层共三套。两种模式下，各尺度的残差系数都独立训练。
* 所有聚合块结束后，对特征执行一次通道 LayerNorm。

### 5.3 Semantic–Detail Fusion

`SemanticDetailFusion` 在 72×72 执行一次：

* 输入、四路独立 1×1 Conv + GroupNorm 投影及双支路组织保持原定义。
* 聚合特征双线性上采样至 72×72；语义支路与原始 encoder72 相加，细节支路与按图像投影并按提示广播的 SAM3 FPN72 相加。
* 每条支路的独立 256 通道卷积块只在原方向调用一次：3×3 Conv → GN → GELU → 1×1 Conv → GN；卷积结果残差回该支路输入。
* 支路输出各经 1×1 投影，等权相加，再经 `fusion_out_proj` 输出；无旋转、逆旋转或方向平均。
* 模块使用一次 non-reentrant checkpoint；学生梯度仍穿过冻结分割解码器。

## 6. 冻结与梯度边界

* Pixel Decoder 与 semantic head 参数冻结并保持 `eval()`；学生和教师复用同一对象。
* 教师路径：`prompt_conditioned_features_72` → Pixel Decoder → semantic head，全程 `no_grad()`，只保留 detached logits。
* 学生路径：`decoder_features_72` → 同一 Pixel Decoder → semantic head，不使用 `no_grad()`、`detach()` 或 `enable_grad()`。
* 冻结参数不更新，但梯度可穿过冻结卷积回传到 `SemanticDetailFusion` 和聚合模块。
* `core.feature_aggregation` 完整训练；RemoteCLIP 图文默认 `attention` 微调并保持 `eval()`。

## 7. 训练设计

### 7.1 默认超参数

有效基线以 `configs/experiments/main/dlrsd_loveda_full.py` 合并结果为准：

* AdamW 基础学习率为 `1e-4`。RemoteCLIP 图文参数组使用 `0.01` 倍率，归一化层的权重衰减为 0。
* 梯度裁剪 `0.01`；AMP 开启。
* warmup 500 步（0.1 → 1.0），余弦衰减 9500 步，总计 10000 步。
* 余弦使用 `RelativeCosineAnnealingLR`，`min_lr_ratio=0.01`。主模块预热到 `1e-4` 后衰减至 `1e-6`，RemoteCLIP 图文参数组预热到 `1e-6` 后衰减至 `1e-8`。各组预热起点为对应目标的 0.1 倍。4K 调试计划采用相同规则。
* 每 1000 步保存并做完整验证（`val_max_iters=None`）。
* batch size 4；训练 workers 8；验证 workers 2。
* 64 个固定 RemoteCLIP 文本模板。
* dropout 为 0.1，使用 8 个注意力头和 12×12 普通窗口。

### 7.2 数据增强

训练：原图随机裁剪/整图分支 → Resize 1008×1008 → 随机直角旋转 → SSD 颜色增强 → 0.5 概率水平翻转 → Normalize(0.5,0.5)。

* `RandomResizedCrop(size=(1008,1008), scale=(0.6,1.0), prob=0.7)`：70% 概率从原图采样正方形裁剪，面积在原图的 60%～100% 之间；30% 概率保留完整图像。两条分支最终都只做一次缩放，输入始终为 1008×1008。非正方形源图的裁剪面积范围按能放入的最大正方形截断，极小尺寸按整数像素取整，不引入 padding。
* `RandomRotate90`：等概率选择 0°/90°/180°/270°，不插值；与水平翻转组合覆盖八种朝向，每个样本只生成一个训练视图。
* 所有空间变换共享同一次随机参数，同步作用于 `image`、`raw_image`、`label_map` 和 `eval_label_map`；图像使用双线性插值，标签使用最近邻插值。SSD 颜色增强维持原参数，并同步作用于两路图像；仅 SAM3 的 `image` 执行最后的 Normalize。
* 存在提示由增强后的标签在损失分辨率下重新确定；随机采样使用现有 Python RNG，随训练 runtime RNG 状态一起恢复。

验证：直接 Resize 1008×1008，`keep_ratio=False`。TTA 默认关闭。

### 7.3 损失

```python
total_loss = (
    seg_bce_weight * loss_seg_bce
    + seg_dice_weight * loss_seg_dice
    + reference_distill_weight * loss_reference_distill_bce
)
```

默认 `seg_bce_weight=1.0`、`present_class_weight=0.8`、`seg_dice_weight=0.0`、`reference_distill_weight=0.5`。

* 主 BCE：每张图的每个提示先在所有 `label != 255` 的像素上取平均，类别内部不单独平衡正负区域；存在提示与不存在提示分别在各自组内取平均，然后以 `present_class_weight` 和 `1-present_class_weight` 合并两组损失，最后在 batch 的图像间平均。
* 提示存在性由损失分辨率下该图的 GT 标签确定；同义提示继承原类别存在性，仍按提示通道分别计数。每张图的有效像素数、存在/不存在提示数与组权重在完整提示集合上计算一次；所有 chunk 共享这些归一化统计，不能在 chunk 内重新计算组均值。
* `present_class_weight` 范围为 `[0,1]`，默认存在组占 0.8、不存在组占 0.2。某图全部类别存在时，仅使用存在组并赋全权重；全 ignore 图的主 BCE 为可反传的零，在 batch 平均中仍计为一张图。两组均非空时，调节参数只控制组占比，不随组内提示数量变化。
* Dice：只对存在的提示计算；默认关闭。
* Reference 蒸馏：冻结 SAM3 head 的 soft probability；只监督存在提示的有效标注像素（`label != 255`），全部 ignore 像素排除，不再构造边界环。不监督不存在提示。
* 蒸馏 BCE 在完整 batch 的所有存在图像-提示对的有效像素上取全局平均，所有 chunk 共用完整分母；随后乘固定 `reference_distill_weight`，默认 0.5。训练从头到尾使用同一个系数，不依赖步数，也没有独立调度器。固定权重为零或没有监督像素时跳过教师解码。
* 4K 调试入口沿用相同固定蒸馏规则；旧消融入口已删除。
* 日志键：`loss_seg_bce`、`loss_seg_dice`、`loss_reference_distill_bce`、`loss_reference_distill_weighted`、`reference_distill_weight`、`total_loss`。其中 `reference_distill_weight` 为当前 batch 的真实系数，不按 chunk 累加，也不按日志窗口平滑；JSONL 与离线 W&B 均记录。

损失配置入口为 `configs/_base_/loss/semantic.py`，例如：

```python
loss = dict(
    present_class_weight=0.8,
    reference_distill_weight=0.5,
)
```

原 `absent_class_weight`、四个蒸馏余弦调度配置项 `reference_distill_start_iter` / `reference_distill_end_iter` / `reference_distill_start_weight` / `reference_distill_end_weight`，以及 `reference_boundary_width` 已删除，继续使用时会拒绝配置。新损失和增强参数参与配置指纹，需使用新实验目录启动训练，不能从旧配置直接恢复。

### 7.4 显存设计

* 聚合在全部提示上执行一次。
* 高分辨率按提示块计算；每块立即 loss/backward。
* detached leaf proxy 累积高分辨率梯度，全部块完成后一次回传聚合图。
* optimizer/scaler/scheduler 每 batch 只更新一次；不使用 `retain_graph=True`。
* 新损失与相对调度参数参与配置指纹。使用新设计需启动新实验目录；相同新配置的 checkpoint 可完整恢复 optimizer 和 scheduler 状态，包括预热切换前后。

## 8. 推理与评测

推理输出：

* `prompt_logits`：`[B, P, H, W]` 学生提示 logits。
* `raw_prompt_score_map`：未过滤学生提示概率。
* `raw_final_score_map`：未过滤学生类别概率。
* `final_score_map`：学生预测概率过滤后的类别概率。
* `final_pred`：`final_score_map.argmax(dim=1)`。
* `final_logits`：未过滤模型 logits。

推理仅解码学生路径，每个提示块执行一次分割头，不生成原始 SAM3 reference 掩码。教师解码仅在训练需要蒸馏监督时执行；蒸馏关闭或没有监督像素时跳过。

`prediction_prob_threshold` 在原始类别空间执行：先将全部提示按类别逐像素取最大值合并，再将低于阈值的学生概率置 0，等于阈值保留；`None` 关闭。仅当前 batch metadata 的 `background_enabled=True` 时生效，无背景数据集跳过过滤。普通推理与 TTA 共用该规则。背景回填使用独立的 `evaluation.background_prob_threshold`。

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

TTA 只支持 scale=1.0 与空间翻转。每个视图仅解码学生路径，逆翻转并对齐后平均未过滤的 `raw_prompt_score_map`，再合并到原始类别并统一执行一次预测概率过滤。禁止平均已过滤结果。

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
| `configs/protocols/dlrsd_loveda_semantic.py` | DLRSD train → LoveDA val 完整协议 |
| `configs/experiments/main/dlrsd_loveda_full.py` | DLRSD 训练、LoveDA 七类验证入口 |
| `configs/experiments/dev/dlrsd_loveda_4k.py` | 短调试入口（partial 验证） |
| `configs/test/` | 各数据集独立评测 |
| `configs/paths.local.example.py` | 机器路径覆盖示例 |
| `configs/sweeps/` | 外部 sweep 入口 |

根配置职责：`experiment`、`training`、`model`、`loss`、`train_dataloader`/`val_dataloader`、`optimizer`/`param_scheduler`/`train_cfg`、`evaluation`、`runtime`、`logging`、`visualization`。

* `runtime` 是 seed/device/AMP/work_dir 的唯一权威来源；model 与 train_cfg 不保存可编辑副本。
* 64 条模板集中在 `data/prompt_templates.py`；最终 `config.json` 展开实际字符串。
* `evaluation.background_prob_threshold` 用于背景回填，`evaluation.postprocess.prediction_prob_threshold` 用于学生预测过滤；关闭均用 `None`。
* 解析顺序：继承 → `--cfg-options` → 专用 CLI（`--seed`、`--device`、`--work-dir`）→ 默认值 → 校验 → 打印/保存/构建。专用 CLI 优先于同名 `--cfg-options`。
* `--print-config` 输出最终有效配置，包含实际 seed、路径、默认值和阈值。
* 配置指纹由规范化有效配置生成；排除输出目录、设备、日志、可视化、显示名称、保存/验证间隔、workers 和数据路径等运行字段。batch size、模型、损失、优化器、调度、训练/评测协议和阈值参与指纹。
* 未知字段、类型/范围错误、继承循环、scheduler 与 `max_iters` 不一致均直接报错。

### 10.2 命令

```bash
python tools/train.py configs/experiments/main/dlrsd_loveda_full.py
python tools/train.py configs/experiments/main/dlrsd_loveda_full.py --print-config
python tools/train.py configs/experiments/main/dlrsd_loveda_full.py \
  --resume-from work_dirs/reca_sam3/full/seed_42/train/latest.pth
python tools/train.py configs/test/loveda.py --eval-only \
  --load-model-from work_dirs/reca_sam3/full/seed_42/train/best.pth

python tools/collect_experiments.py work_dirs/reca_sam3 \
  --output paper/generated/data/results_long.csv
python tools/import_results.py --input external/baseline.csv \
  --mapping external/mapping.json --protocol external/protocol.json \
  --input-unit percent --output-dir work_dirs/reca_sam3/imported
python tools/export_paper_results.py --root work_dirs/reca_sam3 \
  --spec paper/export_config.json --output paper/generated
python tools/export_paper_results.py --root work_dirs/reca_sam3 \
  --spec paper/export_config.json --output paper/generated --tables-only
```


DLRSD 预处理：`python tools/datasets_converters/dlrsd.py data/datasets/DLRSD --overwrite`。压缩包为该目录下的 `DLRSD.zip`，使用官方 `D2masks` 的 0..16 编号，不再减 1。脚本保留官方目录定义的训练与验证成员，生成训练配置可直接读取的目录、划分清单和处理元数据；`all` 包含训练样本。

训练仅保留 DLRSD：使用 `data/datasets/DLRSD/{img_dir,ann_dir}/train`，保留官方 OVRSISBench 划分（5601 train / 1401 val / 7002 all），17 类且不启用背景。LoveDA 验证沿用七类口径。iSAID 保留独立评测配置与数据处理脚本。

UAVid 采用合并车辆的七类标签：background、building、road、tree、low vegetation、car、human（0..6）。原始 moving/static car 的 ID 5/6 均映射为 5，human 由 7 映射为 6；忽略值保持 255。前向排除背景，使用六个类别提示。重新转换命令：`python tools/datasets_converters/uavid.py data/datasets/UAVid -o data/datasets/UAVid --overwrite`。脚本从原始八类 RGB/索引标签重建，清理旧 `img_dir`/`ann_dir`，保留原始数据，并写出七类映射元数据；独立评测协议更新为 `uavid_7class_merged_car_val_resize1008_v2`。

## 11. 主要实现文件

| 文件 | 职责 |
| --- | --- |
| `models/reca_sam3.py` | ReCASAM3 外层 wrapper |
| `models/sam3_image.py` | 提示分块、缓存、encoder、逐 chunk 解码 |
| `models/feature_aggregation.py` | 36×36 聚合、最终 feature norm、72×72 融合接口 |
| `models/aggregation_attention.py` | 统一类间/类内块与 encoder Value 注意力 |
| `models/remoteclip_guidance.py` | 64 模板分数、两次 L2 融合、空间增强 |
| `models/semantic_detail_fusion.py` | 语义—细节双支路单方向融合 |
| `models/maskformer_segmentation.py` | SAM3 prompt attention、Pixel Decoder、semantic head |
| `models/adapters/semantic_adapter.py` | 学生概率过滤、提示合并、推理输出 |
| `losses/semantic_criterion.py` | 主 BCE、Dice、训练教师蒸馏 |
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
3. 特征通道为 256，模板数为 64，聚合默认 2 块。
4. `clip_guidance_36` 完全由 RemoteCLIP 模板分数与 dense 特征生成，不注入 SAM3 FPN。
5. 两次拼接前分别执行逐像素通道 L2 归一化。
6. 类内使用 SAM3 FPN36 指导；RemoteCLIP 中间层仅保留提取与缓存，不参与聚合。
7. 聚合先在全部提示上执行，再按 chunk 高分辨率解码；聚合不能放进 chunk 循环。
8. 教师使用 `prompt_conditioned_features_72`，必须 `no_grad()` 并 detach；学生不得使用 `no_grad()`/`detach()`/`enable_grad()`。
9. SemanticDetailFusion 只在 72×72 执行一次；每条支路在原方向调用一次独立卷积块；输出替换 Pixel Decoder 最后一层 FPN 输入。
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
* 本次包含 CPU 合成张量的注意力、融合及梯度验证；未执行正式权重整模型前向、跨 GPU 数值等价或正式训练。不同 GPU 内核之间不承诺逐位一致。

## 13. 本次命名重构的非兼容变更

* `SAM3Segmentor` → `ReCASAM3`；`core.encoder_refiner` → `core.feature_aggregation`。
* `ClipScoreEmbedding` → `RemoteCLIPGuidanceEmbedding`。
* `ClassConditionedEncoderRefiner` → `RemoteCLIPGuidedAggregation`。
* 聚合块统一为 `AggregationAttentionBlock`，类间注意力更新后接三路并行的普通窗口注意力。
* `ClassScoreAttention` → `CrossClassAttention`；`WindowScoreAttention` → `WindowSpatialAttention`。
* `DecoderInputFusion` → `SemanticDetailFusion`；方向聚合函数已删除。
* 配置：`encoder_refiner_cfg` → `aggregation`；`openclip_cfg` → `remoteclip`；`freeze_cfg` → `freeze`；`criterion_cfg` → 根 `loss`。
* 参数路径整体变化，旧 checkpoint 不能严格恢复。checkpoint 容器 schema 仍为 4。
* 不提供旧参数映射、自动转换或兼容别名；迁移对照见 `docs/migration_reca_sam3.md`。

可用 `python -m unittest discover -s tests -v` 运行 CPU 回归测试，覆盖合成特征的前向与反向传播、有效配置、标签空间、schema、导入导出、多种子统计和结果记录。


## 14. 新基线与实验清理

* 公共配置为 batch 4、10000 步、LR 1e-4、固定蒸馏系数 0.5、W&B offline。SAM3、RemoteCLIP 与标签共享数据增强的空间朝向。
* 完整模型的实现标识为 `reca-sam3-multiscale-windows`。默认训练写入 `work_dirs/reca_sam3/full/seed_42/train`。
* 旧消融配置、专用模型分支、批量任务/权重选择、论文消融计划和表格均已清理；不自动迁移旧消融或旧 Full 成绩。
* 批量训练仅保留新 Full；独立验证保留 Full 与两个原始 SAM3 对比，继续使用相同八数据集。
* 删除 `cross_class_mode`、`intra_class_enabled`、`semantic_init`、`update_streams`、融合 `orientations`、concat fusion 和 simple adapter 开关；旧字段由配置校验拒绝。
* 两个原始 SAM3 比较仍由 `sam3_semantic` / `sam3_semantic_presence` 实现，不构建 RemoteCLIP 或聚合模块。
* CPU 检查覆盖三路普通窗口的覆盖范围、各残差系数的初始化和加权求和、两种参数共享设置的梯度、固定语义引导梯度、checkpoint 输出与梯度一致性、单方向融合、配置与批量入口。
