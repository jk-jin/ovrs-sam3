# ovrs-sam3 当前设计说明

更新时间：2026-06-29（清理背景映射与阈值配置：收敛到 dataset + adapter_cfg）
目标分支：`master`
主配置：
- 基础配置：`configs/ovrs_sam3_isaid_loveda_base.py`
- 实验短跑：`configs/ovrs_sam3_isaid_loveda_exp.py`
- 完整训练：`configs/ovrs_sam3_isaid_loveda_full.py`

训练入口：

```bash
python tools/train.py configs/ovrs_sam3_isaid_loveda_exp.py
python tools/train.py configs/ovrs_sam3_isaid_loveda_full.py
```

---

## 0. 一句话总结

当前 `ovrs-sam3` 是一个面向遥感开放词汇语义分割的 SAM3 + RemoteCLIP 融合框架，refiner 主工作尺度统一为 36×36。

核心思路：

```text
RemoteCLIP image encoder (504×504 input → 36×36 patch grid)
  → dense value-branch last block
  → remoteclip_feat_map [B, 768, 36, 36]

SAM3 encoder → 72×72 encoder features (bilinear downsample → 36×36)

32 固定 prompt templates × remoteclip_feat_map
  → clip_score_embed [B, C, 192, 36, 36]

SAM3 原始 segmentation_head → mask prior logits
  → progressive conv downsample
  → sam_score_embed [B, C, 64, 36, 36]

Concat + conv fuse → score_embed [B, C, 256, 36, 36]

Refiner layers at 36×36:
  ClassScoreAttention (dual value: feature + score)
  → WindowScoreAttention regular (ws=12, shift=0, relative pos bias)
  → WindowScoreAttention shifted (ws=12, shift=6)
  → FFN
  Both feature and score updated every layer.

EncoderFeatureUpsampler: 36 → 72
  ConvTranspose2d + FPN fusion + original encoder fusion

Write back → frozen segmentation_head → final_logits

DynamicClassThresholdHead:
  final_logits.detach().sigmoid() + refined_encoder_features_72.detach()
  → score encoder + cross-attention (query-memory) + class self-attention + MLP
  → class_thresholds [B, C] ∈ [0, 1]
```

---

## 1. 设计原则

### 1.1 冻结与可训练

**冻结**（始终 eval mode）：
- SAM3 backbone、transformer encoder、geometry encoder
- SAM3 segmentation_head
- RemoteCLIP image encoder
- RemoteCLIP text encoder

**可训练**：
- `ClassConditionedEncoderRefiner` 及其内部所有子模块
- `ClipScoreEmbedding` 中的 score conv
- `SamMaskScoreEmbedding` 中的下采样卷积
- `CombinedScoreEmbeddingBuilder` 中的融合卷积
- `EncoderRefinerLayer` 中的 class attention、window attention、FFN
- `EncoderFeatureUpsampler` 中的 deconv、FPN projection、融合卷积

### 1.2 预测方式

per-class binary score → argmax：

```text
raw_final_score_map = sigmoid(final_logits)
final_pred = final_score_map.argmax(dim=1)
```

---

## 2. 关键符号

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `C` | 类别数量 |
| `K` | 固定 prompt 模板数量，固定为 32 |
| `D` / `D_sam` | SAM3 hidden dim = 256 |
| `D_clip` | RemoteCLIP 图文对齐投影维度，ViT-L/14 = 768 |
| `D_native` | CLIP ViT 中间层原生通道数，ViT-L/14 = 1024 |
| `D_score` | fused score embedding 通道数 = 256 |
| `D_clip_score` | CLIP score 通道数 = 192 |
| `D_sam_score` | SAM score 通道数 = 64 |
| `H, W` | SAM3 encoder 最后一层空间分辨率 = 72×72 |
| `Hr, Wr` | refiner 主工作尺度 = 36×36 |
| `Hc, Wc` | RemoteCLIP dense patch grid = 36×36 |
| `L` | refiner 层数 = 4 |
| `ws` | 窗口注意力边长 = 12 |
| `shift` | shifted-window 平移距离 = 6 |

---

## 3. 配置组织

### 3.1 OpenCLIP 配置

```python
openclip_cfg=dict(
    enabled=True,
    model_name="ViT-L-14",
    pretrained="weights/RemoteCLIP-ViT-L-14.pt",
    default_output="feat_map",
    image_size=504,
    image_intermediate_layers=[7, 15],
    prompt_templates=[...],  # 32 templates
    normalize_label_for_clip=True,
)
```

新增 `image_size=504`，使 ViT-L/14 天然输出 36×36 patch grid。

### 3.2 Encoder Refiner 配置

```python
encoder_refiner_cfg=dict(
    enabled=True,
    fusion_layers=4,
    num_heads=8,
    dropout=0.1,
    hidden_dim=256,

    score_embed_dim=256,
    clip_score_embed_dim=192,
    sam_score_embed_dim=64,
    clip_score_conv_kernel=7,

    refiner_hw=36,
    encoder_hw=72,

    window_size=12,
    shift_size=6,

    use_checkpoint=True,
    early_prompt_attention=False,
)
```

关键变化：
- 删除了 `score_base_hw`
- 新增 `score_embed_dim=256`、`clip_score_embed_dim=192`、`sam_score_embed_dim=64`
- 新增 `refiner_hw=36`
- `window_size=12`、`shift_size=6`（适配 36×36）

### 3.3 Freeze 配置

```python
freeze_cfg=dict(
    train_adapters_only=True,
    trainable_modules=["core.encoder_refiner"],
    frozen_modules=[],
    openclip_text_finetune="frozen",
    openclip_image_finetune="frozen",
)
```

### 3.4 Optimizer 配置

```python
optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                "core.encoder_refiner": dict(lr_mult=4.0, decay_mult=1.0),
            },
        ),
    )
)
```

---

## 4. 核心数据流

### 4.1 RemoteCLIP 36×36 dense feature

```text
raw image → resize 504×504 → CLIP normalize
  → patch embedding (conv1) → [B, 1296, width]
  → add class token → [B, 1297, width]
  → interpolate positional embedding (bicubic, to 36×36)
  → ln_pre
  → blocks 0..L-2: standard resblock forward
  → last block: dense value-branch forward (no QK attention)
  → ln_post + proj → remove class token
  → reshape → remoteclip_feat_map [B, D_clip, 36, 36]
```

位置编码插值：class token 位置编码保留，patch 位置编码用 bicubic 插值到 36×36。

dense value-branch last block：取最后一层 attention 的 v 分支，经 out_proj 后加 class token 信息，再过 MLP。不使用 QK attention。

中间层特征 [7, 15] 继续提取但暂不接入主路径。

### 4.2 CLIP score embed

```text
class_names × 32 templates → frozen text encoder
  → template_clip_text [C, 32, 768]

template_clip_text × remoteclip_feat_map [B, 768, 36, 36]
  → einsum dot product * 20.0
  → score_maps_36 [B, C, 32, 36, 36]
  → Conv(32→192, 7×7) + GroupNorm + GELU
  → Conv(192→192, 3×3) + GroupNorm + GELU
  → clip_score_embed_36 [B, C, 192, 36, 36]
```

### 4.3 SAM mask prior score embed

```text
原始 encoder_out_chunks → frozen segmentation_head
  → sam_prior_logits [B, C, H_mask, W_mask]
  → progressive stride-2 conv downsample (3 stages)
  → bilinear to 36×36 if needed
  → Conv(hidden→64) + GroupNorm + GELU
  → sam_score_embed_36 [B, C, 64, 36, 36]
```

下采样使用逐步卷积（3 个 stride-2 conv），不使用 AvgPool 或 AdaptiveAvgPool。整个过程用 `torch.no_grad()`。

### 4.4 Score fusion

```text
concat(clip_score_embed_36, sam_score_embed_36)  → [B, C, 256, 36, 36]
  → Conv(256→256, 3×3) + GroupNorm + GELU
  → Conv(256→256, 3×3) + GroupNorm + GELU
  → score_embed_36 [B, C, 256, 36, 36]
```

### 4.5 Refiner layers

```text
encoder_features_72 → bilinear downsample → feature_36 [B, C, 256, 36, 36]

For each of L=4 layers:
  1. ClassScoreAttention:
     q/k = concat(feature, sam_text_mean, score_embed)  → 768 dims
     v_feat = feature, v_score = score_embed
     Attention across C classes at each spatial position.
     Both updated with learnable residual scales (init 1e-3).

  2. WindowScoreAttention regular (ws=12, shift=0):
     q/k = concat(feature, score_embed)  → 512 dims
     Dual value updates with relative position bias.

  3. WindowScoreAttention shifted (ws=12, shift=6):
     Same as above with cyclic shift.

  4. Per-token FFN (Linear, D→4D→D) for both feature and score.

  5. Output LayerNorm for both.
```

不再使用 SAM image features 作为 window attention 的 q/k 输入。

### 4.6 相对位置偏置

每个 WindowScoreAttention 维护一个可学习参数表：

```text
relative_position_bias_table: [(2*ws-1)², num_heads] = [529, 8]
```

attention 计算：`attn = q @ k^T * scale + rel_pos_bias [+ shift_mask] → softmax`。

### 4.7 36→72 upsampler

```text
refined_feature_36 → ConvTranspose2d(256, 256, k=2, s=2) → 72×72

sam_fpn_72 → 1×1 Conv → 64 channels → expand to [B*C, 64, 72, 72]
original_encoder_72 → reshape to [B*C, 256, 72, 72]

Concat → [B*C, 576, 72, 72]
  → Conv(576→256, 3×3) + GN + GELU
  → Conv(256→256, 3×3) + GN + GELU
  → final_refined_feature_72 [B, C, 256, 72, 72]
```

### 4.8 写回与最终输出

```text
refined_feature_72 → 写回 encoder_hidden_states
  → 冻结 segmentation_head
  → final_logits [B, C, H_out, W_out]
```

---

## 5. 子模块文件

| 文件 | 主要类/职责 |
|---|---|
| `models/openclip_image_encoder.py` | `OpenCLIPImageEncoder`: 504×504 dense 36×36 输出，bicubic pos embed 插值，dense value-branch last block |
| `models/openclip_text_encoder.py` | `OpenCLIPTextEncoder`: 冻结文本编码，`encode_prompt_templates()` 输出 [C, 32, 768] |
| `models/clip_score_embedding.py` | `ClipScoreEmbedding`: CLIP score [B, C, 192, 36, 36]<br>`SamMaskScoreEmbedding`: SAM mask prior [B, C, 64, 36, 36]<br>`CombinedScoreEmbeddingBuilder`: fusion → [B, C, 256, 36, 36] |
| `models/encoder_refiner_attention.py` | `ClassScoreAttention`: 类间双 value 注意力<br>`WindowScoreAttention`: 类内窗口注意力 + 相对位置偏置<br>`EncoderRefinerLayer`: 单层 refiner |
| `models/encoder_refiner.py` | `EncoderFeatureUpsampler`: 36→72 上采样融合<br>`DynamicClassThresholdHead`: 动态类别阈值预测<br>`ClassConditionedEncoderRefiner`: 顶层 refiner + threshold head |
| `models/sam3_image.py` | `Sam3Image`: pipeline coordinator，含 `build_sam_mask_prior_logits()`、`run_encoder_refiner()` |
| `config_dataclasses.py` | `EncoderRefinerConfig`、`OpenCLIPConfig` 等 dataclass |
| `model_builder.py` | `SAM3ModelBuilder`: 模型构建、冻结策略 |

---

## 6. Debug output keys

| key | 形状 | 含义 |
|-----|------|------|
| `final_logits` | `[B, C_active, H_out, W_out]` | 最终 mask logits（仅 active 类别） |
| `final_pred` | `[B, H, W]` | 最终预测（完整数据集 label id） |
| `final_score_map` | `[B, C_full, H_out, W_out]` | 阈值过滤后的完整类别 score map |
| `raw_final_score_map` | `[B, C_full, H_out, W_out]` | 原始 sigmoid score map（完整类别顺序，未阈值过滤） |
| `active_class_ids` | `[C_active]` | active 类别在完整类别空间中的 id |
| `original_num_classes` | scalar tensor | 完整数据集类别数 |
| `background_region` | `[B, H, W]` | 推理时所有 active 类都低于阈值的位置（bool） |
| `object_region` | `[B, H, W]` | 推理时至少一个 active 类通过阈值的位置（bool） |
| `encoder_features` | `[B, C_active, 256, 72, 72]` | 原始 encoder feature |
| `refined_encoder_features` | `[B, C_active, 256, 72, 72]` | refiner 更新后的 encoder feature |
| `refiner_features_36` | `[B, C_active, 256, 36, 36]` | refiner 输出的 feature_36 |
| `score_embed_36` | `[B, C_active, 256, 36, 36]` | fused score embedding |
| `clip_score_embed_36` | `[B, C_active, 192, 36, 36]` | CLIP score embedding |
| `sam_score_embed_36` | `[B, C_active, 64, 36, 36]` | SAM mask score embedding |
| `clip_score_maps` | `[B, C_active, 32, 36, 36]` | 图文相似度 maps |
| `template_clip_text_features` | `[C_active, 32, 768]` | 32 模板文本特征 |
| `clip_mid_features` | `List[[B, 1024, 36, 36]]` | CLIP 中间层特征（暂不接入主路径） |
| `class_thresholds` | `[B, C_active]` | 每图每类的动态置信度阈值，范围 [0, 1] |
| `class_threshold_logits` | `[B, C_active]` | 阈值 logits（sigmoid 前） |

注意：`C_active` 表示参与模型前向的类别数（删除背景类后），`C_full` 表示完整数据集类别数。

---

## 7. Loss 设计

### 7.1 Mask loss（BCE 主路径，不变）

```text
total_loss = final_bce_weight × BCE + final_dice_weight × Dice + dynamic_threshold_loss_weight × threshold_loss
```

BCE 始终使用原始 `final_logits`，不受阈值过滤影响。

### 7.2 动态阈值 loss（仅对出现过的类别）

```text
m = min(sigmoid(final_logits.detach()) inside GT mask region)
threshold_target = clamp(m - margin, 0, 1)

loss = SmoothL1(pred, threshold_target) + λ_over * ReLU(pred - m)^2
```

- `pred`：模型预测的动态阈值
- `m`：该类别真实区域内的最低分数
- `margin`：安全间隔，让阈值略低于真实最低分
- `λ_over`：越界惩罚权重（阈值超过真实最低分时严厉惩罚）

阈值 loss 只对 `presence_target > 0.5` 的类别计算。

---

## 8. 背景类处理（新增 2026-06-29）

### 8.1 设计原则

- **背景类别语义属于数据集**：是否启用、背景 id 是多少，在 dataset 配置的 `background_mapping` 中声明。
- **背景 id 以 reduce_zero_label 之后为准**。
- **训练和推理前向都删除背景类**：`find_text_batch` 只含 active 类别，core 只前向 active 类。
- **训练不做推理背景判定**：用 active_class_ids 对齐真实标签监督，背景像素自然成为所有 active 类的负样本。
- **推理阈值逻辑全部在 adapter**：`max(eval_cfg.prob_thd, dynamic_threshold)` 逐类过滤 → 背景区域判定 → active→full id 映射。
- **Evaluator 不再做阈值过滤**：只负责拿最终 `final_pred` 算指标。

### 8.2 配置职责划分

**背景 id**：唯一来源是 `dataset.background_mapping`。
**推理基础阈值**：唯一来源是 `adapter_cfg.threshold`。

```python
# 数据集配置
dataset=dict(
    ...
    reduce_zero_label=True,
    background_mapping=dict(
        enabled=True,
        background_id=0,        # reduce_zero_label 之后的背景 id
        default_background_id=255,
    ),
)

# adapter 配置（在 model dict 中）
adapter_cfg=dict(
    threshold=0.1,   # 推理基础置信度阈值
)

# eval_cfg 精简为仅保留 ignore_index
eval_cfg=dict(
    ignore_index=255,
)
```

- `background_id`：reduce_zero_label 之后的真实背景类别 id。
- `default_background_id`：adapter 推理时临时表示"背景区域"的默认 id（推荐 255，与 ignore_index 一致）。
- `adapter_cfg.threshold`：推理有效阈值 = max(threshold, dynamic_class_threshold)。
- `eval_cfg` 不再包含 `prob_thd`、`bg_idx`、`use_score_map`。

### 8.3 数据流

**训练**：
```text
dataset → full/active class 信息
collator → find_text_batch = active_class_names
core → 只前向 active 类 → final_logits [B, C_active, H, W]
adapter(final) → compact logits + active_class_ids
criterion → 用 active_class_ids 对齐 label_map
```

**推理**：
```text
core → 只前向 active 类
adapter(infer) → sigmoid → max(prob_thd, dynamic_threshold) 过滤
  → 所有 active 类低于阈值 = 背景区域
  → 非背景区域 argmax → active→full id 映射
  → final_pred (完整数据集 id), final_score_map (完整类别顺序)
evaluator → 直接用 final_pred 算指标
```

### 8.4 Adapter 三种模式

| 模式 | 用途 | 行为 |
|------|------|------|
| `final` | 训练 | 返回 compact logits + active_class_ids，不做 sigmoid/阈值/映射 |
| `infer` | 普通推理 | 完整后处理：阈值过滤 → 背景判定 → id 映射 |
| `infer_raw` | TTA | 返回 raw_final_score_map (完整类别顺序) + class_thresholds，不做最终后处理 |

TTA 流程：各 view → `infer_raw` → 平均 raw score map 和 thresholds → `adapter.postprocess_infer_outputs()`。

**有效阈值**：`max(adapter_cfg.threshold, dynamic_class_threshold[b, c])`。

**argmax 修正**：未通过阈值的类别被 mask 为 `-inf`，只在通过阈值的类别中选最大值。所有类别都不通过才是背景区域。

---

## 9. 文件结构

```text
models/
  encoder_refiner.py            ← ClassConditionedEncoderRefiner + EncoderFeatureUpsampler
  score_embeddings.py           ← ClipScoreEmbedding + SamMaskScoreEmbedding + CombinedScoreEmbeddingBuilder
  encoder_refiner_attention.py  ← ClassScoreAttention + WindowScoreAttention + EncoderRefinerLayer
  sam3_image.py                 ← Sam3Image pipeline coordinator (含 build_sam_score_embed_36)
  segmentor.py                  ← SAM3Segmentor wrapper
  openclip_image_encoder.py     ← RemoteCLIP dense 36×36 encoder
  openclip_text_encoder.py      ← frozen CLIP text encoder
  task_modes.py                 ← output key 定义
  adapters/semantic_adapter.py  ← 推理输出 adapter

losses/
  semantic_criterion.py

configs/
  ovrs_sam3_isaid_loveda_base.py  ← 基础配置
  ovrs_sam3_isaid_loveda_exp.py   ← 实验短跑
  ovrs_sam3_isaid_loveda_full.py  ← 完整训练

config_dataclasses.py
model_builder.py
```

---

## 10. 与旧版的主要差异

| 项目 | 旧设计 | 新设计 |
|------|--------|--------|
| RemoteCLIP 输入 | CLIP native size | 固定 504×504 |
| RemoteCLIP 输出 | native grid (如 24×24) | 固定 36×36 |
| 位置编码 | 不插值 | bicubic 插值到 36×36 |
| 最后一层 | 标准 forward | dense value-branch |
| CLIP mid features | 用于 score 上采样 | 提取但不接入主路径 |
| score embed 尺度 | 18/36/72 三尺度 | 仅 36×36 |
| score embed 组成 | 仅 CLIP (64ch) | CLIP (192ch) + SAM (64ch) = 256ch |
| SAM score 来源 | 无 | SAM3 seg head mask prior |
| refiner 工作尺度 | 72×72 + 多尺度 18/36 | 统一 36×36 |
| window attention q/k | feature + SAM image + score | feature + score |
| window size | 9 | 12 |
| 相对位置偏置 | 无 | GSNet/Swin 风格 |
| attention value | 单份 | 双份 (feature + score) |
| 残差系数 | 固定 1.0 | 可学习 (init 1e-3) |
| 36→72 上采样 | bilinear + FPN fusion (在 attention 内部) | ConvTranspose2d + FPN + orig encoder fusion (独立模块) |

---

## 11. 推荐检查命令

```bash
python -m py_compile data/dataset.py
python -m py_compile data/collate.py
python -m py_compile models/data_misc.py
python -m py_compile models/task_modes.py
python -m py_compile models/adapters/semantic_adapter.py
python -m py_compile losses/semantic_criterion.py
python -m py_compile models/segmentor.py
python -m py_compile engine/evaluator.py
python -m py_compile engine/trainer.py
python -m py_compile models/openclip_image_encoder.py
python -m py_compile models/clip_score_embedding.py
python -m py_compile models/encoder_refiner_attention.py
python -m py_compile models/encoder_refiner.py
python -m py_compile models/sam3_image.py
python -m py_compile config_dataclasses.py
python -m py_compile model_builder.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_base.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_exp.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_full.py
```

---

## 12. 最小 shape 验收

| 张量 | 期望形状 |
|---|---|
| `remoteclip_feat_map` | `[B, 768, 36, 36]` |
| `clip_score_embed_36` | `[B, C_active, 192, 36, 36]` |
| `sam_score_embed_36` | `[B, C_active, 64, 36, 36]` |
| `score_embed_36` | `[B, C_active, 256, 36, 36]` |
| `feature_36` (per layer) | `[B, C_active, 256, 36, 36]` |
| `refined_feature_72` | `[B, C_active, 256, 72, 72]` |
| `final_logits` | `[B, C_active, H_out, W_out]` |
| `final_pred` | `[B, H_out, W_out]` (完整数据集 label id) |
| `final_score_map` | `[B, C_full, H_out, W_out]` |
| `raw_final_score_map` | `[B, C_full, H_out, W_out]` |
| `active_class_ids` | `[C_active]` |
| `class_thresholds` | `[B, C_active]` |
| `class_threshold_logits` | `[B, C_active]` |
