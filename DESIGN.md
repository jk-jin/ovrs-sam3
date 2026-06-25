# ovrs-sam3 当前设计说明

更新时间：2026-06-25（最终固定 32 模板 + CLIP/SAM 特征融合设计）
目标分支：`master`
主配置：
- 基础配置：`configs/ovrs_sam3_isaid_loveda_base.py`
- 实验短跑：`configs/ovrs_sam3_isaid_loveda_exp.py`
- 完整训练：`configs/ovrs_sam3_isaid_loveda_full.py`

训练入口：

```bash
# 实验短跑
python tools/train.py configs/ovrs_sam3_isaid_loveda_exp.py
# 完整训练
python tools/train.py configs/ovrs_sam3_isaid_loveda_full.py
```

---

## 0. 一句话总结

当前 `ovrs-sam3` 是一个面向遥感开放词汇语义分割的 SAM3 + OpenCLIP 融合框架。

核心思路：

```text
SAM3 encoder 为每个类别生成类别条件 encoder feature；
冻结的 OpenCLIP 文本编码器用 32 个固定遥感模板生成类别文本向量；
冻结的 OpenCLIP 图像编码器提供最终图像特征和第 7 / 第 15 层中间特征；
CLIP 图文相似度生成三尺度 score embedding；
score embedding 和 SAM3 FPN 特征共同指导 refiner 更新 encoder feature；
最终 mask 仍直接由冻结的 SAM3 segmentation_head.forward 的 semantic_seg 输出。
```

整体结构：

```text
SAM3 encoder → e (每类 encoder 最后一层特征, 72×72)

同时提取：
→ SAM3 prompt_after_enc + prompt_mask
→ 有效文本 token 均值
→ sam_text_mean [B, C, D]

OpenCLIP frozen text encoder:
  class_names + 32 fixed remote-sensing prompt templates
  → template_clip_text [C, K, D_clip]，K=32

OpenCLIP frozen image encoder:
  raw image
  → clip_image_feat_map [B, D_clip, Hc, Wc]
  → clip_mid_layer7  [B, D_native, Hc, Wc]
  → clip_mid_layer15 [B, D_native, Hc, Wc]

ClipScoreEmbeddingBuilder:
  template_clip_text × clip_image_feat_map_18
  → dot product → score_maps_18 [B, C, K, 18, 18]
  → 7×7 conv (K→D_score) → score_embed_18 [B, C, D_score, 18, 18]
  → bilinear upsample + fuse CLIP layer15 → score_embed_36 [B, C, D_score, 36, 36]
  → bilinear upsample + fuse CLIP layer7  → score_embed_72 [B, C, D_score, 72, 72]

ClassConditionedEncoderRefiner:
  └ fusion_layers × EncoderRefinerLayer:
      ├ class attention
      │   q/k = encoder_features + sam_text_mean + clip_score_embed_72
      │   v   = encoder_features
      ├ multi-scale spatial window attention
      │   ├ 36×36 regular + shifted window attention
      │   ├ 18×18 regular + shifted window attention
      │   └ 上采样回 72×72 时融合 SAM3 FPN 特征
      └ FFN → LayerNorm

refined_e → 写回 encoder_hidden_states
→ 调用冻结的 SAM3 segmentation_head.forward
→ semantic_seg → final_logits
```

---

## 1. 设计原则

### 1.1 冻结与可训练

**冻结**（始终 eval mode）：
- SAM3 backbone、transformer encoder、geometry encoder
- SAM3 segmentation_head，包括 pixel_decoder、semantic_seg_head、cross_attend_prompt 等子模块
- OpenCLIP image encoder
- OpenCLIP text encoder

**可训练**：
- `ClassConditionedEncoderRefiner` 及其内部子模块
- `ClipScoreEmbeddingBuilder` 中的 score conv、CLIP mid feature projection、CLIP mid fusion conv
- `EncoderRefinerLayer` 中的 class attention、window attention、SAM3 FPN fusion upsample、FFN、归一化层

最终冻结配置：

```python
freeze_cfg=dict(
    train_adapters_only=True,
    trainable_modules=[
        "core.encoder_refiner",
    ],
    frozen_modules=[],
    openclip_text_finetune="frozen",
    openclip_image_finetune="frozen",
)
```

注意：
- OpenCLIP text encoder 不再做 attention q/v 微调。
- OpenCLIP image encoder 不参与训练。
- `OpenCLIPTextEncoder.encode_prompt_templates()` 继续走冻结文本编码路径，内部 `encode_text()` 的 `torch.no_grad()` 保持不变。
- 旧的 OpenCLIP text/image finetune 配置化基础设施可以保留，但当前主配置使用 `"frozen"`。

### 1.2 预测方式

仍采用 per-class binary score，再取类别维 argmax：

```text
raw_final_score_map = sigmoid(final_logits)
final_score_map = raw_final_score_map
final_pred = final_score_map.argmax(dim=1)
```

这里：
- `final_logits` 表示分割头输出的每类 logit。
- `sigmoid` 把每个类别的 logit 转成 0 到 1 之间的分数。
- `argmax(dim=1)` 表示在类别维度上取分数最高的类别作为最终预测。

### 1.3 学习率设计

当前只有 refiner 相关模块训练：

```text
core.encoder_refiner: 1e-4 × 4.0 = 4e-4
```

`grad_clip_norm=0.01`。

如果配置文件里保留了 `core.clip_text_encoder` 或 `core.clip_image_encoder` 的 optimizer custom key，也不会产生有效训练参数；为了配置更干净，建议只保留 `core.encoder_refiner` 的学习率规则。

---

## 2. 关键符号

| 符号 | 含义 |
|---|---|
| `B` | batch size，即一次送入模型的图片数量 |
| `C` | 当前数据集的类别数量 |
| `K` | 固定 prompt 模板数量，当前固定为 32 |
| `D` / `D_sam` | SAM3 hidden dim，当前为 256 |
| `D_clip` | OpenCLIP 图文对齐后的投影维度，ViT-L/14 当前为 768 |
| `D_native` | CLIP ViT 中间层原生通道数，ViT-L/14 当前通常为 1024 |
| `D_score` | score embedding 通道数，当前最终设计为 64 |
| `H, W` | SAM3 encoder 最后一层空间分辨率，当前为 72×72 |
| `Hc, Wc` | CLIP dense patch grid，ViT-L/14 输入 224×224 时通常为 16×16 |
| `H_out, W_out` | 最终语义分割 mask 分辨率，通常为 288×288 |
| `L` | refiner 层数，当前为 4 |
| `window_size` | 窗口注意力边长，当前为 9 |
| `shift_size` | shifted-window 平移距离，当前为 4 |
| `scale_18` | 18×18 分辨率，是 CLIP score embedding 的起点 |
| `scale_36` | 36×36 分辨率，是窗口注意力的中间尺度 |
| `scale_72` | 72×72 分辨率，是 encoder feature 的原始尺度 |

---

## 3. 配置组织

### 3.1 OpenCLIP 配置

```python
openclip_cfg=dict(
    enabled=True,
    model_name="ViT-L-14",
    pretrained="weights/RemoteCLIP-ViT-L-14.pt",
    default_output="feat_map",
    image_intermediate_layers=[7, 15],

    prompt_templates=[
        "a remote sensing image of {}.",
        "a satellite image of {}.",
        "an aerial image of {}.",
        "a high-resolution overhead image of {}.",
        "a top-down view of {}.",
        "a bird's-eye view image of {}.",
        "a remote sensing scene containing {}.",
        "a satellite scene containing {}.",
        "an aerial scene containing {}.",
        "a high-resolution remote sensing scene of {}.",
        "a land cover region of {} in a satellite image.",
        "a land use area of {} in an aerial image.",
        "a semantic segmentation region of {}.",
        "a labeled mask region corresponding to {}.",
        "a continuous area of {} in overhead imagery.",
        "a visible region of {} from above.",
        "the texture pattern of {} in a satellite image.",
        "the spatial pattern of {} in remote sensing imagery.",
        "the shape and boundary of {} in an aerial image.",
        "the object boundary of {} from an overhead view.",
        "a small-scale remote sensing object of {}.",
        "a large-scale remote sensing region of {}.",
        "multiple instances of {} in a satellite image.",
        "dense objects of {} in overhead imagery.",
        "sparse objects of {} in remote sensing imagery.",
        "urban remote sensing imagery showing {}.",
        "rural remote sensing imagery showing {}.",
        "natural land surface containing {}.",
        "man-made structures containing {}.",
        "a homogeneous area of {}.",
        "a complex background with {}.",
        "an object or region classified as {} in remote sensing imagery.",
    ],
    normalize_label_for_clip=True,
)
```

关键点：
- `prompt_templates` 是主路径，数量固定为 32。
- 每个模板都必须包含 `{}`，用于填入类别名。
- `image_intermediate_layers=[7, 15]` 是主路径需要的配置，第 15 层用于 score embed 的 18→36 上采样融合，第 7 层用于 36→72 上采样融合。
- `prompt_template` 单模板字段不再是主路径字段。

### 3.2 Encoder Refiner 配置

```python
encoder_refiner_cfg=dict(
    enabled=True,

    fusion_layers=4,
    num_heads=8,
    dropout=0.1,

    hidden_dim=256,

    clip_score_embed_dim=64,
    clip_score_conv_kernel=7,

    encoder_hw=72,
    score_base_hw=18,
    window_size=9,
    shift_size=4,

    use_checkpoint=True,

    # False = prompt attention inside segmentation_head
    # True  = prompt attention before refiner, skipped in segmentation_head
    early_prompt_attention=False,
)
```

关键变化：
- `num_query_tokens` 已删除或不再使用。
- 原来的 32 不再表示 learnable query token 数量，而是固定 prompt 模板数量 `K=32`。
- `clip_score_embed_dim` 从 128 改为 64。
- `score_base_hw=18` 仍表示 score embedding 起始尺度，之后生成 18 / 36 / 72 三个尺度。

### 3.3 Criterion 配置

```python
criterion_cfg=dict(
    ignore_index=255,
    final_bce_weight=1.0,
    final_dice_weight=0.0,
    bce_absent_class_weight=0.0,
    bce_valid_pixel_weight=1.0,
    bce_ignore_pixel_weight=0.05,
    eps=1e-6,
)
```

loss 仍然只监督 `final_logits`，不直接监督 score maps 或 score embeddings。

### 3.4 Freeze 配置

```python
freeze_cfg=dict(
    train_adapters_only=True,
    trainable_modules=[
        "core.encoder_refiner",
    ],
    frozen_modules=[],
    openclip_text_finetune="frozen",
    openclip_image_finetune="frozen",
)
```

`encoder_refiner` 可训练。SAM3 主体、SAM3 segmentation head、OpenCLIP text encoder、OpenCLIP image encoder 都冻结。

### 3.5 Optimizer 配置

推荐最终配置：

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

### 4.1 chunk 阶段：encoder 后停止 + 提取 `sam_text_mean`

```text
对每个类别 chunk：
  SAM3 prompt encode + transformer encoder
  → 停止，不跑 pixel decoder / segmentation head
  → 保存 encoder_out, prompt, prompt_mask
  → 提取 e_chunk: [B, C_chunk, D, 72, 72]
  → 提取 sam_text_mean_chunk: [B, C_chunk, D]

合并所有 chunk：
  → e: [B, C, D, 72, 72]
  → sam_text_mean: [B, C, D]
```

### 4.2 `sam_text_mean` 提取

```text
prompt_after_enc: [T, B*C, D]
prompt_mask:      [B*C, T]，True 表示无效 token

valid = ~prompt_mask.bool()
mean = sum(tokens * valid) / valid.sum()
reshape → [B, C, D]
```

这里：
- `T` 表示文本 token 数量。
- `B*C` 表示 batch 中每张图和每个类别组成的 pair 数。
- `D` 表示 SAM3 文本 token 的 hidden dim，当前为 256。

`sam_text_mean` 不做 Linear、不做 LayerNorm，直接使用 SAM3 原始有效文本 token 的均值。

### 4.3 CLIP score 路径：32 固定模板

```text
class_names + prompt_templates
→ OpenCLIPTextEncoder.encode_prompt_templates()
→ template_clip_text [C, K, D_clip]
```

这里：
- `C` 是类别数。
- `K=32` 是模板数。
- `D_clip=768` 是 OpenCLIP 图文对齐空间维度。

文本编码路径：
- 使用 `encode_prompt_templates()`。
- 内部调用 `encode_text()`。
- `encode_text()` 保持 `torch.no_grad()`，因为新设计中 CLIP text encoder 完全冻结。
- 不再使用 `encode_class_prompts()` 作为 score embed 主路径。
- 不再使用 `EncoderQueryExtractor` 或 `SingleTokenClipPromptEncoder`。

图文相似度：

```text
clip_image_feat_map [B, D_clip, Hc, Wc]
→ bilinear interpolate to 18×18
→ clip_image_feat_18 [B, D_clip, 18, 18]

template_clip_text [C, K, D_clip]
× clip_image_feat_18 [B, D_clip, 18, 18]
→ dot product
→ score_maps_18 [B, C, K, 18, 18]
```

实现上推荐使用：

```python
score_maps_18 = torch.einsum(
    "ckd,bdhw->bckhw",
    text_norm,
    image_norm,
) * 20.0
```

其中：
- `text_norm` 是 L2 normalize 后的模板文本向量。
- `image_norm` 是 L2 normalize 后的 CLIP 图像特征。
- `20.0` 是固定相似度缩放因子。

### 4.4 score embed 三尺度构建

```text
score_maps_18 [B, C, 32, 18, 18]
→ reshape [B*C, 32, 18, 18]
→ 7×7 conv + GroupNorm + GELU
→ score_embed_18_flat [B*C, 64, 18, 18]
→ reshape → score_embed_18 [B, C, 64, 18, 18]
```

18→36：

```text
score_embed_18_flat
→ bilinear interpolate to 36×36

clip_mid_layer15 [B, D_native, Hc, Wc]
→ bilinear interpolate to 36×36
→ 1×1 conv 降到 32 通道
→ expand 到 [B*C, 32, 36, 36]

concat:
  [B*C, 64, 36, 36] + [B*C, 32, 36, 36]
→ [B*C, 96, 36, 36]
→ 3×3 conv + GroupNorm + GELU
→ score_embed_36_flat [B*C, 64, 36, 36]
→ reshape → score_embed_36 [B, C, 64, 36, 36]
```

36→72：

```text
score_embed_36_flat
→ bilinear interpolate to 72×72

clip_mid_layer7 [B, D_native, Hc, Wc]
→ bilinear interpolate to 72×72
→ 1×1 conv 降到 32 通道
→ expand 到 [B*C, 32, 72, 72]

concat:
  [B*C, 64, 72, 72] + [B*C, 32, 72, 72]
→ [B*C, 96, 72, 72]
→ 3×3 conv + GroupNorm + GELU
→ score_embed_72_flat [B*C, 64, 72, 72]
→ reshape → score_embed_72 [B, C, 64, 72, 72]
```

注意：
- CLIP 中间层特征必须先降通道，不能直接用 1024 通道特征拼接。
- 必须通过 `clip_mid_layer_indices` 找到 layer 15 和 layer 7，不要假设 list 顺序。
- CLIP 中间层特征来自冻结 image encoder，并且保持 detach。

### 4.5 Refiner 阶段：更新 encoder feature

```text
e + sam_text_mean + clip_score_embeds + sam_image_last
→ ClassConditionedEncoderRefiner
→ refined_e [B, C, D, 72, 72]
```

每层 refiner 包含：

```text
ClassTokenScoreClassAttention:
  q/k = concat(encoder_features, sam_text_mean, clip_score_embed_72)
  v   = encoder_features

MultiScaleImageScoreWindowAttention:
  72 → ConvDownsample2d → 36
  36 → ConvDownsample2d → 18

  36 分支:
    regular window attention
    → shifted window attention
    → bilinear upsample to 72
    → fuse SAM3 FPN 72×72
    → LayerNorm

  18 分支:
    regular window attention
    → shifted window attention
    → bilinear upsample to 36
    → fuse SAM3 FPN 36×36
    → bilinear upsample to 72
    → fuse SAM3 FPN 72×72
    → LayerNorm

  两分支相加
  → LayerNorm

spatial_fused = class_attended_features + spatial_features
→ FFN
→ output LayerNorm
```

### 4.6 类间注意力 q/k 组成

```text
q/k = concat(encoder_features, sam_text_mean, clip_score_embed)
v   = encoder_features
```

具体维度：

```text
encoder_features: 256
sam_text_mean:    256
clip_score_embed: 64

qk_in_dim = 256 + 256 + 64 = 576
```

这里：
- `encoder_features` 是当前每类每个空间位置的 SAM3 encoder feature。
- `sam_text_mean` 是 SAM3 文本 token 均值。
- `clip_score_embed` 是 CLIP 图文相似度产生的 score embedding。
- `v` 只使用 encoder feature，避免直接把文本/score 信息当作 value 写入。

### 4.7 类内窗口注意力 q/k 组成

```text
q/k = concat(encoder_features, sam_image_features, clip_score_embed)
v   = encoder_features
```

具体维度：

```text
encoder_features:   256
sam_image_features: 256
clip_score_embed:   64

qk_in_dim = 256 + 256 + 64 = 576
```

其中：
- `sam_image_features` 来自 `sam_image_last`，也就是 SAM3 FPN 的 72×72 特征插值得到对应尺度。
- 36×36 attention 使用 36×36 的 `sam_image_features` 和 `score_embed_36`。
- 18×18 attention 使用 18×18 的 `sam_image_features` 和 `score_embed_18`。

### 4.8 窗口注意力上采样融合 SAM3 FPN

窗口注意力上采样不再只是 bilinear + conv，而是：

```text
attention output
→ bilinear interpolate
→ concat projected SAM3 FPN feature
→ conv 回 hidden_dim=256
```

FPN 融合方式：

```text
sam_fpn_72 [B, 256, 72, 72]
→ 1×1 conv 降到 64 通道
→ 按类别 expand 到 [B*C, 64, 72, 72]
→ 和上采样后的窗口特征拼接
→ 3×3 conv 回到 256 通道
```

18→36 时使用 `sam_fpn_36`：

```text
sam_fpn_72 → bilinear interpolate to 36×36 → sam_fpn_36
```

36→72 时使用原始 `sam_fpn_72`。

### 4.9 最终 mask 生成

```text
对每个 chunk：
  refined_e_chunk → 写回 encoder_hidden_states 的 visual token 区域
  → 调用冻结的 segmentation_head.forward()
  → chunk_logits: [B*C_chunk, 1, H_out, W_out]
  → reshape → [B, C_chunk, H_out, W_out]

合并所有 chunk：
  → final_logits [B, C, H_out, W_out]
```

不经过额外 mask query refiner，不做二次 mask head。最终 `final_logits` 直接来自 SAM3 `semantic_seg`。

### 4.10 显存优化

训练路径：
- `return_debug=False`
- 只返回 `{final_logits: ...}`
- 不生成 sigmoid、argmax、debug 中间量

推理路径：
- `return_debug=True` 时可返回 debug 中间量
- 中间量全部 `.detach().contiguous()`

Activation Checkpoint：
- 训练时可对每层 `EncoderRefinerLayer` 使用 `torch.utils.checkpoint.checkpoint`
- 推理时跳过 checkpoint
- OpenCLIP text encoder 不再参与训练，因此不需要文本路径 activation checkpoint

---

## 5. 子模块说明

### 5.1 ClipScoreEmbeddingBuilder

文件：`models/clip_score_embedding.py`

主职责：

```text
class_names + 32 prompt templates + clip_image_feat_map + clip_mid_features
→ template_clip_text
→ score_maps_18
→ score_embed_18 / score_embed_36 / score_embed_72
```

输入：

```text
class_names: list[str]，长度为 C
clip_image_feat_map: [B, D_clip, Hc, Wc]
clip_mid_features: List[[B, D_native, Hc, Wc]]
clip_mid_layer_indices: tuple[int, ...]
```

输出：

```text
clip_score_embeds:
  scale_18: [B, C, 64, 18, 18]
  scale_36: [B, C, 64, 36, 36]
  scale_72: [B, C, 64, 72, 72]

clip_score_maps_18: [B, C, 32, 18, 18]
template_clip_text: [C, 32, D_clip]
```

关键设计：
- 不再接收 `dynamic_clip_text`。
- 不再接收 `class_query_tokens`。
- 不再使用 ConvTranspose2d 做 score 上采样。
- 使用 CLIP layer 15 辅助 18→36。
- 使用 CLIP layer 7 辅助 36→72。
- CLIP 中间层先通过 1×1 conv 降到 32 通道，再与 64 通道 score embed 拼接。
- GroupNorm 自动选择安全组数。

### 5.2 OpenCLIPTextEncoder

文件：`models/openclip_text_encoder.py`

主路径使用：

```text
encode_prompt_templates(class_names, templates)
→ [C, 32, D_clip]
```

注意：
- `encode_prompt_templates()` 内部调用 `encode_text()`。
- `encode_text()` 中的 `torch.no_grad()` 保持不变。
- 当前最终设计中 OpenCLIP text encoder 完全冻结。
- `encode_class_prompts()` 可以保留作为工具函数，但不再是 score embed 主路径。

### 5.3 OpenCLIPImageEncoder

文件：`models/openclip_image_encoder.py`

输出：

```text
feat_map: [B, D_clip, Hc, Wc]
mid_features:
  layer 7:  [B, D_native, Hc, Wc]
  layer 15: [B, D_native, Hc, Wc]
mid_layer_indices: (7, 15)
```

关键设计：
- image encoder 默认冻结。
- `feat_map` 作为最终 CLIP dense 图像特征参与图文相似度。
- `mid_features` 作为 score embed 上采样的辅助特征。
- `mid_features` 保持 detach，不反传到 CLIP image encoder。

### 5.4 ClassTokenScoreClassAttention

文件：`models/encoder_refiner_attention.py`

类间注意力，在每个空间位置让不同类别互相看：

```text
q/k = concat(encoder_features, sam_text_mean, clip_score_embed)
v   = encoder_features
```

当前 `D_score=64`，因此 q/k 输入维度是 576。

### 5.5 ImageScoreWindowAttention

文件：`models/encoder_refiner_attention.py`

类内窗口注意力：

```text
输入:
  encoder_features   [B, C, 256, H, W]
  sam_image_features [B, 256, H, W]
  clip_score_embed   [B, C, 64, H, W]

处理:
  flatten batch+class
  → window partition
  → q/k = encoder + sam_image + score
  → v = encoder
  → regular 或 shifted window attention
  → residual + LayerNorm
  → window reverse
```

### 5.6 MultiScaleImageScoreWindowAttention

文件：`models/encoder_refiner_attention.py`

类内多尺度空间窗口注意力：

```text
encoder_features_72 [B, C, 256, 72, 72]
→ downsample → features_36 [B, C, 256, 36, 36]
→ downsample → features_18 [B, C, 256, 18, 18]

36 分支:
  regular attention
  → shifted attention
  → FPN-fused upsample to 72
  → norm_36_to_72

18 分支:
  regular attention
  → shifted attention
  → FPN-fused upsample to 36
  → FPN-fused upsample to 72
  → norm_18_to_72

两个分支相加
→ fused_norm
→ spatial_features_72
```

每个尺度保留独立的 regular/shifted attention 实例。

### 5.7 ClassConditionedEncoderRefiner

文件：`models/encoder_refiner.py`

顶层模块，组装：

```text
ClipScoreEmbeddingBuilder
EncoderRefinerLayer × L
```

不再组装：

```text
EncoderQueryExtractor
SingleTokenClipPromptEncoder
```

forward 签名建议：

```python
forward(
    encoder_features,
    clip_image_feat_map,
    clip_mid_features,
    clip_mid_layer_indices,
    sam_text_mean,
    class_names,
    sam_image_last,
)
```

返回：

```text
refined_encoder_features_72: [B, C, 256, 72, 72]
template_clip_text:          [C, 32, D_clip]
clip_score_embeds:           dict(scale_18, scale_36, scale_72)
clip_score_maps_18:          [B, C, 32, 18, 18]
```

残差设计：
- class attention 内部有 residual。
- window attention 内部有 residual。
- FFN 内部有 residual。
- class attention 输出与 spatial attention 输出相加。
- 每层末尾做 LayerNorm。
- 不使用全局残差，即原始 encoder feature 不加回最终输出。

---

## 6. refined_e 写回 encoder_hidden_states

```text
refined_e_chunk: [B, C_chunk, D, 72, 72]
  → reshape [B*C_chunk, D, 5184]
  → permute [5184, B*C_chunk, D]
  → 替换 encoder_hidden_states[:5184] 对应位置
```

替换区域与 `_embed_pixels` 提取 visual tokens 的区域一致。

---

## 7. pixel decoder 输入仍为 4 张特征图

```text
[fpn_0, fpn_1, fpn_2, refined_visual_tokens]
```

总数仍是 4 张。只替换最后一层 visual tokens，其他 FPN 特征继续作为 segmentation head 的输入。

---

## 8. output keys

| key | 形状 | 含义 |
|-----|------|------|
| `final_logits` | `[B, C, H_out, W_out]` | 最终 mask logits，直接来自 `semantic_seg` |
| `raw_final_score_map` | `[B, C, H_out, W_out]` | `sigmoid(final_logits)` |
| `final_score_map` | `[B, C, H_out, W_out]` | `sigmoid(final_logits)` |
| `final_pred` | `[B, H_out, W_out]` | `final_score_map.argmax(dim=1)` |
| `encoder_features` | `[B, C, 256, 72, 72]` | 原始 encoder feature |
| `refined_encoder_features` | `[B, C, 256, 72, 72]` | refiner 更新后的 encoder feature |
| `template_clip_text_features` | `[C, 32, D_clip]` | 32 模板得到的 CLIP 文本特征 |
| `clip_score_embed` | `[B, C, 64, 72, 72]` | scale_72 的 CLIP score embedding |
| `clip_score_maps` | `[B, C, 32, 18, 18]` | 32 模板对应的图文相似度图 |
| `clip_mid_features` | `List[[B, D_native, Hc, Wc]]` | CLIP ViT 中间层特征，默认 layer 7 和 15 |

不再输出：

```text
class_query_tokens
dynamic_clip_text_features
```

因为新设计中没有 learnable query，也没有 dynamic CLIP text。

---

## 9. Loss 设计

文件：`losses/semantic_criterion.py`

训练目标：

```text
final_logits vs label_map
```

总 loss：

```text
total_loss = final_bce_weight × loss_final_bce
           + final_dice_weight × loss_final_dice
```

其中：
- `loss_final_bce` 是最终 mask 的 BCE loss。
- `loss_final_dice` 是最终 mask 的 Dice loss。
- 当前配置中 `final_dice_weight=0.0`，所以实际只使用 BCE。
- score maps、score embeddings、template text features 不直接参与 loss。

---

## 10. Adapter

文件：`models/adapters/semantic_adapter.py`

`SemanticSegAdapter` 区分两种输出模式：

- `output_mode="final"`：训练使用，只返回 `{final_logits: [B, C, H, W]}`。
- `output_mode="infer"`：推理使用，返回完整输出，包括 `final_logits`、`raw_final_score_map`、`final_score_map`、`final_pred` 和可选 debug 中间量。

---

## 11. 可视化

文件：
- `engine/visualization.py`
- `configs/_base_/visualization.py`

可视化逻辑不影响模型主路径。实验阶段可以通过：

```python
visualization=dict(
    enabled=False,
)
```

关闭可视化文件生成。

---

## 12. 文件结构

```text
models/
  encoder_refiner.py            ← 顶层 ClassConditionedEncoderRefiner
  clip_score_embedding.py       ← 固定 32 模板 + CLIP 中间层融合的 score embedding 构建
  encoder_refiner_attention.py  ← ClassTokenScoreClassAttention,
                                   ImageScoreWindowAttention,
                                   MultiScaleImageScoreWindowAttention,
                                   FPN-fused upsample,
                                   EncoderRefinerLayer
  sam3_image.py                 ← pipeline coordinator，负责 cache、refiner 调用、写回 encoder_hidden_states
  segmentor.py                  ← segmentor wrapper
  openclip_image_encoder.py     ← frozen CLIP image encoder + raw image preprocessing + mid features
  openclip_text_encoder.py      ← frozen CLIP text encoder + encode_prompt_templates
  task_modes.py                 ← output key 定义
  adapters/semantic_adapter.py  ← 推理输出 adapter

已删除或不再主路径引用：
  encoder_query_extractor.py    ← 旧 learnable query 模块
  clip_prompt_encoder.py        ← 旧 dynamic CLIP text 模块
  mask_query_refiner.py
  class_text_guided_mask_prior_clip_fusion_mixer.py
  lowres_aggregator.py
  clip_guided_upsampler.py
  final_mask_head.py
  dynamic_clip_prompt.py

losses/
  semantic_criterion.py         ← loss

configs/
  ovrs_sam3_isaid_loveda_base.py  ← 基础配置（模型结构 + iSAID train + LoveDA val）
  ovrs_sam3_isaid_loveda_exp.py   ← 实验短跑配置
  ovrs_sam3_isaid_loveda_full.py  ← 完整训练配置
  sam3_semantic.py                ← 废弃兼容 shim（→ base）
  sam3_isaid_train_loveda_val.py  ← 废弃兼容 shim（→ exp）

config_dataclasses.py           ← dataclass 配置定义
model_builder.py                ← 模型构建、冻结策略、训练组件构建
```

---

## 13. OpenCLIP text/image finetune 说明

虽然代码中仍可以保留 OpenCLIP text/image finetune 的基础设施，但当前最终设计使用：

```python
openclip_text_finetune="frozen"
openclip_image_finetune="frozen"
```

模式说明：

| 模式 | 含义 | 当前是否使用 |
|------|------|--------------|
| `"frozen"` | 完全冻结，无梯度 | 使用 |
| `"attention"` | 只训练 attention q/v 投影 + positional embedding | 不使用 |
| `"transformer"` | 训练整个 transformer + positional embedding | 不使用 |
| `"full"` | 训练整个 OpenCLIP encoder | 不使用 |

关键点：
- 不要删除 `encode_text()` 中的 `torch.no_grad()`。
- 不要把 `encode_prompt_templates()` 改成可训练路径。
- 不要让 CLIP mid features 保留梯度。
- 当前训练参数应只来自 `core.encoder_refiner`。

---

## 14. 与上一版 baseline 的主要差异

| 项目 | 旧 baseline | 当前最终设计 |
|------|-------------|--------------|
| 文本来源 | 单模板 + learnable query 生成 dynamic CLIP text | 32 个固定遥感模板 |
| 文本编码路径 | `encode_class_prompts()`，可支持 text attention 微调 | `encode_prompt_templates()`，冻结文本编码 |
| query 模块 | `EncoderQueryExtractor` | 删除 |
| CLIP prompt fusion | `SingleTokenClipPromptEncoder` | 删除 |
| score maps 通道 | `Q=32` 个 learnable query 对应通道 | `K=32` 个固定模板对应通道 |
| score embed dim | 128 | 64 |
| score embed 上采样 | ConvTranspose2d | bilinear + CLIP mid feature fusion |
| CLIP layer 15 | 提取但不用 | 用于 18→36 score 上采样 |
| CLIP layer 7 | 提取但不用 | 用于 36→72 score 上采样 |
| window attention 尺度 | 36 和 18 | 36 和 18，保持不变 |
| window attention 上采样 | bilinear + conv | bilinear + SAM3 FPN fusion + conv |
| OpenCLIP text encoder | attention q/v 微调 | frozen |
| OpenCLIP image encoder | frozen | frozen |

---

## 15. 推荐检查命令

```bash
python -m py_compile config_dataclasses.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_base.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_exp.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_full.py
python -m py_compile model_builder.py
python -m py_compile models/openclip_text_encoder.py
python -m py_compile models/openclip_image_encoder.py
python -m py_compile models/clip_score_embedding.py
python -m py_compile models/encoder_refiner.py
python -m py_compile models/encoder_refiner_attention.py
python -m py_compile models/sam3_image.py models/segmentor.py
python -m py_compile models/task_modes.py
python -m py_compile models/adapters/semantic_adapter.py
```

如果旧文件已删除，不要再检查：

```bash
python -m py_compile models/clip_prompt_encoder.py
python -m py_compile models/encoder_query_extractor.py
```

推荐搜索旧主路径残留：

```bash
grep -R "EncoderQueryExtractor" -n models config_dataclasses.py model_builder.py configs || true
grep -R "SingleTokenClipPromptEncoder" -n models config_dataclasses.py model_builder.py configs || true
grep -R "class_query_tokens" -n models config_dataclasses.py model_builder.py configs || true
grep -R "dynamic_clip_text" -n models config_dataclasses.py model_builder.py configs || true
grep -R "num_query_tokens" -n models config_dataclasses.py model_builder.py configs || true
```

推荐打印最终配置：

```bash
python tools/train.py configs/ovrs_sam3_isaid_loveda_base.py --print-config
python tools/train.py configs/ovrs_sam3_isaid_loveda_exp.py --print-config
python tools/train.py configs/ovrs_sam3_isaid_loveda_full.py --print-config
```

重点确认：

```text
openclip_cfg.prompt_templates 有 32 个模板
encoder_refiner_cfg.clip_score_embed_dim = 64
freeze_cfg.openclip_text_finetune = frozen
freeze_cfg.openclip_image_finetune = frozen
没有 encoder_refiner_cfg.num_query_tokens
没有 openclip_cfg.prompt_template
```

---

## 16. 2026-06-23 修复记录

以下修复属于基础稳定性修复，当前最终设计继续保留。

### 16.1 shifted window attention mask 修复

- 删除 `_build_shift_attn_mask()` 中对 `img_mask` 的 `torch.roll`。
- feature map 需要 roll，但 region mask 必须保持在原始坐标系中标记不同区域。
- mask fill 值从 `float("-inf")` 改为 `-100.0`，在 AMP/FP16 下更稳定。

### 16.2 窗口尺寸整除检查

`ImageScoreWindowAttention.forward()` 中检查 H/W 是否能被 `window_size` 整除。当前 18、36、72 都能被 9 整除。

### 16.3 冻结模块 train/eval 状态修复

`SAM3Segmentor.train()` 中，`super().train(mode)` 后强制将冻结模块切回 eval mode：

```text
SAM3 backbone
SAM3 transformer
SAM3 geometry_encoder
SAM3 segmentation_head
OpenCLIP image encoder
OpenCLIP text encoder
```

训练时只有 `encoder_refiner` 进入 train mode。

### 16.4 训练 forward 输出模式修复

`SAM3Segmentor.forward()` 中：

```text
训练时 output_mode="final"
推理时 output_mode="infer"
```

训练时只输出 `final_logits`，减少显存占用。

### 16.5 CLIP score map 显存优化

score map 计算使用 einsum，当前最终设计推荐：

```python
torch.einsum("ckd,bdhw->bckhw", text_norm, image_norm)
```

避免显式展开出过大的中间张量。

### 16.6 clip_mid_features 数量检查

当前最终设计依赖 CLIP layer 7 和 15，因此应检查 `clip_mid_layer_indices` 中包含 7 和 15。  
不建议只检查数量，因为应该根据 layer index 定位具体特征。

### 16.7 未定义变量修复

`build_encoder_refiner_cache()` 的 shape mismatch 异常信息中，继续使用已定义的 `encoder_features_72`，不要引用旧变量 `e`。

---

## 17. 2026-06-23 单卡实验基础设施

### 17.1 `--cfg-options`

`tools/train.py` 支持命令行覆盖配置：

```bash
python tools/train.py configs/ovrs_sam3_isaid_loveda_exp.py \
  --cfg-options train_cfg.max_iters=1000 model.criterion_cfg.final_dice_weight=0.3
```

支持 bool、None、int、float、list、str。不存在的 key 会报错。

### 17.2 `--print-config`

打印合并后的最终配置并退出：

```bash
python tools/train.py configs/ovrs_sam3_isaid_loveda_exp.py --print-config
```

### 17.3 metrics.jsonl

`engine/experiment_hooks.py` 中的 `MetricsJsonlHook` 会写入：

```text
work_dirs/xxx/metrics.jsonl
```

记录：
- `mode="train"`：训练指标
- `mode="val"`：验证指标
- `mode="meta"`：开始/结束标记

### 17.4 W&B

`WandbHook` 默认关闭：

```python
experiment_tracking=dict(
    wandb=dict(
        enabled=False,
    )
)
```

启用后只记录 scalar metrics 和 config，不上传模型权重。

### 17.5 结果汇总

```bash
python tools/collect_experiments.py work_dirs/sprint \
  --monitor semantic.miou \
  --mode max \
  --output summary.csv
```

### 17.6 W&B sweep work_dir 防冲突

可以通过：

```bash
python tools/train.py config.py \
  --work-dir work_dirs/sweeps/v1 \
  --work-dir-suffix-keys model.freeze_cfg.openclip_text_finetune model.encoder_refiner_cfg.clip_score_embed_dim
```

让不同 trial 写入不同子目录。

---

## 18. 验证子集与可视化关闭

### 18.1 `val_max_iters`

```python
train_cfg=dict(
    val_max_iters=500,
)
```

当 `batch_size=1` 时，表示每次验证最多跑 500 张图片。

### 18.2 visualization 关闭

```python
visualization=dict(
    enabled=False,
)
```

关闭可视化文件生成，不影响训练和验证指标。

---

## 19. 推荐实验配置：iSAID train + LoveDA val

已有实验配置：

```text
configs/ovrs_sam3_isaid_loveda_exp.py   ← 实验短跑
configs/ovrs_sam3_isaid_loveda_full.py  ← 完整训练
configs/ovrs_sam3_isaid_loveda_base.py  ← 基础配置
```

推荐用于快速验证最终结构：

```text
训练集：iSAID
验证集：LoveDA
训练步数：4000 (exp) / 20000 (full)
验证频率：每 1000 步 (exp) / 每 2000 步 (full)
每次验证：LoveDA 500 张
可视化：关闭 (exp) / 开启 (full)
```

损失函数、冻结策略和模型结构默认继承 `ovrs_sam3_isaid_loveda_base.py` 的最终设计。

---

## 20. 最终主路径检查清单

最终代码主路径应满足：

```text
使用 32 个固定 prompt templates
score_embed_dim = 64
OpenCLIP text encoder frozen
OpenCLIP image encoder frozen
score embed 上采样融合 CLIP layer 15 / layer 7
window attention 上采样融合 SAM3 FPN
class attention 仍使用 sam_text_mean + score_embed
window attention 仍使用 36 和 18 双尺度
最终 mask 仍来自冻结 SAM3 segmentation_head 的 semantic_seg
```

最终代码主路径不应再包含：

```text
EncoderQueryExtractor
SingleTokenClipPromptEncoder
learnable query
class_query_tokens
dynamic_clip_text
num_query_tokens
单模板 prompt_template 作为主路径
ConvTranspose2d 作为 score embed 上采样主路径
OpenCLIP text attention 微调
OpenCLIP image encoder 训练
```