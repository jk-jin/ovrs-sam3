# ovrs-sam3 项目说明文档

更新时间：2026-07-04  
目标分支：`master`

## 1. 项目定位

`ovrs-sam3` 是一个面向遥感图像开放词汇语义分割的模型框架。项目把 SAM3 的图像编码能力、SAM3 的 mask 预测能力和 RemoteCLIP 的图文对齐能力结合起来，用一个可训练的 encoder refiner 对类别相关的图像特征和 score embedding 进行融合更新，最后仍然通过冻结的 SAM3 segmentation head 输出每个类别的 mask logits。

当前主线设计可以概括为：

```text
遥感图像 + 类别文本
  → SAM3 提取 72×72 encoder feature
  → RemoteCLIP 提取 36×36 dense image feature
  → CLIP 文本模板和图像特征计算类别 score map
  → SAM3 segmentation head 生成 mask prior score
  → CLIP score + SAM score 融合成 score_embed
  → encoder refiner 在 36×36 上同时更新 feature 和 score_embed
  → refined feature 上采样回 72×72
  → 写回 SAM3 encoder_hidden_states
  → 冻结的 SAM3 segmentation head 输出 final_logits
```

最终预测方式是：

```text
raw_final_score_map = sigmoid(final_logits)
final_pred = raw_final_score_map.argmax(dim=1)
```

其中：

- `final_logits` 表示模型输出的每类 mask logits，形状是 `[B, C, H_out, W_out]`。
- `sigmoid` 表示把 logits 转成 0 到 1 之间的类别得分。
- `raw_final_score_map` 表示每个像素属于每个类别的分数图。
- `argmax(dim=1)` 表示在类别维度上取分数最高的类别。
- `final_pred` 表示最终语义分割类别图，形状是 `[B, H_out, W_out]`。

## 2. 主入口和配置

### 2.1 训练入口

短实验：

```bash
python tools/train.py configs/ovrs_sam3_isaid_loveda_exp.py
```

完整训练：

```bash
python tools/train.py configs/ovrs_sam3_isaid_loveda_full.py
```

### 2.2 主配置文件

```text
configs/
  ovrs_sam3_isaid_loveda_base.py  # 基础配置
  ovrs_sam3_isaid_loveda_exp.py   # 短实验配置
  ovrs_sam3_isaid_loveda_full.py  # 完整训练配置
```

`base.py` 负责定义模型、数据、优化器、默认训练参数。  
`exp.py` 继承 `base.py`，用于短实验和 sweep。  
`full.py` 继承 `base.py`，用于完整训练。

## 3. 关键符号说明

| 符号 | 含义 |
|---|---|
| `B` | batch size，也就是一次输入的图像数量。 |
| `C` | 类别数量，也就是当前 batch 中参与开放词汇分割的类别数。 |
| `K` | 每个类别使用的 prompt 模板数量，当前固定为 32。 |
| `D` | SAM3 hidden dimension，当前是 256。 |
| `D_clip` | RemoteCLIP 图文对齐空间维度，ViT-L/14 下通常是 768。 |
| `D_native` | RemoteCLIP ViT 中间层原生通道数，ViT-L/14 下通常是 1024。 |
| `D_score` | 融合后的 score embedding 通道数，当前是 256。 |
| `D_clip_score` | CLIP score embedding 通道数，当前是 192。 |
| `D_sam_score` | SAM mask prior score embedding 通道数，当前是 64。 |
| `H, W` | SAM3 encoder feature 的空间分辨率，当前主流程中是 72×72。 |
| `Hr, Wr` | refiner 主工作分辨率，当前是 36×36。 |
| `Hc, Wc` | RemoteCLIP dense patch grid，当前是 36×36。 |
| `L` | refiner layer 数量，当前默认是 4。 |
| `ws` | window attention 的窗口边长，当前默认是 12。 |
| `shift` | shifted window attention 的平移距离，当前默认是 6。 |

## 4. 冻结模块和可训练模块

### 4.1 冻结模块

训练时以下模块保持冻结，并强制处于 eval mode：

```text
SAM3 backbone
SAM3 transformer encoder
SAM3 geometry encoder
SAM3 segmentation_head
RemoteCLIP image encoder
RemoteCLIP text encoder
```

这些模块不直接更新参数，主要提供稳定的图像特征、文本特征和 mask 解码能力。

### 4.2 可训练模块

当前默认训练对象是：

```text
core.encoder_refiner
```

它内部主要包括：

```text
ClipScoreEmbedding 中的 score_conv
SamMaskScoreEmbedding 中的下采样卷积
CombinedScoreEmbeddingBuilder 中的融合卷积
EncoderRefinerLayer 中的 class attention
EncoderRefinerLayer 中的 window attention
EncoderRefinerLayer 中的 FFN
EncoderFeatureUpsampler 中的融合卷积
```

默认 freeze 配置如下：

```python
freeze_cfg=dict(
    train_adapters_only=True,
    trainable_modules=["core.encoder_refiner"],
    frozen_modules=[],
    openclip_text_finetune="frozen",
    openclip_image_finetune="frozen",
)
```

## 5. RemoteCLIP 图像分支

RemoteCLIP image encoder 用于生成 36×36 的 dense image feature。

处理流程：

```text
raw image
  → resize 到 504×504
  → CLIP mean/std normalize
  → ViT patch embedding
  → 得到 36×36 patch grid
  → 加入 class token
  → positional embedding 插值到 36×36
  → 前 L-1 个 transformer block 正常 forward
  → 最后一个 block 使用 dense value-branch forward
  → ln_post + projection
  → 去掉 class token
  → reshape 成 remoteclip_feat_map
```

输出：

```text
remoteclip_feat_map: [B, D_clip, 36, 36]
clip_mid_features: List[[B, D_native, 36, 36]]
```

说明：

- RemoteCLIP 输入固定为 504×504。
- ViT-L/14 的 patch size 是 14，因此 504 / 14 = 36，天然得到 36×36 patch grid。
- positional embedding 会使用 bicubic 插值到 36×36。
- 最后一个 block 不使用标准 QK attention 聚合，而是提取 value branch，并注入 class token 信息，得到更密集的局部图像特征。
- `clip_mid_features` 当前会提取，但主路径没有直接使用。

## 6. RemoteCLIP 文本分支和 CLIP score embedding

每个类别会使用 32 个固定 prompt templates 生成文本特征。

流程：

```text
class_names × 32 templates
  → RemoteCLIP text encoder
  → template_clip_text: [C, 32, D_clip]
```

然后用文本特征和 RemoteCLIP dense image feature 做点积：

```text
template_clip_text × remoteclip_feat_map
  → clip_score_maps: [B, C, 32, 36, 36]
```

其中：

- `B` 表示图像数量。
- `C` 表示类别数量。
- `32` 表示每个类别的 prompt 模板数量。
- `36×36` 表示 RemoteCLIP 图像特征的空间网格。
- `clip_score_maps` 表示每个类别、每个模板在每个位置上的图文相似度。

随后通过两层卷积生成 CLIP score embedding：

```text
clip_score_maps [B, C, 32, 36, 36]
  → Conv(32→192, 7×7) + GroupNorm + GELU
  → Conv(192→192, 3×3) + GroupNorm + GELU
  → clip_score_embed_36 [B, C, 192, 36, 36]
```

## 7. SAM mask prior score embedding

SAM3 原始 segmentation head 会先根据未 refine 的 encoder output 生成一个 mask prior。为了避免一次性保存完整高分辨率 prior，当前代码按类别 chunk 生成、立刻下采样、然后释放中间 logits。

流程：

```text
encoder_out_chunks
  → frozen SAM3 segmentation_head
  → chunk_logits
  → SamMaskScoreEmbedding
  → sam_score_embed_36
```

`SamMaskScoreEmbedding` 内部结构：

```text
sam_prior_logits [B, C, H_mask, W_mask]
  → Conv stride=2 + GroupNorm + GELU
  → Conv stride=2 + GroupNorm + GELU
  → Conv stride=2 + GroupNorm + GELU
  → bilinear resize 到 36×36
  → Conv(hidden→64) + GroupNorm + GELU
  → sam_score_embed_36 [B, C, 64, 36, 36]
```

说明：

- SAM3 segmentation head 是冻结的。
- `chunk_logits` 会 detach，避免梯度回传到 SAM3。
- `SamMaskScoreEmbedding` 是可训练的。
- 这个分支为 refiner 提供来自 SAM3 mask prior 的空间提示。

## 8. Score embedding 融合

CLIP score embedding 和 SAM score embedding 会在通道维度拼接：

```text
clip_score_embed_36: [B, C, 192, 36, 36]
sam_score_embed_36:  [B, C,  64, 36, 36]
```

拼接后：

```text
concat → [B, C, 256, 36, 36]
  → Conv(256→256, 3×3) + GroupNorm + GELU
  → Conv(256→256, 3×3) + GroupNorm + GELU
  → score_embed_36 [B, C, 256, 36, 36]
```

这里的 `score_embed_36` 是 refiner 中与 image feature 并行更新的类别 score 表征。

## 9. Encoder refiner

`ClassConditionedEncoderRefiner` 是项目当前最核心的可训练模块。

输入包括：

```text
encoder_features_72: [B, C, 256, 72, 72]
clip_image_feat_map: [B, D_clip, 36, 36]
sam_text_mean:       [B, C, 256]
class_names:         List[str]
sam_score_embed_36:  [B, C, 64, 36, 36]
sam_fpn_72:          [B, 256, 72, 72]
```

输出包括：

```text
refined_encoder_features_72: [B, C, 256, 72, 72]
refiner_features_36:         [B, C, 256, 36, 36]
score_embed_36:              [B, C, 256, 36, 36]
clip_score_embed_36:         [B, C, 192, 36, 36]
sam_score_embed_36:          [B, C,  64, 36, 36]
clip_score_maps_36:          [B, C,  32, 36, 36]
template_clip_text:          [C, 32, D_clip]
```

### 9.1 进入 refiner 前的尺度变换

SAM3 encoder feature 原始是 72×72：

```text
encoder_features_72 [B, C, 256, 72, 72]
  → bilinear downsample
  → feature_36 [B, C, 256, 36, 36]
```

`feature_36` 和 `score_embed_36` 会一起进入多个 `EncoderRefinerLayer`。

### 9.2 EncoderRefinerLayer 结构

每一层 refiner 的顺序是：

```text
ClassScoreAttention
  → WindowScoreAttention regular
  → WindowScoreAttention shifted
  → feature FFN + score FFN
  → output LayerNorm
```

默认有 4 层。

### 9.3 ClassScoreAttention

`ClassScoreAttention` 在每个空间位置上做类别之间的 attention。

输入：

```text
feature:      [B, C, 256, 36, 36]
score_embed:  [B, C, 256, 36, 36]
sam_text_mean:[B, C, 256]
```

q/k 构造：

```text
q/k = concat(feature, sam_text_mean, score_embed)
```

拼接后通道数是：

```text
256 + 256 + 256 = 768
```

其中：

- 第一个 256 来自图像 feature。
- 第二个 256 来自 SAM3 text prompt mean。
- 第三个 256 来自 score embedding。

value 分两路：

```text
v_feature = feature
v_score   = score_embed
```

注意力权重只有一套，但会同时更新 feature 和 score_embed：

```text
feature    = LayerNorm(feature + Dropout(feature_update))
score_embed = LayerNorm(score_embed + Dropout(score_update))
```

这里没有额外的可学习残差系数，使用的是普通残差。

### 9.4 WindowScoreAttention

`WindowScoreAttention` 在每个类别内部做局部窗口注意力。

默认参数：

```text
window_size = 12
shift_size = 6
num_heads = 8
```

regular window attention：

```text
window_size = 12
shift_size = 0
```

shifted window attention：

```text
window_size = 12
shift_size = 6
```

当前 q/k 构造：

```text
q/k = concat(feature, score_embed, sam_fpn_36)
```

拼接后通道数是：

```text
256 + 256 + 256 = 768
```

其中：

- 第一个 256 来自当前类别的 feature。
- 第二个 256 来自当前类别的 score embedding。
- 第三个 256 来自 SAM3 FPN feature 下采样后的 `sam_fpn_36`。

value 分两路：

```text
v_feature = feature
v_score   = score_embed
```

输出更新：

```text
feature    = LayerNorm(feature + Dropout(feature_update))
score_embed = LayerNorm(score_embed + Dropout(score_update))
```

这里同样使用普通残差，不使用可学习残差系数。

### 9.5 相对位置偏置

每个 `WindowScoreAttention` 都维护一个可学习的相对位置偏置表：

```text
relative_position_bias_table: [(2 × ws - 1)², num_heads]
```

当前 `ws = 12`，所以：

```text
(2 × 12 - 1)² = 23² = 529
```

因此相对位置偏置表形状是：

```text
[529, 8]
```

其中：

- `ws` 表示窗口边长。
- `num_heads` 表示注意力头数量。
- `529` 表示窗口内任意两个 token 的相对位置种类数。
- `8` 表示每个 attention head 都有自己的相对位置偏置。

attention 计算逻辑：

```text
attn = q @ k^T × scale + relative_position_bias + shift_mask
attn = softmax(attn)
```

其中：

- `q` 表示 query。
- `k` 表示 key。
- `k^T` 表示 key 的最后两个维度转置。
- `scale` 表示缩放系数，用来稳定 attention 数值。
- `relative_position_bias` 表示窗口内部相对位置偏置。
- `shift_mask` 只在 shifted window attention 中使用，用来避免不该相互注意的位置发生混合。
- `softmax` 表示把 attention 分数归一化成权重。

## 10. 36×36 到 72×72 上采样

当前 `EncoderFeatureUpsampler` 的实现比较直接：

```text
refined_feature_36 [B, C, 256, 36, 36]
  → bilinear interpolate 到 72×72
  → 与 original_encoder_72 拼接
  → 1×1 Conv(512→256)
  → refined_encoder_features_72 [B, C, 256, 72, 72]
```

说明：

- 当前 `sam_fpn_72` 参数保留在接口中，但在 `EncoderFeatureUpsampler` 内部没有参与融合。
- 1×1 融合卷积初始化为偏向原始 encoder feature 的 passthrough，使训练开始时更稳定。
- 输出的 `refined_encoder_features_72` 会写回 SAM3 的 `encoder_hidden_states` 图像 token 区域。

## 11. 写回 SAM3 并输出最终 logits

`Sam3Image.run_encoder_refiner()` 会完成最终输出流程：

```text
refined_encoder_features_72
  → 按类别 chunk 切分
  → 写回每个 chunk 的 encoder_hidden_states
  → frozen segmentation_head
  → chunk_logits
  → concat
  → final_logits [B, C, H_out, W_out]
```

写回逻辑只替换 `encoder_hidden_states` 中的图像 token 区域，prompt token 和其他结构保持不变。

## 12. Loss 设计

训练时只监督最终输出：

```text
total_loss = final_bce_weight × BCE + final_dice_weight × Dice
```

其中：

- `total_loss` 表示总损失。
- `final_bce_weight` 表示 BCE 损失的权重。
- `BCE` 表示 binary cross entropy，用来监督每个类别的二值 mask。
- `final_dice_weight` 表示 Dice 损失的权重。
- `Dice` 表示 Dice loss，用来衡量预测 mask 和真实 mask 的重叠程度。

当前配置里：

```text
final_bce_weight = 1.0
final_dice_weight = 0.0
```

因此实际训练主要使用 BCE。

BCE 的目标是 per-class binary mask。也就是说，对于每个类别，模型都学习一个“该像素是否属于这个类别”的二分类 mask。最后推理时再通过 `argmax` 选出每个像素分数最高的类别。

## 13. Debug 输出

当 `return_debug=True` 时，主要调试输出包括：

| key | 形状 | 含义 |
|---|---|---|
| `final_logits` | `[B, C, H_out, W_out]` | 最终 mask logits。 |
| `encoder_features` | `[B, C, 256, 72, 72]` | 原始 encoder feature。 |
| `refined_encoder_features` | `[B, C, 256, 72, 72]` | refiner 更新后的 encoder feature。 |
| `refiner_features_36` | `[B, C, 256, 36, 36]` | refiner 在 36×36 上输出的 feature。 |
| `score_embed_36` | `[B, C, 256, 36, 36]` | CLIP score 和 SAM score 融合后的 score embedding。 |
| `clip_score_embed_36` | `[B, C, 192, 36, 36]` | CLIP score embedding。 |
| `sam_score_embed_36` | `[B, C, 64, 36, 36]` | SAM mask prior score embedding。 |
| `clip_score_maps` | `[B, C, 32, 36, 36]` | 32 个模板对应的图文相似度 map。 |
| `template_clip_text_features` | `[C, 32, D_clip]` | 每个类别、每个模板对应的 CLIP 文本特征。 |
| `clip_mid_features` | `List[[B, D_native, 36, 36]]` | RemoteCLIP 中间层特征，当前主路径不直接使用。 |

默认不保存完整 `sam_prior_logits`，因为它可能占用较多显存。

## 14. 主要文件说明

```text
models/
  openclip_image_encoder.py
    RemoteCLIP dense image encoder，负责 504×504 输入、36×36 输出、positional embedding 插值和 dense value-branch last block。

  openclip_text_encoder.py
    RemoteCLIP text encoder wrapper，负责类别 prompt templates 编码。

  score_embeddings.py
    ClipScoreEmbedding、SamMaskScoreEmbedding、CombinedScoreEmbeddingBuilder。

  encoder_refiner_attention.py
    ClassScoreAttention、WindowScoreAttention、EncoderRefinerLayer。

  encoder_refiner.py
    EncoderFeatureUpsampler、ClassConditionedEncoderRefiner。

  sam3_image.py
    主流程协调器，负责构建 refiner cache、生成 SAM score embedding、运行 refiner、写回 encoder_hidden_states、生成 final_logits。

  segmentor.py
    SAM3Segmentor wrapper，负责训练/推理模式下的输出适配。

  adapters/semantic_adapter.py
    语义分割输出 adapter，训练时返回 final_logits，推理时返回 score map 和 final_pred。

  task_modes.py
    输出 key 和 task mode 定义。

losses/
  semantic_criterion.py
    语义分割损失函数。

configs/
  ovrs_sam3_isaid_loveda_base.py
  ovrs_sam3_isaid_loveda_exp.py
  ovrs_sam3_isaid_loveda_full.py

config_dataclasses.py
  配置 dataclass 定义。

model_builder.py
  模型、criterion、hooks、训练组件构建逻辑。
```

## 15. 最小 shape 验收

| 张量 | 期望形状 |
|---|---|
| `remoteclip_feat_map` | `[B, 768, 36, 36]` |
| `clip_score_maps` | `[B, C, 32, 36, 36]` |
| `clip_score_embed_36` | `[B, C, 192, 36, 36]` |
| `sam_score_embed_36` | `[B, C, 64, 36, 36]` |
| `score_embed_36` | `[B, C, 256, 36, 36]` |
| `feature_36` | `[B, C, 256, 36, 36]` |
| `refiner_features_36` | `[B, C, 256, 36, 36]` |
| `refined_encoder_features_72` | `[B, C, 256, 72, 72]` |
| `final_logits` | `[B, C, H_out, W_out]` |

## 16. 推荐检查命令

```bash
python -m py_compile models/openclip_image_encoder.py
python -m py_compile models/openclip_text_encoder.py
python -m py_compile models/score_embeddings.py
python -m py_compile models/encoder_refiner_attention.py
python -m py_compile models/encoder_refiner.py
python -m py_compile models/sam3_image.py
python -m py_compile models/segmentor.py
python -m py_compile models/adapters/semantic_adapter.py
python -m py_compile models/task_modes.py
python -m py_compile losses/semantic_criterion.py
python -m py_compile config_dataclasses.py
python -m py_compile model_builder.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_base.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_exp.py
python -m py_compile configs/ovrs_sam3_isaid_loveda_full.py
```

如果只检查本次 refiner 相关修改，至少运行：

```bash
python -m py_compile models/encoder_refiner_attention.py
python -m py_compile models/encoder_refiner.py
```

## 17. 当前实现重点

当前代码里，feature 和 score embedding 在 refiner 中是并行更新的：

```text
同一套 attention 权重
  → 更新 feature
  → 更新 score_embed
```

但两路 value projection 是分开的：

```text
v_feature = feature
v_score   = score_embed
```

因此 attention 决定“看哪里”，feature 分支和 score 分支分别决定“更新什么内容”。

当前残差更新是普通 Transformer 风格：

```text
x = LayerNorm(x + Dropout(update))
```

其中：

- `x` 可以是 feature，也可以是 score embedding。
- `update` 表示 attention 或 FFN 产生的更新量。
- `Dropout` 用于训练时随机丢弃部分更新，降低过拟合风险。
- `LayerNorm` 用于稳定每一层输出的特征分布。

这个设计保留了 feature 与 score 的协同更新，同时避免额外引入可学习残差系数，结构更直接。