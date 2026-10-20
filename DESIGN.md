# OVRS-SAM3 设计说明

适用分支：`master`
项目仓库：`jk-jin/ovrs-sam3`
当前任务：开放词汇遥感语义分割

> 本文描述项目当前采用的模型与训练设计。代码实现发生结构性变化时，应同步更新本文。

## 1. 项目目标

OVRS-SAM3 接收一批遥感图像和当前数据集的类别名称，输出每个类别的像素级分割 logits。项目组合三类能力：

* SAM3 提供稳定的多尺度图像特征、文本提示编码、类条件 transformer encoder 和分割解码器。
* RemoteCLIP 提供面向遥感场景的局部图文对齐。
* Encoder refiner 在低分辨率融合两者，通过单层 72×72 语义—细节双路融合生成掩码。

当前对外仍提供 semantic 任务模式，内部同时使用冻结的 SAM3 原始实例掩码与可训练的语义分支。不提供独立实例任务或 hybrid 模式，不支持非空几何提示训练。

整体流程如下：

```text
图像与类别名称
  ├─ SAM3 图像 backbone → 288/144/72 多尺度 FPN
  ├─ SAM3 文本编码器与 transformer encoder layer 1..6
  │    → 每个图像-类别对的完整 6 层 encoder feature
  └─ RemoteCLIP
       ├─ 504×504 图像 → 36×36 dense image feature
       └─ 每类 64 个文本模板 → template text feature

完整 6 层 encoder feature
  → prompt cross-attention
  → 72×72 cross-attended encoder feature

RemoteCLIP 局部相似度图
  → 64 通道模板分数图经 1×1 Conv 投影
  → 与 CLIP dense feature map 分别 L2 归一化后融合
  → 两次拼接前均做逐像素通道 L2 归一化
  → clip_score_embed_36 [B, P, 256, 36, 36]

72×72 cross-attended encoder feature
  → 双线性下采样
  → base_feature_36 [B, P, 256, 36, 36]
  → 直接作为初始 feature_36

clip_score_embed_36 直接作为初始 score_embed_36，不接收 SAM3 FPN 注入

feature_36 + score_embed_36
  → Refiner layer 1..4（A→B→A→B，全提示同时运行）
  → refiner_features_36 [B, P, 256, 36, 36]

随后按 prompt_chunk_size 逐提示块执行：

原始解码路径（始终执行，全程 torch.no_grad()；语义输出按需用于蒸馏）：

original_encoder_feature_72_chunk
  → 冻结的 SAM3 Pixel Decoder（内部继续使用原始 FPN144/FPN288）
  → 冻结的 SAM3 semantic_seg_head
  → sam3_teacher_logits（蒸馏开启时保留，detached）
  → 同一 pixel feature 经 instance_seg_head + 原始查询的 MaskPredictor，生成冻结实例掩码

学生：

refiner_feature_36_chunk + original_encoder_feature_72_chunk + sam_fpn_72
  → 双线性插值 36→72
  → 单层 DecoderInputFusion：语义支路（Refiner + 原始 encoder72），细节支路（Refiner + FPN72）
  → fused_feature_72 [B×P_chunk, 256, 72, 72]

fused_feature_72
  → 替换原始 Pixel Decoder 最后一层 FPN 输入
  → 冻结的 SAM3 Pixel Decoder 与 semantic_seg_head（梯度开启）
  → final_logits_chunk

学生和教师共用同一个冻结 segmentation_head 对象，参数不复制。
每个 chunk 立即计算 loss 和 backward，不再拼接全类别 logits。
```

## 2. 张量约定

| 记号       | 含义                             |
| -------- | ------------------------------ |
| `B`      | batch 中的图像数                    |
| `M`      | 原始前向类别数（背景排除后、逗号拆分前）            |
| `P`      | 逗号拆分后的展开提示数                     |
| `P_chunk` | 当前提示块中的提示数                      |
| `K`      | 每类 RemoteCLIP 文本模板数，固定为 64     |
| `D`      | SAM3 hidden dimension，固定为 256  |
| `D_clip` | RemoteCLIP 投影维度，ViT-L/14 为 768 |
| `L`      | Refiner 层数，当前固定为 4              |

当前固定输入下的主要张量为：

| 张量                            | 形状                                 | 说明                             |
| ----------------------------- | ---------------------------------- | ------------------------------ |
| `backbone_fpn`                | `[B, 256, 288/144/72, 288/144/72]` | SAM3 多尺度图像特征，顺序固定为 288、144、72 |
| `cross_attended_encoder_features_72` | `[B, P, 256, 72, 72]`       | 完整 6 层 encoder 与 prompt cross-attention 后的提示条件特征 |
| `sam_fpn_288`                  | `[B, 256, 288, 288]`               | SAM3 backbone 图像级 FPN288，只在原始 Pixel Decoder 内使用 |
| `sam_fpn_144`                  | `[B, 256, 144, 144]`               | SAM3 backbone 图像级 FPN144，只在原始 Pixel Decoder 内使用 |
| `sam_fpn_72`                   | `[B, 256, 72, 72]`                 | SAM3 backbone 图像级 FPN72，用于单尺度融合的细节支路 |
| `remoteclip_feat_map`         | `[B, 768, 36, 36]`                 | RemoteCLIP dense image feature |
| `template_clip_text`          | `[P, 64, 768]`                     | 每提示 64 个模板的文本特征                |
| `clip_score_maps_36`          | `[B, P, 64, 36, 36]`               | 局部图文相似度图                       |
| `clip_score_embed_36`         | `[B, P, 256, 36, 36]`              | 纯 RemoteCLIP score embedding，直接作为 Refiner 初始 score stream |
| `score_embed_36`              | `[B, P, 256, 36, 36]`              | 经 Refiner 更新后的 score stream |
| `refiner_features_36`         | `[B, P, 256, 36, 36]`              | Refiner 的图像特征流                 |
| `fused_feature_72`            | `[B×P_chunk, 256, 72, 72]`（逐 chunk） | DecoderInputFusion 输出，替换 Pixel Decoder 的最后一层 FPN 输入 |
| `final_logits`                | `[B, P_chunk, 288, 288]`（逐 chunk） | 学生路径输出的最终语义分割 logits |
| `sam3_teacher_logits`         | `[B, P_chunk, 288, 288]`（逐 chunk） | 教师路径输出的 detached teacher logits |

训练损失和评测会在必要时用最近邻插值把标签映射到 logits 尺度。

## 3. SAM3 分支

### 3.1 图像特征

SAM3 接收 1008×1008 的标准化图像。ViT patch size 为 14，主干 token grid 为 72×72。SimpleFPN 产生 288×288、144×144、72×72 和 36×36 四级特征；当前 `scalp=1` 丢弃最低分辨率的 36×36 级，因此主路径保留前三个尺度。

SAM3 图像 backbone 在训练中冻结并运行于 `eval()`。图像特征使用 `torch.no_grad()` 计算并 detach。

### 3.2 类条件 encoder 与提示展开

原始类别名称支持逗号分隔的多个提示词。例如 `"ship, vessel"` 展开为两个独立提示，`"bridge"` 保持一个。展开后的提示按 `prompt_chunk_size` 分块（默认每块 4 个提示），以控制显存。

模型在展开后的提示空间运行。训练标签通过 `prompt_to_class_id` 把同一原始类别的所有提示映射到相同标签；推理时通过像素级最大值合并回原始类别。

每个图像与每个提示组成一个 prompt pair。冻结的 SAM3 文本编码器和 6 层 transformer encoder 为每个 pair 生成类条件图像特征。所有 6 层在 `torch.no_grad()` 中一次运行完毕。

原始 encoder memory 随即供冻结的原始六层查询 decoder 使用；每个图像—提示对生成 200 个 256 维查询，以及一个独立的 presence token 特征。decoder 保持 `eval()`，显式 `apply_dac=False`，训练和推理都只使用这 200 个候选查询。查询 decoder 不读取经过分割头 prompt attention 或 Refiner 更新的特征。

完整 encoder 输出后，执行一次 prompt cross-attention（同样在 `no_grad()` 中），得到 cross-attended full-encoder feature。Refiner 不接收 SAM3 文本 token 均值；原始 SAM3 encoder 和前置 prompt cross-attention 仍正常使用完整文本提示。

所有提示块按原始顺序重新拼接。

### 3.3 共享冻结 Pixel Decoder

同一套冻结 Pixel Decoder 与 semantic head 同时服务学生和教师两条路径，参数不复制。

```text
类条件 72×72 特征（学生为 fused_feature_72，教师为原始 encoder72；替换 FPN 最后一层）
  → 上采样到 144×144 + FPN144 → 3×3 Conv + GroupNorm + ReLU
  → 上采样到 288×288 + FPN288 → 3×3 Conv + GroupNorm + ReLU
  → pixel_feature_288
  → 冻结 semantic_seg_head
```

Pixel Decoder 内部继续使用 `interpolation_mode="nearest"`（SAM3 原始设置），FPN 顺序固定为 288、144、72。Pixel Decoder 参数冻结且保持 `eval()`。

原始路径把未被 Refiner 更新的 encoder72（已经过一次 prompt attention）送入 Pixel Decoder，整个调用位于 `torch.no_grad()` 中，始终执行以生成实例掩码。需要蒸馏时，复用这次 pixel decode 的语义输出，不增加第三次 Pixel Decoder 调用。学生路径把 `fused_feature_72` 送入同一个 Pixel Decoder，梯度开启：冻结参数不更新，但梯度可以穿过冻结卷积回传到 DecoderInputFusion 和 Refiner。

## 4. RemoteCLIP 分支

### 4.1 Dense 图像编码

RemoteCLIP 使用 ViT-L/14。原始图像单独缩放到 504×504，并使用 CLIP mean/std 归一化，得到 36×36 patch grid。

前面的 transformer blocks 正常执行；最后一个 block 使用 dense value-branch：

1. 计算 QKV 投影；
2. 只取 V 分支；
3. 经过 attention output projection；
4. 向空间 token 注入 class token 信息；
5. 执行 MLP 残差；
6. 经过 `ln_post` 和原始 visual projection。

最终输出 `[B, 768, 36, 36]`。配置 `image_intermediate_layers=[7, 15]` 使用零基索引，
对应第 8、16 层；中间特征保留原生通道，ViT-L/14 为 1024，不能按最终投影的 768 通道处理。
第 8、16 层中间特征固定参与 score 类内注意力的 Q/K 指导。
训练且 RemoteCLIP 图像分支允许微调时，在提取与缓存处保留计算图；debug 输出仍 detach。
验证或冻结图像编码时不启用梯度。不额外运行 RemoteCLIP。

### 4.2 模板文本编码

每个类别使用 64 个固定遥感文本模板，生成 `[C, 64, 768]` 的模板特征。文本编码支持 micro-batch 和 non-reentrant activation checkpoint。

缓存规则必须服从参数是否可训练：

* RemoteCLIP 文本分支完全冻结时，可以缓存 detach 后的模板特征。
* 文本分支可训练且全局梯度开启时，每个训练 step 重新编码并保留计算图。
* 验证位于 `torch.no_grad()` 中，可以在一次验证过程中复用当前权重对应的缓存。

不能用模块的 `training` 属性判断是否需要梯度，因为 RemoteCLIP 在部分微调时仍保持 `eval()`。

### 4.3 Score embedding

64 个模板文本特征和 36×36 dense RemoteCLIP 图像特征分别做
L2 归一化，逐像素计算余弦相似度并乘固定系数 20，得到
[B, C, 64, 36, 36] 模板分数图。

模板分数图展平 batch 与类别维后，经过 64→256 的 1×1 Conv、
GroupNorm 和 GELU，得到中间特征 1。

中间特征 1 与 RemoteCLIP dense feature map 在每个空间位置分别沿
通道维执行 L2 归一化。归一化后的 256 通道中间特征与归一化后的
768 通道 CLIP 特征拼接，经 1024→256 的 1×1 Conv、GroupNorm 和
GELU 得到中间特征 2。

中间特征 1 与中间特征 2 再次分别执行逐像素通道 L2 归一化，
拼接为 512 通道。随后依次经过普通 3×3 Conv 512→256 和普通
3×3 Conv 256→256；每层卷积后均使用 GroupNorm 和 GELU。

最终输出 [B, C, 256, 36, 36] 的 clip_score_embed_36。该特征不再
接收 SAM3 FPN 注入，直接作为 Refiner 的初始 score stream。

## 5. Class-conditioned encoder refiner

Refiner 在 36×36 上同时维护图像 feature 流和 score embedding 流。默认使用 4 层、8 个 attention heads、12×12 窗口和 6 像素 shift。

### 5.1 Feature stream 与 score stream 初始化

Cross-attended full-encoder feature（72×72）双线性下采样到 36×36，得到
`base_feature_36`，直接作为 feature stream。

score_embeddings.py 生成的 clip_score_embed_36 直接作为 score
stream。进入所有 Refiner Attention 层之前不再融合任何 SAM3 FPN
特征，也不再设置额外残差系数。

SAM3 FPN72 用于后续单尺度 DecoderInputFusion 的细节支路，并固定下采样到 36×36，
为更新 feature 的类内注意力提供 Q/K 指导，仍不注入初始 score embedding。
FPN144 和 FPN288 由原始 Pixel Decoder 内部消费。

### 5.2 固定图像指导与标准 FFN

四层按 A→B→A→B 执行，每层始终先更新 score stream，再更新 feature stream，
最后执行 Feature FFN 和 Score FFN。各层注意力参数独立。
不保留四个可选指导开关，注意力不使用 CLIP 或 SAM3 文本 token 指导。
原始 CLIP 文本模板编码仍用于生成初始 score embedding。

| 层 | score 更新 | feature 更新 | 最后 |
| --- | --- | --- | --- |
| 1、3（A） | 类内注意力，固定 CLIP 中间层指导 | 类间注意力 | Feature FFN→Score FFN |
| 2、4（B） | 类间注意力 | 类内注意力，固定 FPN36 指导 | Feature FFN→Score FFN |

类内注意力分别对 score、feature 和图像指导做逐像素通道 LayerNorm，
Q/K 按 `[score_embed, feature, guidance]` 拼接后独立线性投影。
两种类间注意力都只拼接 `[score_embed, feature]`。
两路均为 256 通道时，类内 Q/K 输入为 768，类间为 512，投影后为 256 通道。
Value 始终仅来自指定的目标流，输出直接残差到同一条流。
后续子步骤使用已经更新的 score 或 feature，重新计算 pre-norm、Q/K/V 和注意力权重。

**固定图像指导：**

* score 类内：配置 `image_intermediate_layers=[7, 15]` 对应 CLIP 第 8、16 层原生特征。
  两层沿通道拼接，经一个共享 1×1 Conv 到 256 通道，每次 Refiner 前向只计算一次，
  再从图像维广播到全部类别提示。ViT-L/14 中间特征为 1024 通道，卷积输入为 2048。
* feature 类内：取冻结的 `backbone_fpn[2]`，从 72×72 双线性下采样到 36×36
  （align_corners=False），保留 256 通道，不增加通道投影，再广播到全部类别提示。
* 两种指导都是必需输入，缺失特征、原生通道错误或缺少中间层索引 7/15 时直接报错。

**类内空间实现继续可配置：**

| 配置 | 取值 | 含义 |
| --- | --- | --- |
| `score_intra_attn_type` | `window` / `local_3x3` | 更新 score 的类内实现 |
| `feature_intra_attn_type` | `window` / `local_3x3` | 更新 feature 的类内实现 |

基础配置当前均为 `local_3x3`；通用 dataclass 默认值为 `window`。窗口模式执行一次普通窗口与一次移位窗口，沿用窗口相对位置偏置和 shift mask。
`local_3x3` 执行四次连续普通逐像素 3×3 注意力：步长 1，无空洞或采样间隔，
含中心及八个相邻位置，每头九个独立相对位置偏置，图像外的 key 在 softmax 前屏蔽。
每个空间子步骤的投影、LayerNorm、相对位置偏置独立，图像指导在所有子步骤中固定使用。
四次局部子步骤属于一层内部，不增加 Refiner 总层数。

**标准双流 FFN：**

Feature FFN 和 Score FFN 分别处理各自当前注意力后的 256 通道特征。
每路执行通道 LayerNorm(256)、独立逐 token `256→1024→256` FFN（GELU 和 dropout），
输出直接残差到对应当前流。不拼接模块入口初始特征或本层入口特征。
全部层结束后仅对 feature_36 执行一次通道 LayerNorm；最终 score 不加输出 LayerNorm。
图像子层沿用 non-reentrant checkpoint，显式传入固定 CLIP 中间层和 FPN36 指导。

### 5.3 单尺度 Decoder Input Fusion

Refiner 在全部提示上统一执行后，最终 36×36 feature stream 先经过一次通道 LayerNorm，
再进入 `DecoderInputFusion`。融合只在 72×72 执行一次，输出 256 通道 72×72 特征，
替换原始 SAM3 Pixel Decoder 的最后一层 FPN 输入。

**输入**（`N = B×P_chunk`）：

- refiner_feature_36 `[N, 256, 36, 36]`
- original_encoder_feature_72 `[N, 256, 72, 72]`
- sam_fpn_72 `[B, 256, 72, 72]`

Refiner 特征先双线性插值到 72×72（`align_corners=False`）。这里使用缓存中的
`cross_attended_encoder_features_72` 作为原始 encoder72，它已经过完整 6 层 encoder 和一次
前置 prompt cross-attention，且未被 Refiner 更新。

四路独立的 256→128 投影（`1×1 Conv + GroupNorm`，无激活）：

```text
semantic_refiner_proj: refiner_72  [N, 256, 72, 72] → [N, 128, 72, 72]
detail_refiner_proj:   refiner_72  [N, 256, 72, 72] → [N, 128, 72, 72]
encoder_proj:          encoder_72  [N, 256, 72, 72] → [N, 128, 72, 72]
fpn_proj:              sam_fpn_72  [B, 256, 72, 72] → [B, 128, 72, 72]
```

语义和细节分支各自使用独立的 Refiner 投影（`semantic_refiner_proj` 和 `detail_refiner_proj`），参数不共享。

FPN 先按图像投影到 128 通道，再通过广播与 Refiner 逐提示相加。不使用 `repeat` 或 `repeat_interleave` 复制 256 通道 FPN。

两条独立支路（不能共享参数），结构相同：

```text
block: 普通 3×3 Conv (128→128) → GN(8,128) → GELU
     → 1×1 Conv (128→128) → GN(8,128)
output = input + block(input)   # block 内部残差
```

语义支路融合独立 Refiner 语义投影与原始 encoder72；细节支路融合独立 Refiner 细节投影与原始 FPN72。

两条支路分别使用独立的 128→256 `1×1 Conv`（无 norm、无激活）恢复到 256 通道。

两路直接相加，再经过一个 256→256 的 `1×1 Conv`（`fusion_out_proj`）输出：

```python
fused_out = semantic_out + detail_out
output = fusion_out_proj(fused_out)
```

模块不包含融合系数、门控或零初始化，输出端不加 GroupNorm、GELU、ReLU，也不加原始 encoder 残差。融合模块只有这一层，不存在多尺度堆叠。

整个模块使用一次 non-reentrant checkpoint（`self.training and self.use_checkpoint` 时开启）。支路通道数固定为 128，SAM3 通道数固定为 256。

## 6. 冻结 SAM3 分割头与梯度边界

Prompt cross-attention 在完整 6 层 encoder 之后、Refiner 之前执行一次（通过 `apply_prompt_cross_attention()`），位于 `torch.no_grad()` 中。融合之后不再追加 prompt cross-attention。

Pixel Decoder 与 semantic head 的参数始终冻结（`requires_grad=False`）并保持 `eval()`。学生和教师复用同一个 `segmentation_head` 对象，参数不复制。

- **原始路径**：原始 encoder72 送入 Pixel Decoder，整个调用位于 `torch.no_grad()` 中，生成 detached 实例掩码；需要蒸馏时同时保留 detached 语义教师 logits。学生计算图此时尚未建立或已释放。
- **学生路径**：`fused_feature_72` 送入同一个 Pixel Decoder，不使用 `no_grad()`、不 detach，也不主动 `enable_grad()`，完全服从外层的梯度状态。

冻结参数的 `requires_grad=False` 只阻止参数更新，不阻止对输入求梯度。学生路径误加 `no_grad()` 会同时切断主 BCE 和蒸馏对 DecoderInputFusion、Refiner 的监督。

DecoderInputFusion 消费 Refiner36、原始 encoder72 和 FPN72；Pixel Decoder 内部继续使用原始 FPN144 与 FPN288。教师路径不消费融合输出。

## 7. 训练设计

### 7.1 冻结与微调

以下 SAM3 模块冻结并保持 `eval()`：

* backbone；
* transformer encoder（完整 6 层，在 `no_grad()` 中执行）与原始六层查询 decoder；
* `dot_prod_scoring`、decoder 内的 presence token、归一化和 presence MLP；
* geometry encoder；
* segmentation head（Pixel Decoder 参数冻结并保持 `eval()`。教师路径在 `no_grad()` 中执行，学生路径在梯度开启状态下执行。semantic head 在教师路径中产生 detach 的 teacher logits，在学生路径中产生可回传梯度的 student logits）。

完整 SAM3 encoder 和前置 prompt cross-attention 不保留计算图，均在 `torch.no_grad()` 中执行。

`core.encoder_refiner` 完整训练。其内部的 Refiner 层、`feature_output_norm`、score embedding 和 `DecoderInputFusion` 同属一个参数组，由 `trainable_modules=["core.encoder_refiner", "core.instance_score_adapter"]` 中的前者覆盖，使用基础学习率 `1e-4`。训练使用的语义 logits 由冻结的 SAM3 `semantic_seg_head` 产生。新增 `core.instance_score_adapter` 使用同一基础学习率训练，只有两层交叉注意力及其 LayerNorm，无额外 FFN 或查询自注意力。

RemoteCLIP 图像和文本分支默认使用 `attention` 微调模式，仅训练注意力 Q/V 与位置嵌入，同时保持 `eval()` 以关闭 dropout 和 patch dropout。

OpenCLIP 常把 Q/K/V 存在同一个融合参数中。项目对该参数注册梯度 mask，使 K 区域梯度为 0；同时把整个融合参数组的 weight decay 强制设为 0。恢复 optimizer 状态后会重新应用这一不变量。

默认 AdamW 基础学习率为 `1e-4`：

* encoder refiner 使用 1.0 倍学习率；
* RemoteCLIP text/image 使用 0.01 倍学习率，即 `1e-6`；
* normalization 参数不使用 weight decay；
* 梯度裁剪上限为 0.1；
* warmup 保持前 1000 步，线性从 0.1 倍到全额学习率，后续余弦衰减。

### 7.2 数据增强

训练图像确定性短边缩放到 1008，再随机裁剪 1008×1008，`cat_max_ratio=1.0`（不限制单一类别占比）。SSD 风格颜色增强（亮度、对比度、饱和度、色相各以 0.5 概率独立启用，色相使用 `/180` 的 HSV 归一化单位），`image` 和 `raw_image` 共享同一次采样参数。只使用 0.5 概率水平翻转。不使用随机尺度、垂直翻转和 90° 旋转。

验证和测试不应用颜色增强或随机裁剪。TTA 默认关闭。

### 7.3 损失

每个类别通道独立使用 binary mask 监督，不使用跨类别 softmax。

**主损失：朴素 BCE**（监督 `final_logits`）：

所有有效像素（label ≠ 255）等权参与全局均值。不做正负像素分离、不按类别是否出现分组加权。每个 `[B, P, H, W]` 位置只要 label ≠ 255 就对损失有相同贡献。多提示类别的所有提示使用同一原始标签作为监督，每个提示通道等权。

```python
# 全局分母 = Σ valid_pixels × P
bce_per_pixel = BCEWithLogits(final_logits, target)
loss_final_bce = (bce_per_pixel * valid_mask).sum() / total_valid_pixels
```

标签 255 被排除（不参与 BCE），对所有提示统一处理。

**Dice 损失**（`final_dice_weight=0.0`）：只对图像中存在的提示计算，默认关闭。开启时按全局 `N_present` 做逐 chunk 贡献归一化。

**SAM3 teacher 掩码蒸馏**（基础配置 `sam3_mask_distill_weight=0.5`）：

蒸馏权重在整个训练过程中保持不变。原始实例路径始终执行；当前 batch 存在可蒸馏提示且蒸馏启用时，额外保留该路径的 teacher logits。

蒸馏监督范围：

1. 冻结的 SAM3 semantic head 产生的 teacher logits 做 sigmoid，得到 soft probability 目标。
2. student 使用 raw `final_logits`。
3. 用 `binary_cross_entropy_with_logits` 逐像素计算蒸馏损失。
4. 只对 GT 中存在的图像—提示对计算。
5. 所有 `label != 255` 的有效像素参与蒸馏。
6. 对每个存在原始类别，额外包含 GT 外侧指定宽度的边界环。
7. 外侧边界环只保留 `label == 255` 的位置。
8. 外环宽度由 `sam3_mask_distill_boundary_width` 控制，基础配置为 3。
9. 宽度 0 表示不增加外环，仅蒸馏全部有效像素。
10. 远离物体的 255 区域不参与蒸馏。
11. 不存在类别不参与蒸馏。
12. teacher 和 student 都在 288×288 分辨率，不做尺度变换。
13. teacher 必须 detach。
14. 分母是全部存在提示对应的有效像素与类别外环像素总数（全局分母，不按 chunk 单独计算）。
15. 多个提示映射到同一类别时复用相同外环，并在全局分母中按提示独立计数。

总损失：

```python
total_loss = (
    1.0 * loss_final_bce
    + 0.0 * loss_final_dice
    + 0.5 * loss_sam3_mask_distill_bce
    + 0.1 * loss_instance_presence_bce
    + 0.1 * loss_instance_score_bce
)
```

`loss_sam3_mask_distill_bce` 为未经权重的原始蒸馏 BCE；`loss_sam3_mask_distill_weighted` 为真正加入总损失的加权贡献。蒸馏权重在整个训练期间固定不变。

上面的数值取自 `configs/_base_/model/ovrs_sam3.py` 的当前取值（蒸馏权重 0.5、外环宽度 3）。`config_dataclasses.py` 中的通用默认值不同，实验实际使用的数值取决于最终合并配置。

**训练显存设计**：

* Refiner36 对全部提示一次计算。
* 高分辨率按 chunk 计算。
* 每个 chunk 立即计算 loss 和 backward。
* 使用 detached leaf proxy 累积高分辨率梯度。
* 所有 chunk 完成后把 proxy.grad 一次传回真实 Refiner 图。
* optimizer/scaler/scheduler 每个 batch 只更新一次。
* 不保留多个 chunk 的融合特征或 logits。
* 不使用 `retain_graph=True`。

## 8. 推理与评测

推理时模型在提示空间输出 `prompt_logits` [B, P, H, W]。Adapter 先对语义 logits 做 sigmoid，再与筛选后的实例掩码概率逐像素取最大值，得到 `raw_prompt_score_map`，再通过像素级最大值把同一原始类别的所有提示合并为 `raw_final_score_map` [B, M, H, W]。可选的逐类别相对阈值在原始类别空间执行，归一化后只把未通过位置置 0；保留位置继续使用原始 sigmoid 分数。最终对类别维取 argmax。

标签空间中两个机制职责不同：

* `reduce_zero_label` 用于从数据集标签空间彻底删除原始 0 类，并把其余类别只重映射一次；类别名称、前向通道和评测元数据必须使用同一重映射结果。
* `background_cfg` 用于有实际背景语义的数据集。背景可以不进入模型前向，但仍属于评测类别空间，并由统一后处理映射回来。

两条路径不能对同一标签连续执行两次 0 类删除或索引平移。

Evaluator 输出整体 mIoU、mAcc、pixel accuracy 和逐类别指标。`metric_groups` 可以按 `class_ids` 或 `class_names` 定义命名类别组，并分别计算组内 mIoU/mAcc。完整 iSAID→LoveDA 配置使用前景类别组作为 checkpoint monitor。

TTA 当前只支持 `scale=1.0` 和空间翻转。多个视图必须先平均 `raw_prompt_score_map`（提示空间），再合并提示到原始类别，最后统一执行一次非线性相对阈值过滤。

## 9. Checkpoint、恢复与实验追踪

训练只在显式提供 `--resume-from` 时恢复完整状态，不自动扫描 `work_dir`。未提供该参数时不会加载任何已有训练产物；若目标目录已经包含训练 checkpoint，则直接报错，避免混合实验。

完整训练 checkpoint 自包含：

* `global_iter`；
* model、optimizer、AMP scaler 和 scheduler；
* Python、NumPy、Torch CPU 与各 CUDA device 的 RNG 状态；
* 可恢复随机 batch sampler 的排列、增强种子、游标和 generator 状态；
* checkpoint manager 的 best score；
* train/validation 统计及 validation 状态。

实验追踪状态（W&B）不保存在 checkpoint 中，每次程序启动创建新的 W&B run。`trainer/global_iter` 仍然作为 W&B 图表横轴。

NumPy RNG 数组以 Tensor 保存，因此统一加载入口可以安全使用 `torch.load(..., weights_only=True)`。写入 iteration checkpoint、`latest.pth` 和 `best.pth` 时使用临时文件与原子替换。

`latest.pth` 只在保存或完成一次 checkpoint finalization 时更新，不随普通日志输出更新。`best.pth` 只在 monitor 指标严格改善时更新。

恢复顺序为：严格加载模型与训练状态、恢复 sampler/hook、构建 DataLoader iterator、初始化 W&B（新 run）、准备缓存，最后恢复 RNG。若 checkpoint 标记验证尚未完成，恢复后先重放该次验证，再继续训练。

W&B 每次创建新 run，不复用旧 run ID 或 `last_history_step`。JSONL 在恢复时追加，在全新训练时重建。

两种加载模式必须区分：

```bash
# 完整、严格地继续训练
python tools/train.py configs/train/isaid_loveda_full.py \
  --resume-from work_dirs/full/isaid_loveda/latest.pth

# 只加载模型参数，重新开始 optimizer、scheduler、RNG、sampler 和 W&B
python tools/train.py configs/train/isaid_loveda_full.py \
  --load-model-from /path/to/checkpoint.pth \
  --work-dir /path/to/new_work_dir

# 只评测模型参数
python tools/train.py configs/test/loveda.py \
  --eval-only \
  --load-model-from /path/to/checkpoint.pth
```

Ctrl+C 使用 Python 默认 `KeyboardInterrupt`。训练和验证均立即退出，不保存新的 checkpoint。已有周期 checkpoint 保留不变。非人工异常仍可保存 exception checkpoint。

旧格式或缺少完整运行状态的权重不能用于 `--resume-from`，但可以通过 `--load-model-from` 只加载模型参数。

训练日志中记录以下蒸馏相关项：

| 键 | 含义 |
| --- | --- |
| `loss_final_bce` | 朴素 BCE，所有有效像素等权全局均值（跨 chunk 累加后求均值） |
| `loss_sam3_mask_distill_bce` | 未经权重的原始蒸馏 BCE（跨 chunk 累加后求均值） |
| `loss_sam3_mask_distill_weighted` | 真正加入总损失的加权蒸馏贡献（跨 chunk 累加后求均值） |

## 10. 配置与主要文件

完整训练入口为：

```bash
python tools/train.py configs/train/isaid_loveda_full.py
```

关键配置：

| 文件                                            | 职责                               |
| --------------------------------------------- | -------------------------------- |
| `configs/_base_/model/ovrs_sam3.py`           | 模型、RemoteCLIP、refiner、冻结策略和 loss |
| `configs/_base_/optimizer/ovrs_sam3_adamw.py` | AdamW 参数组和学习率倍率                  |
| `configs/_base_/schedule/full_20k.py`         | 20K iteration 计划                 |
| `configs/_base_/dataloader/`                  | 公共训练/评测 DataLoader 与 transforms  |
| `configs/datasets/`                           | 数据集路径、类别与标签空间                    |
| `configs/train/isaid_loveda_full.py`          | 完整训练组合、LoveDA 指标组、W&B 与可视化       |

主要实现：

| 文件                                    | 职责                                       |
| ------------------------------------- | ---------------------------------------- |
| `models/sam3_image.py`                | 类别 chunk、缓存、SAM3 encoder、低分辨率 refiner、逐 chunk 学生/教师解码 |
| `models/encoder_refiner.py`           | 全类别 Refiner、最终 feature LayerNorm 与单尺度融合接口 |
| `models/decoder_input_fusion.py`      | 单尺度语义—细节双路融合，输出替换 Pixel Decoder 最后一层 FPN 输入 |
| `models/encoder_refiner_attention.py` | 交替单 Value 跨类别注意力、独立可选窗口/局部 3×3 类内注意力、双流 FFN、pre-norm 与直接残差更新 |
| `models/maskformer_segmentation.py`   | prompt attention、共享 Pixel Decoder、原始 semantic/instance head 和 MaskPredictor |
| `models/instance_scoring.py` | 两层查询交叉注意力、joint score 筛选和不加权的实例概率聚合 |
| `models/score_embeddings.py`          | 64 模板相似度图、归一化 CLIP 融合和空间卷积增强 |
| `models/openclip_image_encoder.py`    | 36×36 dense RemoteCLIP 图像特征              |
| `models/openclip_text_encoder.py`     | 模板文本编码、micro-batch 与梯度控制                 |
| `losses/semantic_criterion.py`        | Streaming 正负平衡 BCE、Dice 和 SAM3 teacher 蒸馏 |
| `engine/trainer.py`                   | 高分辨率逐 chunk loss/backward 与 proxy 梯度回传 |
| `engine/checkpoint.py`                | 安全、原子、严格的 checkpoint 保存与加载               |
| `engine/runtime_state.py`             | RNG 捕获与恢复                                |
| `data/resumable_sampler.py`           | 可精确恢复的数据顺序与增强种子                          |
| `engine/experiment_hooks.py`          | JSONL 与 W&B 生命周期                         |
| `engine/evaluator.py`                 | 语义指标、命名指标组、背景映射与 TTA                     |

## 11. 实现不变量与限制

修改代码时必须保持：

1. 类别 chunk 完整、无重复且按原顺序拼接。
2. SAM3 encoder、refiner 和 RemoteCLIP grid 分别固定为 72×72、36×36 和 36×36。
3. SAM3 hidden dimension 固定为 256。
4. 模板数固定为 64，RemoteCLIP 图文投影维度一致。
5. clip_score_embed_36 完全由 RemoteCLIP 模板分数图和 dense RemoteCLIP feature map 生成。
6. 模板分数中间特征与 dense CLIP 特征在拼接前必须分别沿通道维执行逐像素 L2 归一化。
7. 两路 256 通道中间特征在第二次拼接前也必须分别执行逐像素 L2 归一化。
8. 初始 score embedding 不注入 SAM3 FPN；固定将 FPN72 下采样到 36×36 用作更新 feature 的类内 Q/K 指导。
9. SAM3 FPN72 用于固定 Refiner Q/K 指导及 72 尺度融合；FPN144 和 FPN288 只在原始 Pixel Decoder 内部使用。
10. 可训练 RemoteCLIP 文本特征不能跨 optimizer step 缓存。
11. 验证不得重新开启 RemoteCLIP 图像分支的 autograd。
12. Refiner 必须先在全部提示上执行，再按 chunk 做高分辨率解码。Refiner 不能放进 chunk 循环。
13. 学生和教师复用同一个冻结 Pixel Decoder 与 semantic head；教师路径必须在 `torch.no_grad()` 中，学生路径不得使用 `no_grad()`、`detach()` 或 `enable_grad()`。
14. DecoderInputFusion 只在 72×72 执行一次，不做多层堆叠或支路前额外 encoder 拼接，输出替换原始 Pixel Decoder 的最后一层 FPN 输入。
15. 教师始终使用缓存中的原始 encoder72，不使用融合输出或被 Refiner 更新后的特征。
16. clip_score_embed_36 保持纯 RemoteCLIP 输出，用于 debug。
17. teacher 只来自原始 encoder72 解码路径并且必须 detach。teacher 和 student 都为 288×288。
18. 蒸馏只监督存在类别。每个存在提示均监督全部 GT 有效像素，并额外监督其原始类别 GT 外侧 `sam3_mask_distill_boundary_width` 像素范围内且标签为 255 的边界环。远处 255 区域不参与蒸馏。多个提示映射到同一类别时复用相同外环，并在全局分母中按提示独立计数。
19. 训练使用的语义 logits 由冻结的 SAM3 `semantic_seg_head` 产生。新增 `core.instance_score_adapter` 使用同一基础学习率训练，只有两层交叉注意力及其 LayerNorm，无额外 FFN 或查询自注意力。
20. Refiner 按 A→B→A→B 交替，每层先更新 score 再更新 feature。score 类内固定拼接 CLIP 中间层融合特征，feature 类内固定拼接 FPN36；两种类间 Q/K 都不拼接文本。所有注意力单 Value、pre-norm、直接残差。双路 FFN 各处理当前 256 通道特征，256→1024→256 后直接残差到当前流；保留最终 feature LayerNorm。
21. TTA 必须先平均提示空间分数（`raw_prompt_score_map`），再合并提示到原始类别，最后进行相对阈值过滤。
22. `reduce_zero_label` 与 `background_cfg` 各自只执行其定义的一次标签空间变换。
23. 完整恢复必须严格校验 checkpoint schema；模型权重迁移必须走独立入口。
24. 训练不使用完整 `[B,C,288,288]` 计算图。逐 chunk backward 通过 proxy leaf 隔离。
25. optimizer.zero_grad / scaler.step / scaler.update / scheduler.step 每 batch 只执行一次。
26. 不使用 `retain_graph=True`。
27. backbone_fpn 顺序固定为 `[288, 144, 72]`，即 `backbone_fpn[0]` 为 288、`[1]` 为 144、`[2]` 为 72。
28. 原始 FPN 在 256 通道时不按提示复制；FPN 先按图像投影到 128 通道，再按提示广播。
29. FPN 投影模块可训练，每个 chunk 重新计算，不跨 chunk 缓存计算图。
30. 融合支路固定 `branch_dim=128`，3×3 卷积使用普通卷积（无分组）。
31. 融合输出直接作为 Pixel Decoder 的最后一层输入；模块内无融合系数、门控、零初始化、额外残差或输出归一化。
32. 最终 logits 由冻结 `semantic_seg_head` 生成。

当前限制：

* 只支持 semantic 模式；
* 不支持非空几何 prompt；
* 不支持动态 SAM3/refiner 空间尺寸或多尺度 TTA；
* `clip_mid_features` 固定参与 score 类内 Q/K 指导，debug 输出单独 detach；
* 当前默认训练集为 iSAID，验证集为 LoveDA。
* 查询评分只能选择原始实例候选，不能修复错误轮廓或生成原始候选中不存在的目标。最大值融合只能增加类别响应。

## 12. Checkpoint 非兼容变更

Checkpoint schema 版本为 4。实验追踪状态不再保存在 checkpoint 中；`WandbHook` 不再产生持久化状态。

以下记录该分支历次非兼容变更；最新变更见本节末尾：

* 删除 `pyramid_decoder.final_fusion_288.*`（`FinalPixelFeatureFusion288`）。
* 删除旧 `pyramid_decoder.stage_*.detail_dim=64` 单细节支路。
* 新增每层 `fpn_proj`、`semantic_block`、`detail_block`、双输出投影和独立的 `detail_scale`。
* 三个 stage 从旧 64 通道单支路改成新的 128 通道语义—细节双路结构。
* 旧 stage 中同名参数（如 `refiner_proj`、`pixel_proj`）的 shape 也已变化（64→128 通道）。
* 旧 checkpoint 不能通过 `--resume-from` 严格恢复。
* 可以用 `--load-model-from` 非严格迁移未变化参数。
* 必须使用新的 work directory。
* 不创建旧参数映射或兼容层。
* `_CHECKPOINT_VERSION` 继续保持 4，因为 checkpoint 容器格式没有改变。
* 原始 SAM3 Pixel Decoder 和 semantic head 参数名称、形状保持不变。

后续一次重构中：

* 删除 `pyramid_decoder.stage_*.refiner_proj.*`（共享 Refiner 投影）。
* 删除 `pyramid_decoder.stage_*.detail_scale`（可学习残差系数）。
* 新增 `pyramid_decoder.stage_*.semantic_refiner_proj.*`（独立语义 Refiner 投影）。
* 新增 `pyramid_decoder.stage_*.detail_refiner_proj.*`（独立细节 Refiner 投影）。
* 新增 `pyramid_decoder.stage_*.fusion_out_proj.*`（256→256 最终融合 1×1 Conv）。
* 不复制旧 `refiner_proj` 到两个新投影。
* 蒸馏损失不再使用余弦衰减，权重在训练全程保持固定。
* 配置字段 `sam3_mask_distill_decay_start_iter` 和 `sam3_mask_distill_decay_end_iter` 已删除。
* `_CHECKPOINT_VERSION` 继续保持 4。

本次重构中：

* 删除 `core.encoder_refiner.fpn_score_fusion_36.*`（FPN score 注入卷积模块）。
* 删除 `core.encoder_refiner.fpn_score_injection_scale`（FPN 注入残差系数）。
* 删除 `core.encoder_refiner.clip_score_embed.score_encoder.*`（旧多尺度 score encoder）。
* 新增 `core.encoder_refiner.clip_score_embed.score_stem.*`（64→256 1×1 Conv）。
* 新增 `core.encoder_refiner.clip_score_embed.score_clip_fusion.*`（score+CLIP 融合 1×1 Conv）。
* 新增 `core.encoder_refiner.clip_score_embed.spatial_fusion.*`（空间双 3×3 Conv）。
* 模板数从 32 扩展至 64。
* RemoteCLIP score embedding 生成流程完全重构，使用两次 L2 归一化拼接融合。
* 旧训练 checkpoint 不能通过 `--resume-from` 恢复。
* 可以用 `--load-model-from` 非严格迁移未变化参数。
* 必须使用新的 work directory。
* 不创建旧参数映射或兼容层。
* `_CHECKPOINT_VERSION` 继续保持 4，因为 checkpoint 容器格式没有改变。

本次 Refiner 残差与输出归一化重构：

* 删除 `core.encoder_refiner.layers.*.class_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.class_score_scale`。
* 删除 `core.encoder_refiner.layers.*.regular_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.regular_score_scale`。
* 删除 `core.encoder_refiner.layers.*.shifted_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.shifted_score_scale`。
* 删除 `core.encoder_refiner.layers.*.ffn_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.ffn_score_scale`。
* 删除配置字段 `encoder_refiner_cfg.residual_scale_init` 及其完整构建链路。
* 删除 `residual/refiner_internal/*` 训练、JSONL 和 W&B 日志。
* 新增 `core.encoder_refiner.feature_output_norm.weight`。
* 新增 `core.encoder_refiner.feature_output_norm.bias`。
* Refiner 内部继续保持 pre-norm；所有注意力和 FFN 子层改为直接残差相加。
* 全部 Refiner 层结束后，对最终 feature_36 执行一次通道 LayerNorm。
* 最终 score_embed_36 不增加输出 LayerNorm。
* 不增加旧字段兼容、参数占位、参数映射或固定缩放常量。
* 旧 checkpoint 不能通过 `--resume-from` 严格恢复。
* 可以通过 `--load-model-from` 非严格加载未变化参数，但旧输出投影是在 LayerScale 存在时训练的，不保证删除系数后具有等价数值行为。
* 非严格加载旧 checkpoint 时，新的 feature_output_norm 使用默认初始化：weight 为 1、bias 为 0。
* 新实验必须使用新的 work directory。
* `_CHECKPOINT_VERSION` 保持为 4，因为 checkpoint 容器格式没有变化。

本次单尺度融合重构（可训练解码从三阶段金字塔改为 72×72 单尺度融合）：

* 删除 `models/refiner_pyramid_decoder.py`（`RefinerPyramidDecoder` 和 `SemanticDetailFusionStage`）。
* 删除 `core.encoder_refiner.pyramid_decoder.stage_72.*`、`stage_144.*`、`stage_288.*` 的全部参数。
* 新增 `core.encoder_refiner.decoder_input_fusion.*`（四路 256→128 投影、两条支路 block、两个 128→256 输出投影和最终 256→256 融合投影），使用常规初始化。
* 删除 `UniversalSegmentationHead.forward_semantic_pixel_pyramid()` 与 `PixelDecoder.forward_multiscale()`；`PixelDecoder.forward()` 只返回最终特征，其参数名称、数量和形状不变。
* 学生路径改为：`fused_feature_72` → 原始 Pixel Decoder → 原始 semantic head；教师路径仍使用原始 encoder72。
* 旧 checkpoint 不能通过 `--resume-from` 严格恢复。
* 可以用 `--load-model-from` 非严格迁移未变化参数（Refiner 层、`feature_output_norm`、score embedding、原始 Pixel Decoder 与 semantic head）；新融合模块使用默认初始化。
* 不把旧 stage72 权重搬到新融合模块，不添加兼容参数或兼容层。
* 新实验必须使用新的 work directory。
* `_CHECKPOINT_VERSION` 保持为 4，因为 checkpoint 容器格式没有变化。
* 本次参数结构变化意味着旧模型可以迁移部分参数，但不等于可以完整继续旧训练。


本次交替单 Value Refiner 重构：

* 四层由同构双流更新改为 A→B→A→B；各层参数独立。
* 类间 Q/K 输入从 768 改为 512 通道；类内 Q/K 仍为 512 通道，拼接顺序统一为 score→feature。
* 删除各注意力的 `v_feature_proj`、`v_score_proj`、`out_feature_proj`、`out_score_proj`。
* 每个注意力仅新增单路 `v_proj` 和 `out_proj`，目标流由所在层的类型决定。
* 删除 `class_norm_text`、SAM 文本均值计算与缓存以及 Refiner 参数传递链路。
* 保留原始 SAM3 文本编码、encoder 与 prompt cross-attention。
* 保留窗口相对位置偏置、shift mask、双流 FFN、pre-norm、直接残差及最终 feature LayerNorm。
* Score embedding、DecoderInputFusion、原始 SAM3 分割流程、损失和逐 chunk 梯度回传设计不变。
* 旧 checkpoint 不能通过 `--resume-from` 严格恢复；新实验使用新的 work directory。
* 旧 checkpoint 也不能直接使用 `--load-model-from`：当前加载器即使 `strict=False`，仍会拒绝类间 Q/K 的同名 shape 不匹配。若另行准备仅包含名称和 shape 均匹配参数的权重文件，可以加载其中的未变化参数；本次不添加旧参数映射、自动过滤或兼容层。
* 类内 Q/K 拼接顺序也发生变化，即使旧权重 shape 匹配，直接加载亦不保证数值等价；建议本次新实验从默认初始化开始。
* `_CHECKPOINT_VERSION` 保持 4，因为 checkpoint 容器格式不变。

CPU 回归验证入口：`python -m unittest discover -s tests -v`。
测试不需要下载 SAM3 或 RemoteCLIP 权重，也不依赖训练数据。

本次简化为固定图像指导版本：

* 固定保留 score 类内的 CLIP 第 8、16 层融合指导和 feature 类内的 FPN36 指导。
* 两种类间注意力都只拼接 score 与 feature，无额外 CLIP/SAM3 文本指导。
* 不恢复四个指导开关或模板注意力；保留两种类内空间模式配置。
* 删除模块入口初始特征的 FFN 拼接和传递；FFN LayerNorm、首个 Linear 输入恢复为 256 通道。
* 删除 `encoder_refiner_cfg.decoder_input_fusion_layers` 及其完整构建链路。
* 删除多层融合与支路前 `input_fusion`，恢复原来的单层语义—细节融合和参数名称。
* 初始 CLIP score embedding、原始 SAM3 流程、教师、冻结解码头、损失和逐 chunk 梯度设计保留。
* 旧 checkpoint 的 FFN 形状和融合参数名称不匹配；新实验使用新的 work directory，不提供兼容层。
* checkpoint 容器 schema 保持 4。



## 13. 冻结实例掩码与融合特征驱动的查询评分（本次新增）

### 13.1 原始掩码与新评分各用一份查询

每个提示的原始 encoder memory 先运行官方同构的六层查询 decoder，保留最后一层 200 个查询和独立 presence token。随后原始 encoder memory 经过一次分割头 prompt cross-attention，所得 encoder72 继续作为现有 Refiner 输入和原始像素解码输入。

每个高分辨率 chunk 执行两遍共享的冻结 Pixel Decoder：

1. 原始 encoder72 → Pixel Decoder → `instance_seg_head` → 原始查询经三层 `mask_predictor.mask_embed` MLP → 逐通道点积生成 200 张 288×288 实例 logits。这一整条路径无梯度。需要语义蒸馏时，在同一次 pixel decode 上复用 `semantic_seg_head` 的输出。
2. 原有 `fused_feature_72` → 同一 Pixel Decoder → `semantic_seg_head`，生成原有语义 logits。其损失、存在类别蒸馏和 GT 外侧忽略区域边界环均保持不变。

新增评分模块不参与生成实例掩码：实例轮廓始终使用原始查询。当前对外输出仍为类别语义分割结果。

### 13.2 两层交叉注意力

将一个 presence token 和 200 个原始查询拼成 201 token 序列，依次经过两层参数独立的 pre-norm 交叉注意力和残差更新。Q 来自当前 token，K/V 来自同一提示的 `fused_feature_72` 展平后的 5184 个空间 token。默认 8 heads，dropout=0，无额外 FFN、自注意力或新的评分头。

原始查询、presence 特征和融合特征在评分模块入口全部 detach。因此新增评分损失只训练 `core.instance_score_adapter` 的注意力投影与 LayerNorm；Refiner、DecoderInputFusion 和 RemoteCLIP 继续只接受原有语义损失。

更新后的实例查询送入冻结 `dot_prod_scoring`，使用原始文本提示，经过文本 MLP、文本与查询投影和点积生成匹配 logits。更新后的 presence token 送入冻结的 `presence_token_out_norm` 和 `presence_token_head`。保留原有 presence logit 限幅。两条冻结评分路径允许输入 autograd，不得放入 `no_grad()`，也不再次乘入旧 presence。

### 13.3 新增监督

所有有效查询都参与监督；不得先按预测匹配分数或 presence 筛选训练样本。不对原始实例掩码、查询特征或边界框施加新的训练损失。

**Presence BCE：** 对每个图像—提示对，从原始分辨率 GT 判断该提示对应类别是否存在：存在为 1，否则为 0。使用原始 GT 可以避免小目标因语义损失尺度的最近邻下采样消失。整张图都为 255 时跳过 presence 监督。

**匹配分数 BCE：** 语义标签仍按现有方式最近邻映射到 288×288，与实例掩码对齐。将冻结掩码 sigmoid 后按 `instance_mask_threshold` 二值化；只统计其中 GT 非 255 的区域。用“掩码覆盖的有效像素中属于当前类别的比例”作为软监督目标，范围 0 到 1。例如有效面积 100、同类交集 90，目标为 0.9。这个指标是类别 precision，不是掩码与全类别 GT 的 IoU；掩码准确覆盖多个物体中的一个时可以得到高分。

- 二值掩码为空：目标 0。
- 掩码非空但全部覆盖 255：跳过该查询。
- 不存在类别：覆盖有效区域的掩码目标均为 0。
- 每个图像—提示对内，对目标大于 0 和目标等于 0 的两组查询分别平均 BCE，再对非空组等权平均，避免大量背景查询掩盖有用候选。
- 对图像—提示对使用全局分母：原始 GT 含有效标签的图像数乘全部提示数；所有查询都未知的提示对贡献 0。presence 同样使用这个全局分母。
- 全部 chunk 的损失贡献相加与整批提示计算一致；保留现有 proxy leaf 和一次低分辨率 backward，不使用 `retain_graph=True`。

### 13.4 推理融合

对匹配和 presence logits 分别 sigmoid，每个查询的两项概率相乘，严格高于 `query_keep_threshold` 才保留。分数只用于筛选，不再乘入通过筛选的实例掩码。

每个提示先对保留实例的原始 soft mask 概率逐像素取最大值，再与语义 sigmoid 概率取最大值。没有实例通过时，实例聚合图为 0，结果自然退化为语义结果。掩码像素阈值仅用于生成评分训练目标，推理融合不把实例掩码二值化。不增加 NMS。

聚合在每个 chunk 内完成，不缓存全类别的 200 张高分辨率掩码。推理只拼接每提示的 288×288 `instance_score_map` 和语义 logits。Adapter 在融合后合并同类别提示，再执行原有相对过滤与 argmax。推理 `prompt_logits`/`final_logits` 对应融合概率，训练 `final_logits` 始终是纯语义 logits。TTA 继续先平均每个视图融合后的原始提示分数，再合并类别和相对过滤。

### 13.5 配置与运行

| 配置路径 | 默认值 | 含义 |
| --- | --- | --- |
| `model.instance_scoring_cfg.num_heads` | 8 | 两层新增注意力的头数 |
| `model.instance_scoring_cfg.dropout` | 0.0 | 新增注意力 dropout |
| `model.instance_scoring_cfg.query_keep_threshold` | 0.5 | 匹配概率乘 presence 后的查询保留阈值 |
| `model.criterion_cfg.instance_mask_threshold` | 0.5 | 生成评分监督的掩码像素阈值 |
| `model.criterion_cfg.instance_presence_bce_weight` | 0.1 | 新增 presence BCE 权重 |
| `model.criterion_cfg.instance_score_bce_weight` | 0.1 | 新增匹配分数软目标 BCE 权重 |

新增四项训练日志：`loss_instance_presence_bce`、`loss_instance_score_bce` 及对应 `_weighted` 贡献。两项权重设为 0 可关闭新增监督，但实例路径及推理筛选仍执行。

训练命令不变，使用新的输出目录：

```bash
python tools/train.py configs/train/isaid_loveda_full.py --work-dir work_dirs/instance_rescoring
```

本版本需要官方 SAM3 原始权重初始化恢复的查询 decoder、`dot_prod_scoring`、`instance_seg_head` 和 MLP MaskPredictor。加载 SAM3 预训练权重时，若缺少实例路径必要参数则直接报错，避免冻结随机实例模块。恢复 MLP MaskPredictor 后不再保留旧 semantic-only 的无用卷积 `mask_predictor.weight/bias`，不添加兼容层。

SAM3 所有原始模块强制冻结并保持 eval，包括恢复的 decoder 和评分头。Refiner、RemoteCLIP 原有微调规则不变；新增适配模块按基础学习率训练。decoder 的坐标缓存首次运行时在输入设备生成，避免构造阶段强制 CUDA；此调整不改变权重结构或坐标值。

旧训练 checkpoint 不能通过 `--resume-from` 继续本结构的完整训练；必须使用新的 work directory。checkpoint 容器格式仍为 schema 4。CPU 测试覆盖原始 decoder、原始掩码点积、prompt attention 次数、梯度隔离、忽略/空掩码、类别缺失、小目标 presence、别名融合和分块损失/梯度一致性；真实数据指标及 GPU 显存需要在实际训练环境验证。
