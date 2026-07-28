# OVRS-SAM3 设计说明

适用分支：`master`
项目仓库：`jk-jin/ovrs-sam3`
当前任务：开放词汇遥感语义分割

> 本文描述项目当前采用的模型与训练设计。代码实现发生结构性变化时，应同步更新本文。

## 1. 项目目标

OVRS-SAM3 接收一批遥感图像和当前数据集的类别名称，输出每个类别的像素级分割 logits。项目组合三类能力：

* SAM3 提供稳定的多尺度图像特征、文本提示编码、类条件 transformer encoder 和分割解码器。
* RemoteCLIP 提供面向遥感场景的局部图文对齐。
* Encoder refiner 在低分辨率融合两者，通过可训练72输入融合、共享冻结SAM3 Pixel Decoder和可训练288最终融合生成掩码。

当前只实现 semantic 模式，不支持实例分割、hybrid 模式或非空几何提示训练。

整体流程如下：

```text
图像与类别名称
  ├─ SAM3 图像 backbone → 288/144/72 多尺度 FPN
  ├─ SAM3 文本编码器与 transformer encoder layer 1..6
  │    → 每个图像-类别对的完整 6 层 encoder feature
  │    → SAM 文本 token 的 masked mean
  └─ RemoteCLIP
       ├─ 504×504 图像 → 36×36 dense image feature
       └─ 每类 32 个文本模板 → template text feature

完整 6 层 encoder feature
  → prompt cross-attention
  → 72×72 cross-attended encoder feature

RemoteCLIP 局部相似度图
  → 多尺度 score encoder
  → clip_score_embed_36 [B, C, 256, 36, 36]

72×72 cross-attended encoder feature
  → 双线性下采样
  → base_feature_36 [B, C, 256, 36, 36]
  → 直接作为初始 feature_36

SAM3 FPN72 [B, 256, 72, 72]
  → 双线性下采样到 36×36
  → 扩展类别维
  → 与 clip_score_embed_36 通道拼接
  → 1×1 Conv + 3×3 Conv 得到 FPN score 更新量
  → fpn_score_injection_scale 残差注入
  → 初始 score_embed_36 [B, C, 256, 36, 36]

feature_36 + score_embed_36
  → Refiner layer 1..4（全类别同时运行）
  → refiner_features_36 [B, C, 256, 36, 36]

随后按 prompt_chunk_size 逐类别块执行：

original_encoder_feature_72_chunk
  → 冻结的 SAM3 Pixel Decoder（torch.no_grad() 中）
  → original_pixel_feature_288 [B×C_chunk, 256, 288, 288]

original_pixel_feature_288
  → 冻结的 SAM3 semantic_seg_head
  → sam3_teacher_logits（训练时用于轻量辅助蒸馏）

refiner_feature_36_chunk
  → bilinear 插值到 72×72
  → refiner_feature_72 [B×C_chunk, 256, 72, 72]
  → 与 original_encoder_feature_72 通道拼接
  → 1×1 Conv + 3×3 Conv
  → fused_pixel_decoder_input_72 [B×C_chunk, 256, 72, 72]
  → 同一个冻结 SAM3 Pixel Decoder（梯度开启）
  → refined_pixel_feature_288 [B×C_chunk, 256, 288, 288]

refined_pixel_feature_288 + original_pixel_feature_288
  → 通道拼接 → 1×1 Conv + 3×3 Conv + GroupNorm + ReLU
  → fused_feature_288
  → 冻结 SAM3 semantic_seg_head
  → final_logits_chunk

所有 chunk 按原始类别顺序拼接
  → final_logits [B, C, 288, 288]
```

## 2. 张量约定

| 记号       | 含义                             |
| -------- | ------------------------------ |
| `B`      | batch 中的图像数                    |
| `C`      | 当前前向传播的类别数                     |
| `K`      | 每类 RemoteCLIP 文本模板数，固定为 32     |
| `D`      | SAM3 hidden dimension，固定为 256  |
| `D_clip` | RemoteCLIP 投影维度，ViT-L/14 为 768 |
| `L`      | Refiner 层数，当前固定为 4              |

当前固定输入下的主要张量为：

| 张量                            | 形状                                 | 说明                             |
| ----------------------------- | ---------------------------------- | ------------------------------ |
| `backbone_fpn`                | `[B, 256, 288/144/72, 288/144/72]` | SAM3 多尺度图像特征                   |
| `cross_attended_encoder_features_72` | `[B, C, 256, 72, 72]`       | 完整 6 层 encoder 与 prompt cross-attention 后的类条件特征 |
| `sam_fpn_72`                   | `[B, 256, 72, 72]`                 | SAM3 backbone 的图像级 FPN72，不带类别维度 |
| `sam_fpn_36`                   | `[B, 256, 36, 36]`                 | 双线性下采样后的 FPN，随后扩展类别维 |
| `fpn_score_update_36`           | `[B, C, 256, 36, 36]`              | score_embed 与 FPN36 拼接并经 1×1+3×3 Conv 得到的更新量 |
| `sam_text_mean`               | `[B, C, 256]`                      | SAM 文本 token 的 masked mean     |
| `remoteclip_feat_map`         | `[B, 768, 36, 36]`                 | RemoteCLIP dense image feature |
| `template_clip_text`          | `[C, 32, 768]`                     | 每类 32 个模板的文本特征                 |
| `clip_score_maps_36`          | `[B, C, 32, 36, 36]`               | 局部图文相似度图                       |
| `clip_score_embed_36`         | `[B, C, 256, 36, 36]`              | 未经FPN注入的纯 RemoteCLIP score embedding |
| `score_embed_36`              | `[B, C, 256, 36, 36]`              | 经FPN score注入及Refiner更新后的 score stream |
| `refiner_features_36`         | `[B, C, 256, 36, 36]`              | Refiner 的图像特征流                 |
| `refiner_feature_72`          | `[B×C_chunk, 256, 72, 72]`         | Refiner 36 经 bilinear 插值后的 72×72 特征 |
| `fused_pixel_decoder_input_72` | `[B×C_chunk, 256, 72, 72]`       | Refiner72 与原始 encoder72 融合后的 Pixel Decoder 输入 |
| `original_pixel_feature_288`  | `[B×C_chunk, 256, 288, 288]`       | 冻结 Pixel Decoder 原始分支输出的 288×288 特征 |
| `refined_pixel_feature_288`   | `[B×C_chunk, 256, 288, 288]`       | 冻结 Pixel Decoder Refiner 分支输出的 288×288 特征 |
| `fused_pixel_feature_288`     | `[B×C_chunk, 256, 288, 288]`       | 两路 288 特征拼接融合后的特征 |
| `final_logits`                | `[B, C, 288, 288]`                 | 可训练 mask decoder 输出的最终语义分割 logits |
| `sam3_teacher_logits`         | `[B, C, 288, 288]`（仅训练时）          | 冻结 SAM3 semantic head 输出的 detached teacher logits |

训练损失和评测会在必要时用最近邻插值把标签映射到 logits 尺度。

## 3. SAM3 分支

### 3.1 图像特征

SAM3 接收 1008×1008 的标准化图像。ViT patch size 为 14，主干 token grid 为 72×72。SimpleFPN 产生 288×288、144×144、72×72 和 36×36 四级特征；当前 `scalp=1` 丢弃最低分辨率的 36×36 级，因此主路径保留前三个尺度。

SAM3 图像 backbone 在训练中冻结并运行于 `eval()`。图像特征使用 `torch.no_grad()` 计算并 detach。

### 3.2 类条件 encoder

同一 batch 的所有样本必须共享完全相同的类别名称和顺序。类别按 `prompt_chunk_size` 分块，默认每块 8 类，以控制显存。

每个图像与每个类别组成一个 prompt pair。冻结的 SAM3 文本编码器和 6 层 transformer encoder 为每个 pair 生成类条件图像特征。所有 6 层在 `torch.no_grad()` 中一次运行完毕。

完整 encoder 输出后，执行一次 prompt cross-attention（同样在 `no_grad()` 中），得到 cross-attended full-encoder feature。SAM 文本向量通过有效 token 的 masked mean 得到，padding token 不参与平均。

所有类别块按原始顺序重新拼接。

### 3.3 共享冻结 Pixel Decoder

冻结的 Pixel Decoder 只返回最终 288×288 特征。原始分支和 Refiner 分支共享同一个 Pixel Decoder 实例和同一组 SAM3 权重。

```text
类条件 72×72 特征（替换 FPN 最后一层）
  → 上采样到 144×144 + FPN144 → 3×3 Conv + GroupNorm + ReLU
  → 上采样到 288×288 + FPN288 → 3×3 Conv + GroupNorm + ReLU
  → pixel_feature_288
```

Pixel Decoder 内部继续使用 `interpolation_mode="nearest"`（SAM3 原始设置）。只有 Refiner 36→72 的第一步使用 bilinear 插值。

原始分支在 `torch.no_grad()` 中运行；Refiner 分支在梯度开启状态下运行。Pixel Decoder 参数冻结且保持 `eval()`，但梯度可以穿过冻结的插值、加法、卷积、GroupNorm 和 ReLU 回传至 Refiner 输入。Refiner 分支可以使用 non-reentrant activation checkpoint（通过 `_embed_pixels()` 中的 `self.act_ckpt and torch.is_grad_enabled()` 条件控制）。

原始 semantic head 始终冻结并只用于 teacher logits。

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

最终输出 `[B, 768, 36, 36]`。配置指定的中间层特征只作为 debug 数据保留，不进入当前主路径。

### 4.2 模板文本编码

每个类别使用 32 个固定遥感文本模板，生成 `[C, 32, 768]` 的模板特征。文本编码支持 micro-batch 和 non-reentrant activation checkpoint。

缓存规则必须服从参数是否可训练：

* RemoteCLIP 文本分支完全冻结时，可以缓存 detach 后的模板特征。
* 文本分支可训练且全局梯度开启时，每个训练 step 重新编码并保留计算图。
* 验证位于 `torch.no_grad()` 中，可以在一次验证过程中复用当前权重对应的缓存。

不能用模块的 `training` 属性判断是否需要梯度，因为 RemoteCLIP 在部分微调时仍保持 `eval()`。

### 4.3 Score embedding

模板文本特征和 dense 图像特征分别做 L2 归一化，再计算余弦相似度并乘固定系数 20，得到 32 通道模板分数图。

32 通道分数图经过：

```text
1×1 stem：32 → 256
  → 三个并行 depthwise 3×3 分支，dilation 为 1/2/3
  → 每个分支用 pointwise 1×1 投影到 128 通道
  → 拼接为 384 通道
  → 1×1 融合到 256 通道
  → 与 stem 残差相加
  → GroupNorm + GELU
```

输出 `clip_score_embed_36`，作为后续 FPN score 注入前的纯 RemoteCLIP score embedding。Refiner 实际使用的初始 score stream 是经过 FPN36 残差注入后的结果。

## 5. Class-conditioned encoder refiner

Refiner 在 36×36 上同时维护图像 feature 流和 score embedding 流。默认使用 4 层、8 个 attention heads、12×12 窗口和 6 像素 shift。

### 5.1 Feature stream 初始化与 FPN score 注入

Cross-attended full-encoder feature（72×72）双线性下采样到 36×36，得到 `base_feature_36`。Feature stream 直接使用 `base_feature_36`，不接收 FPN 注入：

```python
feature_36 = base_feature_36
```

SAM3 backbone 的图像级 `sam_fpn_72 [B, 256, 72, 72]` 双线性下采样到 36×36，通过 `expand` 扩展类别维为 `[B, C, 256, 36, 36]`。

FPN36 与 `clip_score_embed_36` 在第 3 维（通道维）拼接为 `[B, C, 512, 36, 36]`，经 `1×1 Conv + 3×3 Conv`（Xavier 初始化，bias=0）得到 FPN score 更新量 `fpn_score_update_36`，并通过独立可学习标量 `fpn_score_injection_scale` 做残差注入：

```python
score_embed_36 = clip_score_embed_36 + fpn_score_injection_scale * fpn_score_update_36
```

注入只发生一次，位于所有 Refiner Attention 之前。两个卷积后没有 norm 和 activation，允许更新量可正可负，且不强制归一化影响后续注意力计算。FPN 只注入 score stream，不进入 feature stream。

`fpn_score_injection_scale` 初值来自现有 `residual_scale_init`（默认 0.1），由 `make_residual_scale()` 创建并自动关闭 weight decay。

### 5.2 单层 refiner

每层采用 pre-norm，并依次执行：

1. **ClassScoreAttention**：在每个空间位置跨类别做注意力。Q/K 由图像 feature、SAM 文本均值和 score embedding 拼接后投影；feature 与 score 使用独立 value/output 分支。
2. **Regular WindowScoreAttention**：每个类别内部执行非移位窗口注意力。
3. **Shifted WindowScoreAttention**：使用 shift mask 和相对位置偏置连接相邻窗口。
4. **Feature FFN**：逐 token 更新图像流。
5. **Score FFN**：逐 token 更新分数流。

每层共 8 个 LayerScale 标量，统一初始化为 0.1。

全部 refiner 层结束后，不再对 score embedding 执行最终 LayerNorm。

### 5.3 Pixel Decoder 输入融合与最终掩码头

Refiner 在所有类别上统一执行后，36×36 特征经 bilinear 插值到 72×72。Refiner72 与同一 chunk 的原始 encoder72 通过 `PixelDecoderInputFusion72` 融合后，送入共享的冻结 SAM3 Pixel Decoder 上采样到 288×288。

72 输入融合模块 `PixelDecoderInputFusion72` 位于 `models/refined_mask_decoder.py`，执行：

```text
refiner_feature_72 [N, 256, 72, 72]
  + original_feature_72 [N, 256, 72, 72]
  → 通道拼接 → [N, 512, 72, 72]
  → Conv2d(512→256, k=1)
  → Conv2d(256→256, k=3, p=1)
  → fused_pixel_decoder_input_72
```

融合模块不使用 norm 和 activation，无残差系数，不读取 FPN。只有融合结果进入 Refiner 分支 Pixel Decoder。训练时使用 non-reentrant activation checkpoint。属于 `core.encoder_refiner`。

两条 Pixel Decoder 分支的输出在 288×288 尺度通过 `FinalFeatureFusion288` 融合。该模块执行：

```text
refined_pixel_feature_288 [N, 256, 288, 288]
  + original_pixel_feature_288 [N, 256, 288, 288]
  → 通道拼接 → [N, 512, 288, 288]
  → Conv2d(512→256, k=1)
  → Conv2d(256→256, k=3, p=1)
  → GroupNorm(8, 256)
  → ReLU
  → fused_feature_288 [N, 256, 288, 288]
```

最终掩码 logits 由冻结的 SAM3 `semantic_seg_head`（Conv2d 256→1, k=1）在外部对 `fused_feature_288` 产生。该卷积权重冻结，但调用时不在 `no_grad()` 中，梯度可穿过它回传至融合模块和 Refiner。

`RefinerMaskDecoder` 只负责两张 288×288 特征的融合。不再执行任何插值、上采样或掩码预测。训练时只对 `final_fusion_288` 使用 non-reentrant activation checkpoint。

### 5.4 残差系数日志

训练日志、JSONL 和 W&B 记录 Refiner 相关的残差系数：

| 类别 | 前缀 | 参数 |
| --- | --- | --- |
| Refiner 内部 | `residual/refiner_internal/` | 每层 8 个 LayerScale |
| FPN score 注入 | `residual/fpn_score_injection/` | 单个 `fpn_score_injection_scale` |

每类记录 `count`、`mean`、`abs_mean`、`min`、`max` 和
`negative_ratio`。`residual/refiner_internal/count` 为 32（4 层 × 8 个标量），`residual/fpn_score_injection/count` 为 1。

## 6. 冻结 SAM3 分割头与梯度边界

Prompt cross-attention 在完整 6 层 encoder 之后、Refiner 之前执行一次（通过 `apply_prompt_cross_attention()`），位于 `torch.no_grad()` 中。

Pixel Decoder 参数始终冻结（`requires_grad=False`）并保持 `eval()`。两条分支共享同一个 Pixel Decoder 实例：

- **原始分支**：在 `torch.no_grad()` 中执行，`original_pixel_feature_288.requires_grad` 为 False，teacher logits 不保留计算图。
- **Refiner 分支**：在梯度开启状态下执行。Pixel Decoder 参数虽冻结，但梯度可以穿过其插值、加法、卷积、GroupNorm 和 ReLU 回传至72融合模块和 `refiner_features_36`。semantic head 只在原始 teacher 分支中调用并始终位于 `no_grad` 中。

原始 semantic head 始终冻结并只用于 teacher logits。

语义主路径只消费冻结分割头的 `pixel_feature_288` 输出和 teacher logits。`pred_masks` 和 `presence_head` 等子模块定义保留但不调用。

## 7. 训练设计

### 7.1 冻结与微调

以下 SAM3 模块冻结并保持 `eval()`：

* backbone；
* transformer encoder（完整 6 层，在 `no_grad()` 中执行）；
* geometry encoder；
* segmentation head（Pixel Decoder 参数冻结并保持 `eval()`。原始分支在 `no_grad()` 中执行，Refiner 分支在梯度开启状态下执行。semantic head 只在原始 teacher 分支中调用并位于 `no_grad()` 中）。

完整 SAM3 encoder 和前置 prompt cross-attention 不保留计算图，均在 `torch.no_grad()` 中执行。

`core.encoder_refiner` 完整训练。其内部的 Refiner 层、`PixelDecoderInputFusion72` 和 `FinalFeatureFusion288` 同属一个参数组，由现有 `trainable_modules=["core.encoder_refiner"]` 自动覆盖，使用基础学习率 `1e-4`。最终掩码 logits 由冻结的 SAM3 `semantic_seg_head` 产生。

RemoteCLIP 图像和文本分支默认使用 `attention` 微调模式，仅训练注意力 Q/V 与位置嵌入，同时保持 `eval()` 以关闭 dropout 和 patch dropout。

OpenCLIP 常把 Q/K/V 存在同一个融合参数中。项目对该参数注册梯度 mask，使 K 区域梯度为 0；同时把整个融合参数组的 weight decay 强制设为 0。恢复 optimizer 状态后会重新应用这一不变量。

默认 AdamW 基础学习率为 `1e-4`：

* encoder refiner 使用 1.0 倍学习率；
* RemoteCLIP text/image 使用 0.01 倍学习率，即 `1e-6`；
* normalization 参数不使用 weight decay；
* 所有残差系数（Refiner 内部 32 个 LayerScale + FPN score 注入 1 个 `fpn_score_injection_scale`，共 33 个可学习标量）使用 `_ovrs_disable_weight_decay` 标记，weight decay 强制为 0；全部由同一个 `residual_scale_init` 配置初始化；
* 梯度裁剪上限为 0.1；
* warmup 保持前 1000 步，线性从 0.1 倍到全额学习率，后续余弦衰减。

### 7.2 损失

每个类别通道独立使用 binary mask 监督，不使用跨类别 softmax。

**GT 主损失**（监督 `final_logits`）：

* 存在于图像中的类别：有效像素参与 BCE，ignore 像素作为低权重负样本抑制泄漏。
* 不存在于图像中的类别：只在有效像素上计算 BCE，并使用较小的 pair 权重。
* Dice 只对存在类别计算；当前默认权重为 0。
* 全部像素均为 ignore 时跳过 backward 和 optimizer step。

**SAM3 teacher 掩码蒸馏**（监督 `final_logits`，仅训练时）：

1. 冻结的 SAM3 semantic head 产生的 teacher logits 做 sigmoid，得到 soft probability 目标。
2. student 使用 raw `final_logits`。
3. 用 `binary_cross_entropy_with_logits` 逐像素计算蒸馏损失。
4. 只监督 GT 中存在的图像—类别对的有效像素。
5. teacher 和 student 都在 288×288 分辨率，不做尺度变换。
6. teacher 必须 detach。
7. 蒸馏权重 `sam3_mask_distill_weight=0.05`。

总损失：

```python
total_loss = final_bce_weight * loss_final_bce
           + final_dice_weight * loss_final_dice
           + sam3_mask_distill_weight * loss_sam3_mask_distill_bce
```

## 8. 推理与评测

推理先对 `final_logits` 做 sigmoid，得到未经筛选的 `raw_final_score_map`。可选的逐类别相对阈值在每个图像、每个类别内部按空间最小值和最大值归一化，只把未通过位置置 0；保留位置继续使用原始 sigmoid 分数。最终对类别维取 argmax。

标签空间中两个机制职责不同：

* `reduce_zero_label` 用于从数据集标签空间彻底删除原始 0 类，并把其余类别只重映射一次；类别名称、前向通道和评测元数据必须使用同一重映射结果。
* `background_cfg` 用于有实际背景语义的数据集。背景可以不进入模型前向，但仍属于评测类别空间，并由统一后处理映射回来。

两条路径不能对同一标签连续执行两次 0 类删除或索引平移。

Evaluator 输出整体 mIoU、mAcc、pixel accuracy 和逐类别指标。`metric_groups` 可以按 `class_ids` 或 `class_names` 定义命名类别组，并分别计算组内 mIoU/mAcc。完整 iSAID→LoveDA 配置使用前景类别组作为 checkpoint monitor。

TTA 当前只支持 `scale=1.0` 和空间翻转。多个视图必须先反变换并平均 `raw_final_score_map`，再统一执行一次非线性相对阈值过滤。

## 9. Checkpoint、恢复与实验追踪

训练只在显式提供 `--resume-from` 时恢复完整状态，不自动扫描 `work_dir`。未提供该参数时不会加载任何已有训练产物；若目标目录已经包含训练 checkpoint，则直接报错，避免混合实验。

完整训练 checkpoint 自包含：

* `global_iter`；
* model、optimizer、AMP scaler 和 scheduler；
* Python、NumPy、Torch CPU 与各 CUDA device 的 RNG 状态；
* 可恢复随机 batch sampler 的排列、增强种子、游标和 generator 状态；
* hook 状态，包括 W&B run identity 和 `last_history_step`；
* checkpoint manager 的 best score；
* train/validation 统计及 validation 状态。

NumPy RNG 数组以 Tensor 保存，因此统一加载入口可以安全使用 `torch.load(..., weights_only=True)`。写入 iteration checkpoint、`latest.pth` 和 `best.pth` 时使用临时文件与原子替换。

`latest.pth` 只在保存或完成一次 checkpoint finalization 时更新，不随普通日志输出更新。`best.pth` 只在 monitor 指标严格改善时更新。

恢复顺序为：严格加载模型与训练状态、恢复 sampler/hook、构建 DataLoader iterator、初始化或恢复 W&B、准备缓存，最后恢复 RNG。若 checkpoint 标记验证尚未完成，恢复后先重放该次验证，再继续训练。

W&B run ID、project、entity、run name 和 `last_history_step` 来自 checkpoint hook 状态，不依赖 `work_dir/wandb_run.json`。恢复时先查询远端 `lastHistoryStep`：远端尾部超过 checkpoint 时使用 `resume_from` 截断旧历史并继续同一个 run；远端未超过 checkpoint 时使用 `resume="must"` 正常接续。`last_history_step` 是 W&B 内部 `_step` 序号，不等于 `trainer/global_iter`。所有 `run.log()` 使用显式严格递增的 `_step`，summary 在 rewind 后由 W&B 重新计算。JSONL 在恢复时追加，在全新训练时重建。

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

W&B rewind 示例：

```text
远端 run：trainer/global_iter 已到 9900
恢复 checkpoint：global_iter=8000
checkpoint 内 last_history_step=K
远端 lastHistoryStep > K
→ resume_from="<run_id>?_step=K"
→ 删除 K 之后的旧历史
→ 从 checkpoint 状态重新训练并上传
```

其中 `K` 是 checkpoint 保存时最后一条 W&B history 的内部序号，不是训练步数 8000。

旧格式或缺少完整运行状态的权重不能用于 `--resume-from`，但可以通过 `--load-model-from` 只加载模型参数。

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
| `models/sam3_image.py`                | 类别 chunk、缓存、SAM3 encoder、refiner 调用、逐 chunk 解码 |
| `models/encoder_refiner.py`           | 全类别 Refiner、FPN score 注入及两处高分辨率融合的公开接口 |
| `models/refined_mask_decoder.py`      | PixelDecoderInputFusion72、FinalFeatureFusion288 |
| `models/encoder_refiner_attention.py` | 跨类别/窗口注意力、双流 FFN 与 LayerScale            |
| `models/maskformer_segmentation.py`   | prompt attention、共享冻结 Pixel Decoder 和原始 semantic head |
| `models/score_embeddings.py`          | 32 模板相似度图和多尺度 score encoder              |
| `models/openclip_image_encoder.py`    | 36×36 dense RemoteCLIP 图像特征              |
| `models/openclip_text_encoder.py`     | 模板文本编码、micro-batch 与梯度控制                 |
| `losses/semantic_criterion.py`        | GT 主损失（present/absent 加权 BCE + Dice）和 SAM3 teacher 蒸馏 BCE |
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
4. 模板数固定为 32，RemoteCLIP 图文投影维度一致。
5. 可训练 RemoteCLIP 文本特征不能跨 optimizer step 缓存。
6. 验证不得重新开启 RemoteCLIP 图像分支的 autograd。
7. Refiner 必须先在全部类别上执行，再按 chunk 做72融合和288解码。Refiner 不能放进 chunk 循环。
8. Refiner 36 特征只做一次 bilinear 36→72 插值。
9. Refiner72 必须先与相同 chunk 的原始 encoder72 融合（1×1+3×3 Conv，无 norm 和 activation，无残差系数）。只有融合结果进入 Refiner 分支 Pixel Decoder。
10. base_feature_36 不接收直接 FPN 残差。FPN36 只注入 score stream。
11. FPN score 注入必须位于所有 Refiner 层之前且只执行一次。使用 1×1+3×3 Conv，无 norm 和 activation。
12. clip_score_embed_36 保持纯 RemoteCLIP 输出，用于 debug。
13. 原始分支和 Refiner 分支必须共享同一个 Pixel Decoder 实例。不复制、不新建、不解冻 Pixel Decoder。
14. 原始 Pixel Decoder 分支必须在 `no_grad()` 中。Refiner Pixel Decoder 分支必须保持输入梯度。
15. Pixel Decoder 内部继续使用 `interpolation_mode="nearest"`。Pixel Decoder 上采样使用 FPN144 和 FPN288。不在 Pixel Decoder 入口再次叠加 FPN72。
16. 最终只在 288×288 尺度拼接原始与 Refiner 特征。最终融合模块不执行插值。
17. teacher 只来自原始分支并且必须 detach。teacher 和 student 都为 288×288，不做尺度变换。
18. 最终掩码 logits 由冻结的 SAM3 `semantic_seg_head` 产生，不创建独立的 mask head。
19. Refiner 内部残差系数统一由 `residual_scale_init` 控制，默认值为 0.1；这些标量不使用 weight decay。
20. TTA 必须先平均原始分数，再进行相对阈值过滤。
21. `reduce_zero_label` 与 `background_cfg` 各自只执行其定义的一次标签空间变换。
22. 完整恢复必须严格校验 checkpoint schema；模型权重迁移必须走独立入口。
23. 完整 6 层 encoder → prompt cross-attention → Refiner（全类别）→ 36→72 bilinear → 72 融合 → 逐 chunk 共享 Pixel Decoder → 逐 chunk 288 融合 → 冻结 semantic_seg_head → 拼接。
24. 最终融合模块属于 `core.encoder_refiner` 并使用基础学习率 `1e-4`。冻结 `semantic_seg_head` 属于 `segmentation_head`。
25. FPN 注入必须在全类别 Refiner 中执行，不能移入高分辨率 chunk 循环。
26. 72/288 高分辨率特征不跨 chunk 累积。

当前限制：

* 只支持 semantic 模式；
* 不支持非空几何 prompt；
* 不支持动态 SAM3/refiner 空间尺寸或多尺度 TTA；
* `clip_mid_features` 只用于 debug；
* 当前默认训练集为 iSAID，验证集为 LoveDA。

## 12. Checkpoint 非兼容变更

Checkpoint schema 版本为 4。新增 `runtime_state.hooks.WandbHook.last_history_step` 持久化字段。旧 version 3 checkpoint 不支持完整恢复。

本次模型参数结构发生了非兼容变化：

* 删除 `fpn_injection_proj_36.*`、`fpn_injection_scale`（旧 feature-stream FPN 注入）。
* 新增 `fpn_score_fusion_36.*`、`fpn_score_injection_scale`（新 score-stream FPN 注入）。
* 新增 `mask_decoder.pixel_decoder_input_fusion_72.*`（72 输入融合模块）。
* 删除 `mask_decoder.mask_head.*`（最终掩码现由冻结 SAM3 `semantic_seg_head` 产生）。
* `mask_decoder.final_fusion_288.*` 名称和形状保持不变。
* Pixel Decoder 权重结构不变，继续加载原始 SAM3 权重。

旧 checkpoint 不能通过 `--resume-from` 严格恢复。如使用 `--load-model-from` 迁移旧模型参数：

* 旧 `fpn_injection_proj_36.*` 和 `fpn_injection_scale` 应成为 unexpected keys。
* 新 score 融合卷积、score 残差系数和 72 融合卷积应成为 missing keys，使用自身 Xavier 初始化。
* 新残差系数使用当前 `residual_scale_init`。
* final fusion 名称不变时继续加载已有参数。
* 旧 `mask_head` 参数成为 unexpected keys。
* 不创建兼容层或旧参数占位符。
