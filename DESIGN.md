# OVRS-SAM3 设计说明

适用分支：`master`
项目仓库：`jk-jin/ovrs-sam3`
当前任务：开放词汇遥感语义分割

> 本文描述项目当前采用的模型与训练设计。代码实现发生结构性变化时，应同步更新本文。

## 1. 项目目标

OVRS-SAM3 接收一批遥感图像和当前数据集的类别名称，输出每个类别的像素级分割 logits。项目组合三类能力：

* SAM3 提供稳定的多尺度图像特征、文本提示编码、类条件 transformer encoder 和分割解码器。
* RemoteCLIP 提供面向遥感场景的局部图文对齐。
* Encoder refiner 在低分辨率上融合两者，并把更新后的类条件特征交回 SAM3 分割头。

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
  │
  ├─ 第一次 Pixel Decoder + Semantic Head（torch.no_grad）
  │    → original_logits [B, C, 288, 288]
  │    → detach + clamp[-32, 32]
  │    → MaskPromptEncoder36
  │    → mask_prompt_embed_36 [B, C, 256, 36, 36]
  │
  └─ 双线性下采样 → base_feature_36

base_feature_36 + mask_prompt_embed_36 → 初始 feature_36

RemoteCLIP 局部相似度图
  → 多尺度 score encoder
  → 36×36 score embedding

feature_36 + score_embed_36
  → Refiner layer 1 → shared 1×1 Conv → aux logit 1
  → Refiner layer 2 → shared 1×1 Conv → aux logit 2
  → Refiner layer 3 → shared 1×1 Conv → aux logit 3
  → Refiner layer 4 → shared 1×1 Conv → aux logit 4
  → refiner_aux_logits_36 [4, B, C, 36, 36]

最终 refiner feature 双线性插值到 72×72
  → 与 cross-attended full-encoder feature 拼接
  → 1×1 Conv + 3×3 Conv 输出融合
  → refined_encoder_features_72
  → 第二次 Pixel Decoder + Semantic Head（保留梯度）
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
| `original_logits`             | `[B, C, 288, 288]`                 | 第一次分割头的 detached raw logits，用作 mask prompt 输入 |
| `mask_prompt_embed_36`        | `[B, C, 256, 36, 36]`              | MaskPromptEncoder36 输出的 dense mask embedding |
| `sam_text_mean`               | `[B, C, 256]`                      | SAM 文本 token 的 masked mean     |
| `remoteclip_feat_map`         | `[B, 768, 36, 36]`                 | RemoteCLIP dense image feature |
| `template_clip_text`          | `[C, 32, 768]`                     | 每类 32 个模板的文本特征                 |
| `clip_score_maps_36`          | `[B, C, 32, 36, 36]`               | 局部图文相似度图                       |
| `score_embed_36`              | `[B, C, 256, 36, 36]`              | refiner 的语义分数流                 |
| `refiner_features_36`         | `[B, C, 256, 36, 36]`              | refiner 的图像特征流                 |
| `refiner_aux_logits_36`       | `[4, B, C, 36, 36]`                | 四层 Refiner 共享 1×1 Conv 的辅助 logits |
| `refined_encoder_features_72` | `[B, C, 256, 72, 72]`              | Refiner 输出的更新特征               |
| `final_logits`                | `[B, C, 288, 288]`                 | 最终语义分割 logits                  |

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

### 3.3 两遍分割头

训练时共用同一套 Pixel Decoder 和 Semantic Head，前向实际执行两次：

1. **第一次**（`torch.no_grad()`）：输入 cross-attended full-encoder feature，输出
   `original_logits [B, C, 288, 288]`。该原始 logits detach 后作为
   `MaskPromptEncoder36` 的输入，不参与反向传播。
2. **第二次**（保留梯度）：输入 Refiner 更新后的 `refined_encoder_features_72`，
   输出 `final_logits [B, C, 288, 288]`，由 GT 主损失监督。

推理同样需要第一遍来构造 mask prompt，但只向外部暴露 `final_logits`。

Pixel Decoder 内部继续融合 144×144 和 288×288 的 SAM3 FPN。

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

输出 `clip_score_embed_36`，并直接作为 refiner 的初始 score 流。

## 5. Class-conditioned encoder refiner

Refiner 在 36×36 上同时维护图像 feature 流和 score embedding 流。默认使用 4 层、8 个 attention heads、12×12 窗口和 6 像素 shift。

### 5.1 Mask prompt 注入

Cross-attended full-encoder feature（72×72）双线性下采样到 36×36，作为
`base_feature_36`。

第一次分割头输出的 `original_logits [B, C, 288, 288]` 经过 detach 和
`clamp(-32, 32)` 后，由 `MaskPromptEncoder36` 编码为
`mask_prompt_embed_36 [B, C, 256, 36, 36]`。

`MaskPromptEncoder36` 固定结构：

```text
Conv2d(1→4, k=2, s=2) + LayerNorm2d + GELU   → 144×144
Conv2d(4→16, k=2, s=2) + LayerNorm2d + GELU  →  72×72
Conv2d(16→64, k=2, s=2) + LayerNorm2d + GELU →  36×36
Conv2d(64→256, k=1)                           →  36×36×256
```

最后一层 `Conv2d(64→256, k=1)` 权重和 bias 初始化为 0，因此旧 checkpoint
加载后 mask prompt 初始为 0，不污染已有 encoder feature。

Refiner 初始 feature 流固定为：

```python
feature_36 = base_feature_36 + mask_prompt_embed_36
```

这里没有额外的残差系数。Refiner 入口不再接收 SAM3 FPN。

### 5.2 单层 refiner

每层采用 pre-norm，并依次执行：

1. **ClassScoreAttention**：在每个空间位置跨类别做注意力。Q/K 由图像 feature、SAM 文本均值和 score embedding 拼接后投影；feature 与 score 使用独立 value/output 分支。
2. **Regular WindowScoreAttention**：每个类别内部执行非移位窗口注意力。
3. **Shifted WindowScoreAttention**：使用 shift mask 和相对位置偏置连接相邻窗口。
4. **Feature FFN**：逐 token 更新图像流。
5. **Score FFN**：逐 token 更新分数流。

每层共 8 个 LayerScale 标量，统一初始化为 0.1。

每层结束后，对 `feature_36` 使用共享的 `layer_mask_head`（`Conv2d(256→1, k=1)`）产生该层的辅助 logit `[B, C, 36, 36]`。四层 logit 堆叠为
`refiner_aux_logits_36 [4, B, C, 36, 36]`。

层内不使用 post-norm。全部 refiner 层结束后，只对 score embedding 执行一次最终 LayerNorm；feature 流不做最终 LayerNorm。

### 5.3 36→72 输出融合

完整 `refiner_features_36` 双线性插值到 72×72，并与原始 cross-attended full-encoder
`encoder_features_72` 在通道维拼接为 `[B*C, 512, 72, 72]`。

拼接特征先经过 1×1 Conv 压缩到 256 通道，再经过 padding=1 的 3×3 Conv 保持尺寸。输出直接作为 `refined_encoder_features_72`。输出融合训练时使用 non-reentrant activation checkpoint。

### 5.4 残差系数日志

训练日志、JSONL 和 W&B 记录 Refiner 内部的残差系数：

| 类别 | 前缀 | 参数 |
| --- | --- | --- |
| Refiner 内部 | `residual/refiner_internal/` | 每层 8 个 LayerScale |

每类记录 `count`、`mean`、`abs_mean`、`min`、`max` 和
`negative_ratio`。`residual/refiner_internal/count` 为 32（4 层 × 8 个标量）。

不再记录 `residual/fpn/*`。

## 6. SAM3 分割头

Prompt cross-attention 在完整 6 层 encoder 之后、Refiner 之前执行一次（通过 `apply_prompt_cross_attention()`），位于 `torch.no_grad()` 中。

Pixel decoder 用 72×72 类条件特征替换 FPN 最后一层，再依次与 144×144、288×288 的 SAM3 FPN 融合。Pixel Decoder 内部的 144/288 FPN 融合仍然保留。

SAM3 segmentation head 的参数保持冻结，但前向不能放入 `no_grad()`（第二次调用时），因为损失梯度仍需穿过该头回到 refiner。

语义主路径只消费 `semantic_seg`。`pred_masks` 和 `presence_head` 等子模块定义保留但不调用。

## 7. 训练设计

### 7.1 冻结与微调

以下 SAM3 模块冻结并保持 `eval()`：

* backbone；
* transformer encoder（完整 6 层，在 `no_grad()` 中执行）；
* geometry encoder；
* segmentation head（参数冻结，但第二次前向允许梯度穿过）。

完整 SAM3 encoder 和前置 prompt cross-attention 不保留计算图，均在 `torch.no_grad()` 中执行。

`core.encoder_refiner` 完整训练。其内部的 `MaskPromptEncoder36`、`layer_mask_head` 和 Refiner 层同属一个参数组，由现有 `trainable_modules=["core.encoder_refiner"]` 自动覆盖，使用基础学习率 `1e-4`。

RemoteCLIP 图像和文本分支默认使用 `attention` 微调模式，仅训练注意力 Q/V 与位置嵌入，同时保持 `eval()` 以关闭 dropout 和 patch dropout。

OpenCLIP 常把 Q/K/V 存在同一个融合参数中。项目对该参数注册梯度 mask，使 K 区域梯度为 0；同时把整个融合参数组的 weight decay 强制设为 0。恢复 optimizer 状态后会重新应用这一不变量。

默认 AdamW 基础学习率为 `1e-4`：

* encoder refiner 使用 1.0 倍学习率；
* RemoteCLIP text/image 使用 0.01 倍学习率，即 `1e-6`；
* normalization 参数不使用 weight decay；
* 所有残差系数（Refiner 内部 32 个 LayerScale）使用 `_ovrs_disable_weight_decay` 标记，weight decay 强制为 0；
* 梯度裁剪上限为 0.1；
* warmup 保持前 1000 步，线性从 0.1 倍到全额学习率，后续余弦衰减。

### 7.2 损失

每个类别通道独立使用 binary mask 监督，不使用跨类别 softmax。

**GT 主损失**（监督 `final_logits`）：

* 存在于图像中的类别：有效像素参与 BCE，ignore 像素作为低权重负样本抑制泄漏。
* 不存在于图像中的类别：只在有效像素上计算 BCE，并使用较小的 pair 权重。
* Dice 只对存在类别计算；当前默认权重为 0。
* 全部像素均为 ignore 时跳过 backward 和 optimizer step。

**辅助蒸馏损失**（监督 `refiner_aux_logits_36`）：

1. detached `original_logits` 做 sigmoid，得到原始分割概率。
2. 使用 `mode="area"` 将概率从 288×288 降采样到 36×36。
3. 扩展为 `[4, B, C, 36, 36]` 与四层辅助 logits 一一对应。
4. 用 `binary_cross_entropy_with_logits` 在层、batch、类别和空间位置统一取 mean。
5. 不使用 GT valid mask、ignore mask、presence 权重或 absent-class 权重。
6. 辅助损失固定权重 `refiner_aux_distill_weight=0.05`。

总损失：

```python
total_loss = final_bce_weight * loss_final_bce
           + final_dice_weight * loss_final_dice
           + refiner_aux_distill_weight * loss_refiner_aux_distill_bce
```

先对四层辅助 BCE 统一取 mean，再乘 `0.05`。

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
* hook 状态，包括 W&B run identity；
* checkpoint manager 的 best score；
* train/validation 统计及 validation 状态。

NumPy RNG 数组以 Tensor 保存，因此统一加载入口可以安全使用 `torch.load(..., weights_only=True)`。写入 iteration checkpoint、`latest.pth` 和 `best.pth` 时使用临时文件与原子替换。

`latest.pth` 只在保存或完成一次 checkpoint finalization 时更新，不随普通日志输出更新。`best.pth` 只在 monitor 指标严格改善时更新。

恢复顺序为：严格加载模型与训练状态、恢复 sampler/hook、构建 DataLoader iterator、初始化或恢复 W&B、准备缓存，最后恢复 RNG。若 checkpoint 标记验证尚未完成，恢复后先重放该次验证，再继续训练。

W&B run ID、project、entity 和 run name 来自 checkpoint hook 状态，不依赖 `work_dir/wandb_run.json`。正常恢复使用同一个 run；全新训练生成新 run。JSONL 在恢复时追加，在全新训练时重建。

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

旧格式或缺少完整运行状态的权重不能用于 `--resume-from`，但可以通过 `--load-model-from` 只加载模型参数。

本次模型参数结构发生了非兼容变化（删除 `feature_fpn_fusion.*` 和 `feature_fpn_res_scale`，新增 `mask_prompt_encoder.*` 和 `layer_mask_head.*`）：

* 不得使用旧 checkpoint 做 `--resume-from` 精确续训；
* 如需复用旧模型权重，只能使用 `--load-model-from` 并创建新的 `work_dir`；
* 非严格加载时旧 FPN 参数显示为 unexpected keys，新模块显示为 missing keys。这是预期行为。

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
| `models/sam3_image.py`                | 类别 chunk、缓存、SAM3 encoder、两遍分割头、refiner 调用 |
| `models/encoder_refiner.py`           | MaskPromptEncoder36、mask prompt 注入、refiner 主体、逐层辅助预测和 72×72 输出融合 |
| `models/encoder_refiner_attention.py` | 跨类别/窗口注意力、双流 FFN 与 LayerScale            |
| `models/maskformer_segmentation.py`   | prompt attention、pixel decoder 与语义 head  |
| `models/score_embeddings.py`          | 32 模板相似度图和多尺度 score encoder              |
| `models/openclip_image_encoder.py`    | 36×36 dense RemoteCLIP 图像特征              |
| `models/openclip_text_encoder.py`     | 模板文本编码、micro-batch 与梯度控制                 |
| `losses/semantic_criterion.py`        | GT主损失（present/absent 加权 BCE + Dice）和辅助蒸馏 BCE |
| `engine/checkpoint.py`                | 安全、原子、严格的 checkpoint 保存与加载               |
| `engine/runtime_state.py`             | RNG 捕获与恢复                                |
| `data/resumable_sampler.py`           | 可精确恢复的数据顺序与增强种子                          |
| `engine/experiment_hooks.py`          | JSONL 与 W&B 生命周期                         |
| `engine/evaluator.py`                 | 语义指标、命名指标组、背景映射与 TTA                     |

## 11. 实现不变量与限制

修改代码时必须保持：

1. 类别 chunk 完整、无重复且按原顺序拼接。
2. SAM3 encoder、refiner 和 RemoteCLIP grid 分别固定为 72×72、36×36 和 36×36。
3. SAM3 hidden dimension 与 score embedding dimension 均为 256。
4. 模板数固定为 32，RemoteCLIP 图文投影维度一致。
5. 可训练 RemoteCLIP 文本特征不能跨 optimizer step 缓存。
6. 验证不得重新开启 RemoteCLIP 图像分支的 autograd。
7. 冻结的 segmentation head 第二次调用时必须允许梯度穿过。
8. Refiner 内部残差系数统一由 `residual_scale_init` 控制，默认值为 0.1；这些标量不使用 weight decay。72×72 输出融合和 mask prompt 注入没有残差系数。
9. TTA 必须先平均原始分数，再进行相对阈值过滤。
10. `reduce_zero_label` 与 `background_cfg` 各自只执行其定义的一次标签空间变换。
11. 完整恢复必须严格校验 checkpoint schema；模型权重迁移必须走独立入口。
12. 完整 6 层 encoder → 一次 prompt cross-attention → 第一遍分割头（no_grad）→ MaskPromptEncoder → Refiner + 逐层辅助 logits → 第二遍分割头（保留梯度）。
13. Pixel Decoder 内部 144/288 FPN 融合仍然保留。
14. 四层 Refiner 共用一个 `layer_mask_head`（`Conv2d(256→1, k=1)`）。

当前限制：

* 只支持 semantic 模式；
* 不支持非空几何 prompt；
* 不支持动态 SAM3/refiner 空间尺寸或多尺度 TTA；
* `clip_mid_features` 只用于 debug；
* 当前默认训练集为 iSAID，验证集为 LoveDA。
