# OVRS-SAM3 设计说明

适用分支：当前分支（由 `master` 重构）
项目仓库：`jk-jin/ovrs-sam3`
当前任务：开放词汇遥感语义分割

> 本文描述当前实现；代码发生结构性变化时必须同步更新。

## 1. 项目目标与整体流程

输入遥感图像和类别名称，输出每类独立 sigmoid 掩码，最后逐像素 argmax。
SAM3 提供冻结的图像特征、类条件 encoder、Pixel Decoder 和 semantic head；
RemoteCLIP 提供遥感图文相似度；可训练 Refiner 在低分辨率更新 feature 与 score 两路特征。
当前仅支持 semantic 模式。

```text
图像 → 冻结 SAM3 backbone → FPN288、FPN144、FPN72
类别提示 → 冻结完整6层 encoder → 前置 prompt cross-attention → encoder72
图像与64个文本模板 → RemoteCLIP → 模板分数与图像特征融合 → score36
encoder72 → 双线性下采样 → feature36

全部提示一起运行4层 Refiner；每层：
  类间双 value 注意力（含 SAM 文本均值）→ 更新 feature36、score36
  feature36 → 2×2平均池化到18 → 全图自注意力（相对位置偏置）→ 插值到36
  四次逐像素普通3×3局部注意力；复用该层全局上下文；更新 feature、score
  Feature FFN → Score FFN
最后 feature36 → 通道 LayerNorm

随后按提示块执行：
  学生：feature36 → 插值72 → 与原始encoder72/FPN72双分支融合
       → 冻结 SAM3 Pixel Decoder（保留输入梯度）→ 冻结 semantic head → student logits288
  教师（按需）：原始encoder72 → 同一个冻结 Pixel Decoder（no_grad）
       → 同一个冻结 semantic head（no_grad）→ detached teacher logits288
每个块立即计算原有 loss 和 backward，proxy 梯度汇总后回传至低分辨率 Refiner。
```

删除自定义 RefinerPyramidDecoder，不再在144和288尺度做可训练的双分支融合。
SAM3 原始 Pixel Decoder 仍执行72→144→288上采样，继续使用 FPN144、FPN288。

## 2. 张量约定

`B` 为图像数，`M` 为原始类别数，`P` 为逗号拆分后的提示数，`P_chunk` 为当前块的提示数，
`N` 为当前块的图像—提示对数（B乘P_chunk），`D` 为 SAM3 特征通道数256，
`D_clip` 为 RemoteCLIP 特征通道数768，`K` 为每类模板数64。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `backbone_fpn` | 各为 `[B,256,H,W]` | 顺序固定为288、144、72，H/W为对应尺度 |
| `cross_attended_encoder_features_72` | `[B,P,256,72,72]` | 原始冻结encoder与前置文本交叉注意力输出 |
| `sam_text_mean` | `[B,P,256]` | SAM文本有效token均值 |
| `remoteclip_feat_map` | `[B,768,36,36]` | RemoteCLIP图像特征 |
| `template_clip_text` | `[P,64,768]` | 模板文本特征 |
| `clip_score_maps_36` | `[B,P,64,36,36]` | 模板相似度图 |
| `clip_score_embed_36` | `[B,P,256,36,36]` | 初始RemoteCLIP score stream |
| `score_embed_36` | `[B,P,256,36,36]` | Refiner更新后的score stream |
| 层内全图注意力 | 每个图像—提示对324个token | 18×18全图空间注意力；不跨提示 |
| `context_36` | `[B,P,256,36,36]` | 层内全局上下文，四次局部注意力复用 |
| `refiner_features_36` | `[B,P,256,36,36]` | 最终经过通道LayerNorm的feature stream |
| 融合输出 | `[N,256,72,72]` | 学生Pixel Decoder的类条件输入 |
| `final_logits` | `[B,P_chunk,288,288]` | 学生最终logits，逐块计算 |
| `sam3_teacher_logits` | `[B,P_chunk,288,288]` | detached教师logits，按需计算 |

训练损失和评测按需用最近邻插值把标签映射到logits尺度。

## 3. SAM3 分支

### 3.1 图像特征

SAM3 接收 1008×1008 的标准化图像。ViT patch size 为 14，主干 token grid 为 72×72。SimpleFPN 产生 288×288、144×144、72×72 和 36×36 四级特征；当前 `scalp=1` 丢弃最低分辨率的 36×36 级，因此主路径保留前三个尺度。

SAM3 图像 backbone 在训练中冻结并运行于 `eval()`。图像特征使用 `torch.no_grad()` 计算并 detach。

### 3.2 类条件 encoder 与提示展开

原始类别名称支持逗号分隔的多个提示词。例如 `"ship, vessel"` 展开为两个独立提示，`"bridge"` 保持一个。展开后的提示按 `prompt_chunk_size` 分块（默认每块 4 个提示），以控制显存。

模型在展开后的提示空间运行。训练标签通过 `prompt_to_class_id` 把同一原始类别的所有提示映射到相同标签；推理时通过像素级最大值合并回原始类别。

每个图像与每个提示组成一个 prompt pair。冻结的 SAM3 文本编码器和 6 层 transformer encoder 为每个 pair 生成类条件图像特征。所有 6 层在 `torch.no_grad()` 中一次运行完毕。

完整 encoder 输出后，执行一次 prompt cross-attention（同样在 `no_grad()` 中），得到 cross-attended full-encoder feature。SAM 文本向量通过有效 token 的 masked mean 得到，padding token 不参与平均。

所有提示块按原始顺序重新拼接。

### 3.3 共享冻结 Pixel Decoder

学生与教师调用相同的 `UniversalSegmentationHead.forward()`，先将类条件72特征
替换 FPN 列表最后一项，再由 Pixel Decoder 产生288特征并交给 semantic head。
Pixel Decoder 内部仍使用原始 nearest 插值、FPN相加、3×3卷积、GroupNorm和ReLU。
不再暴露 `forward_multiscale()` 或 `forward_semantic_pixel_pyramid()` 接口。

参数均冻结且保持eval，但学生调用保留autograd，教师调用位于no_grad。
开启蒸馏且当前块存在可蒸馏提示时，Pixel Decoder执行两次；否则仅执行学生一次。

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

### 5.1 初始化与层数

cross-attended encoder72双线性下采样为初始feature36；RemoteCLIP的score embedding
直接作为score36。进入Refiner前不注入SAM3 FPN。默认4层、8个heads、dropout=0.1。
所有层都在全部提示上执行，不能移到提示块循环中。

### 5.2 类间注意力

保留原来的类间注意力：每个空间位置跨提示计算。Q/K拼接feature、SAM文本有效token均值、
score embedding；feature和score分别作为两路value，使用独立投影，并分别残差更新。
此处的Q为查询特征、K为用于相关性匹配的键特征、V为提供聚合内容的值特征。

### 5.3 全图上下文

类间残差更新后的feature36，按图像—提示对展开，用普通2×2平均池化下采样到18×18。
沿通道LayerNorm后，由同一特征产生Q/K/V，做324个空间token的全图自注意力。
注意力只在同一个提示内部进行，不跨图像或类别。输出投影后双线性插值回36×36。

全图注意力加入可学习二维相对位置偏置：每个head有35×35个偏置，
覆盖行和列方向分别从-17到17的相对位移；根据查询和键的位置差查表，
在softmax之前加入注意力分数。偏置正常初始化并参与训练。

每层计算一次全局上下文，供该层全部局部注意力复用；不detach，不额外残差到主feature。
下一层根据更新后的feature重新计算自己的上下文，各层参数独立。

### 5.4 四次普通3×3局部注意力

用滑动3×3邻域替换原来的regular/shifted窗口注意力。每个查询像素只关注自己及八个相邻像素。
步长1、padding1、dilation1；无空洞、采样间隔、移位或大窗口划分。
边缘的padding位置在softmax前屏蔽，不参与权重归一化。

| 输入 | 拼接内容 | 默认投影尺寸 |
| --- | --- | --- |
| Q/K | 当前feature + 当前score + 本层全局上下文 | 768→256，各自独立投影 |
| feature value | 当前feature + 全局上下文 | 512→256 |
| score value | 当前score，单独使用 | 256→256 |

拼接前各输入沿通道执行各自的LayerNorm。局部注意力还为九个相对偏移设置每head可学习偏置。
一组注意力权重聚合两路value；两路使用独立输出投影，各自直接残差加到当前feature和score。
后一次读取前一次更新后的两路特征。默认四次，次数由 `local_attn_steps=4` 配置，
每次局部注意力参数独立。随后保留Feature FFN和Score FFN，均为pre-norm直接残差。
所有Refiner层结束后，只对feature增加一次输出通道LayerNorm；score不增加输出norm。

### 5.5 单尺度 Decoder Input Fusion

`DecoderInputFusion`只在72×72融合。Refiner36先双线性插值到72。
四路独立的 `1×1 Conv + GroupNorm` 把256通道投影到128（无激活）：
Refiner语义投影、Refiner细节投影、原始encoder72投影、原始FPN72投影。
这里的原始encoder72始终指前置prompt cross-attention之后、Refiner之前的冻结特征。

语义输入为Refiner语义投影与encoder投影相加；细节输入为Refiner细节投影与FPN投影相加。
FPN先按图像投影到128，再按提示广播，不复制256通道图像级FPN。
每条分支分别执行普通3×3卷积→GN→GELU→1×1卷积→GN，并将块输出残差加到分支输入。
两路分别用独立1×1卷积恢复256通道，相加后再用256→256的1×1卷积输出。
输出不加额外norm、激活或原始encoder残差；不设置残差系数或零初始化。

融合输出作为学生Pixel Decoder的72输入；不使用教师Pixel Decoder中间特征作为融合输入。
融合模块在训练时按 `use_checkpoint` 做一次non-reentrant checkpoint。
每个提示块重算可训练FPN投影，不跨块缓存计算图。

## 6. 冻结分割头与梯度边界

完整encoder与前置prompt cross-attention仍在no_grad中执行。
教师只从原始encoder72出发，整条Pixel Decoder与semantic head路径no_grad，输出detach。
学生从融合encoder72出发，Pixel Decoder与semantic head参数冻结，但调用时保留输入梯度。
学生最终BCE与原有蒸馏损失因此回传到DecoderInputFusion、Refiner、全图/局部注意力与RemoteCLIP可训练部分。
Pixel Decoder、semantic head、SAM3原始encoder和backbone参数都不更新。

## 7. 训练设计

### 7.1 冻结与微调

以下 SAM3 模块冻结并保持 `eval()`：

* backbone；
* transformer encoder（完整 6 层，在 `no_grad()` 中执行）；
* geometry encoder；
* segmentation head（Pixel Decoder 参数冻结并保持 `eval()`。原始分支在 `no_grad()` 中执行，Refiner 分支在梯度开启状态下执行。semantic head 在原始分支中产生 detach 的 teacher logits，在 student 分支中产生可回传梯度的 student logits）。

完整 SAM3 encoder 和前置 prompt cross-attention 不保留计算图，均在 `torch.no_grad()` 中执行。

`core.encoder_refiner` 完整训练。其内部的 Refiner 层、`DecoderInputFusion` 和各层全图/局部注意力同属一个参数组，由现有 `trainable_modules=["core.encoder_refiner"]` 自动覆盖，使用基础学习率 `1e-4`。最终掩码 logits 由冻结的 SAM3 `semantic_seg_head` 产生。

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

**SAM3 teacher 掩码蒸馏**（固定权重 `sam3_mask_distill_weight=0.1`）：

蒸馏权重在整个训练过程中保持不变。只要当前 batch 存在可蒸馏提示，就计算 teacher logits。

蒸馏监督范围：

1. 冻结的 SAM3 semantic head 产生的 teacher logits 做 sigmoid，得到 soft probability 目标。
2. student 使用 raw `final_logits`。
3. 用 `binary_cross_entropy_with_logits` 逐像素计算蒸馏损失。
4. 只对 GT 中存在的图像—提示对计算。
5. 所有 `label != 255` 的有效像素参与蒸馏。
6. 对每个存在原始类别，额外包含 GT 外侧指定宽度的边界环。
7. 外侧边界环只保留 `label == 255` 的位置。
8. 外环宽度由 `sam3_mask_distill_boundary_width` 控制，默认 2。
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
    + 0.1 * loss_sam3_mask_distill_bce
)
```

`loss_sam3_mask_distill_bce` 为未经权重的原始蒸馏 BCE；`loss_sam3_mask_distill_weighted` 为真正加入总损失的加权贡献。`0.1` 是整个训练期间固定不变的蒸馏权重。

**训练显存设计**：

* Refiner36 对所有类别一次计算。
* 高分辨率按 chunk 计算。
* 每个 chunk 立即计算 loss 和 backward。
* 使用 detached leaf proxy 累积高分辨率梯度。
* 所有 chunk 完成后把 proxy.grad 一次传回真实 Refiner 图。
* optimizer/scaler/scheduler 每个 batch 只更新一次。
* 不保留多个 chunk 的高分辨率特征或 logits；教师按需计算。
* 不使用 `retain_graph=True`。

## 8. 推理与评测

推理时模型在提示空间输出 `prompt_logits` [B, P, H, W]。Adapter 先做 sigmoid 得到 `raw_prompt_score_map`，再通过像素级最大值把同一原始类别的所有提示合并为 `raw_final_score_map` [B, M, H, W]。可选的逐类别相对阈值在原始类别空间执行，归一化后只把未通过位置置 0；保留位置继续使用原始 sigmoid 分数。最终对类别维取 argmax。

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

Refiner新配置：`local_attn_steps=4`；删除 `window_size`、`shift_size`。
全图尺度固定18×18；普通3×3邻域固定，不添加采样间隔配置。

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
| `models/sam3_image.py`                | 类别 chunk、缓存、SAM3 encoder、低分辨率 refiner、逐 chunk 高分辨率解码 |
| `models/encoder_refiner.py` | 全类别Refiner、最终feature LayerNorm与72融合接口 |
| `models/decoder_input_fusion.py` | 72尺度语义/细节双分支融合 |
| `models/refiner_spatial_attention.py` | 18全图注意力、相对位置偏置与普通3×3局部双value注意力 |
| `models/encoder_refiner_attention.py` | 跨类别注意力与Refiner层执行顺序、双流 FFN、pre-norm 与直接残差更新            |
| `models/maskformer_segmentation.py`   | prompt attention、Pixel Decoder原始上采样和原始 semantic head |
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

1. 提示展开、块顺序、标签映射、sigmoid和最终argmax保持一致。
2. encoder72、Refiner36、RemoteCLIP36和全局注意力18尺度固定；SAM3通道固定256。
3. Refiner先在全部提示上运行，再按提示块解码并backward；不使用retain_graph。
4. 类间注意力保留SAM文本均值与双value；全图注意力只聚合feature，局部score value不拼接上下文。
5. 全图上下文每层一次，保留梯度；局部注意力每次用最新feature/score，默认4次。
6. 全图和局部注意力都有可学习相对位置偏置；局部padding位置必须屏蔽。
7. 删除自定义多尺度Pyramid Decoder；仅在72做可训练双分支融合；SAM3原始上采样仍保留。
8. teacher仅从原始encoder72产生且detach；学生冻结解码器调用必须保留输入autograd。
9. 教师和学生共用一套冻结Pixel Decoder与semantic head；推理和不需要教师的块只执行学生。
10. 初始score embedding保持当前RemoteCLIP相似度/内容融合，两次拼接前逐像素L2归一化。
11. 全部Refiner层后只对feature执行一次LayerNorm；不引入固定或可学习残差系数。
12. 图像级FPN72先降到128通道再按类别广播；FPN144/FPN288继续用于原始Pixel Decoder。
13. 原有蒸馏范围、权重、全局分母、多提示语义以及最终BCE规则保持不变。
14. optimizer/scaler/scheduler每batch只更新一次；proxy梯度一次传回低分辨率图。
15. 可训练RemoteCLIP文本不跨optimizer step缓存；验证不开启额外autograd。
16. TTA先平均提示空间分数，再合并类别并执行相对阈值；标签空间只重映射一次。

限制：仅semantic模式，无非空几何提示；不支持动态空间尺寸或多尺度TTA；
CLIP中间层只用于debug；默认iSAID训练、LoveDA验证。

## 12. 参数结构与验证

本次结构不兼容旧训练checkpoint：删除 `pyramid_decoder.*`、两次窗口注意力及其norm/位置偏置；
新增 `decoder_input_fusion.*`、每层 `global_attn.*`、`local_attns.*`。
旧checkpoint不能通过 `--resume-from` 严格恢复；新实验使用新work directory。
不增加旧配置、旧参数映射或兼容模块。完整checkpoint容器schema继续保持4，
SAM3原始Pixel Decoder与semantic head的参数名和形状不变。

不需要下载模型权重即可运行小张量回归检查：

```bash
python -m unittest discover -s tests -v
```

数值测试需要已安装PyTorch；缺少时明确跳过，不代表模型前向或训练已验证。
覆盖局部3×3结果与显式邻居参考计算、边缘屏蔽、全局相对位置偏置梯度、
类间隔离、全局每层一次、融合的checkpoint梯度以及冻结Pixel Decoder输入梯度。

