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
  ├─ SAM3 文本编码器与 transformer encoder layer 1..n
  │    → 每个图像-类别对的第 n 层 encoder feature
  │    → SAM 文本 token 的 masked mean
  └─ RemoteCLIP
       ├─ 504×504 图像 → 36×36 dense image feature
       └─ 每类 32 个文本模板 → template text feature

RemoteCLIP 局部相似度图
  → 多尺度 score encoder
  → 36×36 score embedding
  → SAM3 FPN 残差注入
  → 4 层双流 encoder refiner
  → FPN 引导的 36→72 细节上采样
  → 写回第 n 层 visual tokens
  → SAM3 encoder layer n+1..6
  → prompt cross-attention
  → pixel decoder
  → semantic head
  → final logits
```

## 2. 张量约定

| 记号       | 含义                             |
| -------- | ------------------------------ |
| `B`      | batch 中的图像数                    |
| `C`      | 当前前向传播的类别数                     |
| `K`      | 每类 RemoteCLIP 文本模板数，固定为 32     |
| `D`      | SAM3 hidden dimension，固定为 256  |
| `D_clip` | RemoteCLIP 投影维度，ViT-L/14 为 768 |

当前固定输入下的主要张量为：

| 张量                            | 形状                                 | 说明                             |
| ----------------------------- | ---------------------------------- | ------------------------------ |
| `backbone_fpn`                | `[B, 256, 288/144/72, 288/144/72]` | SAM3 多尺度图像特征                   |
| `encoder_features_72`         | `[B, C, 256, 72, 72]`              | 第 n 层 encoder feature           |
| `sam_text_mean`               | `[B, C, 256]`                      | SAM 文本 token 的 masked mean     |
| `remoteclip_feat_map`         | `[B, 768, 36, 36]`                 | RemoteCLIP dense image feature |
| `template_clip_text`          | `[C, 32, 768]`                     | 每类 32 个模板的文本特征                 |
| `clip_score_maps_36`          | `[B, C, 32, 36, 36]`               | 局部图文相似度图                       |
| `score_embed_36`              | `[B, C, 256, 36, 36]`              | refiner 的语义分数流                 |
| `refiner_features_36`         | `[B, C, 256, 36, 36]`              | refiner 的图像特征流                 |
| `refined_encoder_features_72` | `[B, C, 256, 72, 72]`              | 写回 SAM3 的更新特征                  |
| `final_logits`                | `[B, C, 288, 288]`                 | 当前固定输入下的语义分割 logits            |

训练损失和评测会在必要时用最近邻插值把标签映射到 logits 尺度。

## 3. SAM3 分支

### 3.1 图像特征

SAM3 接收 1008×1008 的标准化图像。ViT patch size 为 14，主干 token grid 为 72×72。SimpleFPN 产生 288×288、144×144、72×72 和 36×36 四级特征；当前 `scalp=1` 丢弃最低分辨率的 36×36 级，因此主路径保留前三个尺度。

SAM3 图像 backbone 在训练中冻结并运行于 `eval()`。图像特征使用 `torch.no_grad()` 计算并 detach。

### 3.2 类条件 encoder

同一 batch 的所有样本必须共享完全相同的类别名称和顺序。类别按 `prompt_chunk_size` 分块，默认每块 8 类，以控制显存。

每个图像与每个类别组成一个 prompt pair。冻结的 SAM3 文本编码器和 6 层 transformer encoder 为每个 pair 生成：

* 类条件图像特征（前 n 层输出，n 由 `insert_after_encoder_layer` 控制）；
* prompt token；
* prompt mask。

默认在第 4 层后插入 Refiner：前 4 层在 `torch.no_grad()` 中运行，第 5～6 层保留输入梯度并以逐层 activation checkpoint 执行。所有类别块按原始顺序重新拼接。SAM 文本向量通过有效 token 的 masked mean 得到，padding token 不参与平均。

Refiner 对所有类别拼接后的 feature 只运行一次，输出按原 chunk 切分后写回第 n 层 visual tokens，再分别运行剩余 encoder 层和分割头。

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

输出 `clip_score_embed_36`，并直接作为 refiner 的初始 score 流。SAM3 FPN 不注入 score 流。

## 5. Class-conditioned encoder refiner

Refiner 在 36×36 上同时维护图像 feature 流和 score embedding 流。默认使用 4 层、8 个 attention heads、12×12 窗口和 6 像素 shift。

### 5.1 FPN 信息注入

原始 72×72 encoder feature 双线性下采样到 36×36。SAM3 的 72×72 FPN 同样下采样，并沿类别维广播。

两者在通道维拼接为 512 通道，经过两层 3×3 Conv、GroupNorm 和 GELU，产生 256 通道更新量。更新量由零初始化标量 `feature_fpn_res_scale` 调制后加到下采样 encoder feature。

该设计让 refiner 初始时保持原始 encoder 基线，同时允许训练逐步引入共享的 SAM3 高频信息。

### 5.2 单层 refiner

每层采用 pre-norm，并依次执行：

1. **ClassScoreAttention**：在每个空间位置跨类别做注意力。Q/K 由图像 feature、SAM 文本均值和 score embedding 拼接后投影；feature 与 score 使用独立 value/output 分支，因此两路都会更新。
2. **Regular WindowScoreAttention**：每个类别内部执行非移位窗口注意力，同时更新 feature 与 score。
3. **Shifted WindowScoreAttention**：使用 shift mask 和相对位置偏置连接相邻窗口，同样更新两路。
4. **Feature FFN**：逐 token 更新图像流。
5. **Score FFN**：逐 token 更新分数流。

三个 attention 子层分别为 feature 和 score 使用独立残差系数，两个 FFN 也分别使用独立系数，因此每层共有 8 个 LayerScale 标量。所有标量默认初始化为 0。

层内不使用 post-norm。全部 refiner 层结束后，只对 score embedding 执行一次最终 LayerNorm；feature 流不做最终 LayerNorm。

### 5.3 36→72 细节上采样

上采样模块不在进入模块前预先计算更新量，而是执行完整流程：

1. 将完整 `refiner_features_36` 双线性插值到 72×72。
2. 与原始 `encoder_features_72` 相减，得到原始更新量。
3. 将更新量与按类别广播的 `sam_fpn_72` 在通道维拼接为 512 通道。
4. 使用 1×1 Conv、GroupNorm 和 GELU 压回 256 通道，避免拼接扩大最终特征尺度。
5. 通过轻量细节分支提取局部高分辨率信息：depthwise 3×3 Conv、GroupNorm、GELU、pointwise 1×1 Conv。
6. 将细节分支输出残差加到 1×1 融合结果。
7. 使用零初始化 `upsample_res_scale` 调制处理后的更新量，再加回原始 72×72 encoder feature。

这里只使用一个 depthwise 3×3 空间卷积，避免连续普通卷积过度平滑边界；1×1 卷积负责通道融合和容量控制。零初始化外层残差保证模型初始化时的最终 72×72 输出与原始 encoder feature 完全一致。

Refiner 层和上采样模块在训练时支持 activation checkpoint。Refiner 输出写回第 n 层 encoder feature，再经过第 n+1..6 层 encoder（逐层 activation checkpoint），然后进入分割头。

### 5.4 残差系数日志

训练日志、JSONL 和 W&B 分三类记录可学习残差系数：

| 类别         | 前缀                           | 参数                      |
| ---------- | ---------------------------- | ----------------------- |
| FPN 信息注入   | `residual/fpn/`              | `feature_fpn_res_scale` |
| Refiner 内部 | `residual/refiner_internal/` | 每层 8 个 LayerScale       |
| 上采样模块      | `residual/upsample/`         | `upsample_res_scale`    |

每类记录 `count`、`mean`、`abs_mean`、`min`、`max` 和 `negative_ratio`。当前模型未发现第四类可学习残差系数。

## 6. SAM3 分割头

分割头按 SAM3 官方控制流固定执行一次 prompt cross-attention（当 `cross_attend_prompt` 不为 None 时），不再有开关控制。

Pixel decoder 用更新后的 72×72 类条件特征替换 FPN 最后一层，再依次与 144×144、288×288 的 SAM3 FPN 融合。每次融合包含最近邻上采样、逐元素相加、3×3 Conv、GroupNorm 和 ReLU。1×1 semantic head 为每个图像-类别 pair 输出一个 logit map，最终拼接为 `[B, C, 288, 288]`。

SAM3 segmentation head 的参数保持冻结，但前向不能放入 `no_grad()`，因为损失梯度仍需穿过该头回到 refiner。

语义主路径只消费 `semantic_seg`。当前通用 segmentation head 按 SAM3 官方控制流仍计算 `pred_masks`，语义主路径只消费 `semantic_seg`；本次不额外裁剪该官方分支。

## 7. 训练设计

### 7.1 冻结与微调

以下 SAM3 模块冻结并保持 `eval()`：

* backbone；
* transformer encoder；
* geometry encoder；
* segmentation head。

`core.encoder_refiner` 完整训练。RemoteCLIP 图像和文本分支默认使用 `attention` 微调模式，仅训练注意力 Q/V 与位置嵌入，同时保持 `eval()` 以关闭 dropout 和 patch dropout。

OpenCLIP 常把 Q/K/V 存在同一个融合参数中。项目对该参数注册梯度 mask，使 K 区域梯度为 0；同时把整个融合参数组的 weight decay 强制设为 0，防止 AdamW 在无梯度时仍修改 K。恢复 optimizer 状态后会重新应用这一不变量。

默认 AdamW 基础学习率为 `1e-4`：

* encoder refiner 使用 1.0 倍学习率；
* RemoteCLIP text/image 使用 0.01 倍学习率，即 `1e-6`；
* normalization 参数不使用 weight decay。

### 7.2 损失

每个类别通道独立使用 binary mask 监督，不使用跨类别 softmax。

* 存在于图像中的类别：有效像素参与 BCE，ignore 像素作为低权重负样本抑制泄漏。
* 不存在于图像中的类别：只在有效像素上计算 BCE，并使用较小的 pair 权重。
* Dice 只对存在类别计算；当前默认权重为 0。
* 全部像素均为 ignore 时跳过 backward 和 optimizer step，但已消费的数据批次仍会提交给 sampler。

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
| `models/sam3_image.py`                | 类别 chunk、缓存、SAM3 encoder、refiner 调用和特征写回 |
| `models/openclip_image_encoder.py`    | 36×36 dense RemoteCLIP 图像特征              |
| `models/openclip_text_encoder.py`     | 模板文本编码、micro-batch 与梯度控制                 |
| `models/score_embeddings.py`          | 32 模板相似度图和多尺度 score encoder              |
| `models/encoder_refiner_attention.py` | 跨类别/窗口注意力、双流 FFN 与 LayerScale            |
| `models/encoder_refiner.py`           | FPN 注入、refiner 主体和细节上采样                  |
| `models/maskformer_segmentation.py`   | prompt attention、pixel decoder 与语义 head  |
| `losses/semantic_criterion.py`        | present/absent 加权 BCE 与可选 Dice           |
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
7. 冻结的 segmentation head 必须允许梯度穿过。
8. FPN、refiner 内部和上采样的外层残差系数均零初始化。
9. TTA 必须先平均原始分数，再进行相对阈值过滤。
10. `reduce_zero_label` 与 `background_cfg` 各自只执行其定义的一次标签空间变换。
11. 完整恢复必须严格校验 checkpoint schema；模型权重迁移必须走独立入口。
12. `1 <= insert_after_encoder_layer < 6`，前 n 层在 `no_grad()` 中，后 6-n 层保留输入梯度。

当前限制：

* 只支持 semantic 模式；
* 不支持非空几何 prompt；
* 不支持动态 SAM3/refiner 空间尺寸或多尺度 TTA；
* `clip_mid_features` 只用于 debug；
* 当前默认训练集为 iSAID，验证集为 LoveDA。
