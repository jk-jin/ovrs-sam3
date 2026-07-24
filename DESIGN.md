# OVRS-SAM3 设计与实现说明

更新时间：2026-07-24
目标分支：`master`
项目仓库：`jk-jin/ovrs-sam3`
任务状态：当前只实现开放词汇语义分割；hybrid模式尚未实现

> 本文描述的是当前 `master` 分支上的实际实现。后续修改模型时，应同步更新本文，避免文档再次落后于实现。

## 1. 给AI的快速阅读入口

如果需要在最短时间内理解本项目，请先记住下面五点：

1. 项目目标是遥感图像开放词汇语义分割：输入图像和一组类别名称，输出每个像素的类别。
2. SAM3主体负责生成图像特征、SAM文本特征和最终mask logits；SAM3主体在训练中冻结。
3. RemoteCLIP负责建立类别文本与局部图像区域的对应关系；默认只微调其注意力Q/V和位置嵌入。
4. 真正的核心可训练模块是`ClassConditionedEncoderRefiner`：它在36×36尺度上联合更新SAM3图像feature和类别score embedding。
5. 更新后的feature写回SAM3的`encoder_hidden_states`，仍由冻结的SAM3 segmentation head生成最终分割结果。

建议继续按以下顺序阅读代码：

```text
configs/_base_/model/ovrs_sam3.py
  → configs/train/isaid_loveda_exp.py
  → model_builder.py
  → models/sam3_image.py
  → models/openclip_image_encoder.py
  → models/openclip_text_encoder.py
  → models/score_embeddings.py
  → models/encoder_refiner.py
  → models/encoder_refiner_attention.py
  → losses/semantic_criterion.py
  → data/dataset.py
  → engine/evaluator.py
```

## 2. 项目定位

`ovrs-sam3`把三个能力组合在一起：

* SAM3的遥感图像编码能力。
* SAM3基于文本prompt生成mask的能力。
* RemoteCLIP的遥感图文对齐能力。

SAM3原生encoder feature具有较强的图像结构信息，但不一定能直接形成适合开放类别语义分割的类别响应。RemoteCLIP能判断局部图像区域与类别文本的相似程度，但其36×36 dense feature不直接替代SAM3特征。

本项目的做法是：先用RemoteCLIP生成类别相关的score embedding，再让encoder refiner把它与SAM3 encoder feature联合更新，最后把更新后的SAM3 feature交回原有segmentation head解码。

当前主流程为：

```text
遥感图像 + 当前数据集的类别名称
  → 冻结的SAM3图像backbone提取FPN特征
  → 冻结的SAM3文本与transformer encoder生成72×72类别相关feature和SAM文本token
  → RemoteCLIP图像分支生成36×36 dense image feature
  → RemoteCLIP文本分支用32个模板生成类别文本特征
  → 文本特征与dense image feature计算32张类别相似度图
  → 多尺度卷积编码器把32通道相似度图编码成256通道clip_score_embed_36
  → SAM3 FPN下采样到36×36，与encoder feature 36拼接后卷积产生残差更新
  → encoder refiner联合更新feature_36与score_embed_36
  → 计算36×36总更新量（refined feature - baseline feature），双线性插值到72×72
  → 直接加到原始72×72 encoder feature
  → 写回SAM3 encoder_hidden_states的图像token区域
  → 冻结的SAM3 segmentation head输出final_logits
```

## 3. 关键符号与张量约定

| 符号             | 含义                                       |
| -------------- | ---------------------------------------- |
| `B`            | batch size，一次输入的图像数量。                    |
| `C`            | 当前前向传播使用的类别数量。背景被排除时，它可能少于评估类别数。         |
| `K`            | 每个类别的RemoteCLIP文本模板数量，当前固定为32。           |
| `T`            | SAM3文本prompt的token数量。                    |
| `D`            | SAM3 hidden dimension，当前固定为256。          |
| `D_clip`       | RemoteCLIP图文对齐空间维度，ViT-L/14当前为768。       |
| `D_native`     | RemoteCLIP ViT投影前的原生通道数，ViT-L/14通常为1024。 |
| `D_score`      | score embedding通道数，当前固定为256。             |
| `H_out, W_out` | segmentation head最终输出的空间尺寸。              |
| `L_refiner`    | encoder refiner层数，默认4。                   |
| `ws`           | window attention窗口边长，默认12。               |
| `shift`        | shifted window attention平移距离，默认6。        |

主要张量形状：

| 张量                            | 形状                     | 含义                                          |
| ----------------------------- | ---------------------- | ------------------------------------------- |
| `encoder_features_72`         | `[B, C, 256, 72, 72]`  | SAM3 transformer encoder产生的原始类别相关图像feature。 |
| `sam_fpn_72`                  | `[B, 256, 72, 72]`     | SAM3 backbone FPN的72×72纹理feature。           |
| `sam_text_mean`               | `[B, C, 256]`          | SAM3文本token经过masked mean后的类别向量。             |
| `remoteclip_feat_map`         | `[B, 768, 36, 36]`     | RemoteCLIP dense image feature。             |
| `template_clip_text`          | `[C, 32, 768]`         | 每个类别、每个模板的RemoteCLIP文本feature。              |
| `clip_score_maps_36`          | `[B, C, 32, 36, 36]`   | 32个模板分别产生的局部图文相似度图。                         |
| `clip_score_embed_36`         | `[B, C, 256, 36, 36]`  | 相似度图经过卷积编码后的初始score embedding。              |
| `score_embed_36`              | `[B, C, 256, 36, 36]`  | 注入SAM3 FPN并经过refiner更新后的score embedding。    |
| `refiner_features_36`         | `[B, C, 256, 36, 36]`  | refiner最终输出的图像feature。                      |
| `refined_encoder_features_72` | `[B, C, 256, 72, 72]`  | 上采样并融合后的SAM3 encoder feature。               |
| `final_logits`                | `[B, C, H_out, W_out]` | 每个类别的最终mask logits。                         |

## 4. 训练入口与主配置

短实验：

```bash
python tools/train.py configs/train/isaid_loveda_exp.py
```

完整训练：

```bash
python tools/train.py configs/train/isaid_loveda_full.py
```

配置关系：

```text
_base_/model/ovrs_sam3.py          # 模型、RemoteCLIP、refiner、freeze、loss
_base_/optimizer/ovrs_sam3_adamw.py # 优化器与参数组倍率
_base_/schedule/exp_4k.py           # 4K iter 训练计划与scheduler
_base_/schedule/full_20k.py         # 20K iter 训练计划与scheduler
_base_/evaluation.py                # eval_cfg、tta_cfg
_base_/runtime.py                   # seed、work_dir、logger
_base_/tracking.py                  # metrics JSONL与W&B默认设置
_base_/visualization.py             # 可视化默认设置（默认关闭）
_base_/dataloader/semantic_train.py # 训练数据公共transform和collate
_base_/dataloader/semantic_eval.py  # 评估数据公共transform和collate

datasets/train/isaid.py             # iSAID训练集特定参数
datasets/eval/loveda.py             # LoveDA验证集特定参数

train/isaid_loveda_exp.py           # 组合上述模块，覆盖exp专属work_dir、W&B
train/isaid_loveda_full.py          # 组合上述模块，覆盖full专属work_dir、W&B、可视化
```

当前基础设置：

```text
SAM3输入尺寸：1008×1008
RemoteCLIP输入尺寸：504×504
prompt_chunk_size：8
refiner层数：4
refiner尺度：36×36
encoder尺度：72×72
attention heads：8
window_size：12
shift_size：6
score_embed_dim：256
layer_scale_init：0.0
```

训练集默认继承iSAID配置，验证集为LoveDA。

## 5. 冻结、微调与运行模式

### 5.1 始终冻结的SAM3模块

训练中以下模块参数保持冻结，并强制处于`eval()`：

```text
core.backbone
core.transformer
core.geometry_encoder
core.segmentation_head
```

这些模块的前向主要运行在`torch.no_grad()`中。`eval()`用于稳定Dropout等行为；`requires_grad=False`和`no_grad()`负责真正冻结参数与节省显存。

### 5.2 始终训练的encoder refiner

基础配置先冻结整个模型，再解冻：

```python
trainable_modules=["core.encoder_refiner"]
```

它包含：

* `ClipScoreEmbedding.score_encoder`。
* SAM3 FPN与encoder feature的融合卷积（feature_fpn_fusion）。
* ClassScoreAttention。
* regular和shifted WindowScoreAttention。
* feature与score两路FFN和LayerNorm。
* 各层可学习残差系数（LayerScale）和FPN残差系数（feature_fpn_res_scale）。
* final_score_norm（score embedding的最终LayerNorm）。

### 5.3 RemoteCLIP微调模式

文本和图像分支分别支持：

| 模式            | 行为                           |
| ------------- | ---------------------------- |
| `frozen`      | 全部参数冻结，不为该分支建立训练计算图。         |
| `attention`   | 训练注意力Q/V与位置嵌入，K保持冻结。         |
| `transformer` | 训练Transformer与位置嵌入，其他外围参数冻结。 |
| `full`        | 训练整个对应编码器。                   |

当前默认配置：

```python
openclip_text_finetune="attention"
openclip_image_finetune="attention"
```

RemoteCLIP即使部分可训练也保持`eval()`，目的是关闭Dropout和patch dropout的随机行为。这不等于冻结；只要参数`requires_grad=True`、前向建立计算图且输出没有detach，参数仍能正常收到梯度。

### 5.4 Q/V only的严格含义

OpenCLIP的多头注意力常把Q、K、V权重存放在同一个`in_proj_weight`中。项目对融合QKV参数注册梯度mask：

```text
Q区域：保留梯度
K区域：梯度清零
V区域：保留梯度
```

仅把K梯度清零仍不足以严格冻结K，因为AdamW可能对整个融合参数应用weight decay。修复后的实现会给受Q/V mask保护的参数添加运行时标记，`OptimizerBuilder`据此把整个融合参数的weight decay强制设为0。

这样可以保证：

* K没有反向传播梯度。
* K不会被AdamW权重衰减修改。
* Q/V仍可通过梯度更新。
* 代价是同一个融合参数中的Q/V也不使用weight decay。

RemoteCLIP最后一个图像block使用dense value-branch，因此该block实际只消费V分支；它的Q梯度可能合法地为0。前面的标准attention block仍会使用Q和V。

### 5.5 学习率

基础AdamW学习率为`1e-4`。参数组设置为：

```text
encoder refiner学习率倍率：1.0，实际初始学习率1e-4
RemoteCLIP text学习率倍率：0.01，实际初始学习率1e-6
RemoteCLIP image学习率倍率：0.01，实际初始学习率1e-6
```

这里`1e-6`来自`1e-4 × 0.01`：`1e-4`是基础学习率，`0.01`是RemoteCLIP参数组倍率。

## 6. 三种不同缓存必须区分

项目里存在三种含义不同的缓存。

### 6.1 SAM3类别文本缓存

`Sam3Image.prepare_text_cache()`缓存冻结的SAM3文本backbone输出：

```text
language_features
language_mask
language_embeds（存在时）
```

SAM3文本backbone始终冻结，因此该缓存可以跨batch复用。

### 6.2 RemoteCLIP模板文本缓存

`ClipScoreEmbedding`缓存32模板对应的RemoteCLIP文本feature，但只允许在RemoteCLIP文本编码器完全冻结时使用。

| 文本模式          | 是否允许跨step缓存模板feature |
| ------------- | -------------------- |
| `frozen`      | 允许，缓存内容detach。       |
| `attention`   | 不允许，每个step重新编码。      |
| `transformer` | 不允许，每个step重新编码。      |
| `full`        | 不允许，每个step重新编码。      |

原因是可训练文本参数在optimizer step后会变化，旧模板feature会过期；复用带计算图的旧feature还会造成错误的反向传播。

验证阶段处于外层`torch.no_grad()`，因此不建立梯度图。但只要文本编码器属于可训练模式，仍不启用长期模板缓存，以免训练恢复后读到旧参数产生的结果。

### 6.3 单次前向的encoder refiner cache

`build_encoder_refiner_cache()`为当前batch收集：

```text
SAM3 encoder feature chunks
prompt与prompt mask
SAM3 FPN
RemoteCLIP dense image feature
SAM text mean
类别名称和chunk信息
```

该cache只服务同一次模型前向，不是跨训练step的特征缓存。RemoteCLIP图像分支可训练时，cache中的主`clip_image_feat_map`必须保留计算图。

## 7. SAM3分支与类别chunk

### 7.1 SAM3图像backbone

输入的标准化图像经过冻结的SAM3图像backbone，得到FPN特征和位置编码。主流程保存`backbone_fpn`供encoder refiner和segmentation head使用。

### 7.2 类别文本与prompt chunk

同一batch中的所有图像共享相同的类别名称及顺序。为了控制SAM3 encoder和segmentation head的显存，占用类别维度按`prompt_chunk_size`切分，当前默认每个chunk包含8个类别。

对于每个类别chunk：

1. 从SAM3文本缓存切出当前类别的文本feature。
2. 按`B × C_chunk`扩展语义prompt对。
3. 运行冻结的SAM3 grounding encoder。
4. 提取72×72图像token feature。
5. 提取SAM文本token并计算`sam_text_mean`。

所有chunk最终按原始类别顺序拼接，必须完整覆盖`0 ... C-1`，不允许缺失、重复或乱序。

### 7.3 SAM文本masked mean

输入：

```text
prompt_tokens [T, B*C, 256]
prompt_mask   [B*C, T]
```

处理：

```text
valid = logical_not(prompt_mask)
sum_valid_tokens / valid_token_count
  → sam_text_mean [B, C, 256]
```

`T`表示文本token数量，`B*C`表示每张图像与每个类别组成的prompt对数量。padding token不参与均值；分母最小限制为1，避免空prompt导致除零。

## 8. RemoteCLIP图像分支

`OpenCLIPImageEncoder`生成36×36 dense image feature。

```text
raw image
  → resize到504×504
  → RemoteCLIP mean/std normalize
  → ViT-L/14 patch embedding
  → 36×36 patch grid
  → 加入class token
  → positional embedding bicubic插值到36×36
  → 前N-1个Transformer block正常forward
  → 最后一个block执行dense value-branch forward
  → ln_post与原始visual projection
  → 去掉class token
  → remoteclip_feat_map [B, 768, 36, 36]
```

504除以patch size 14得到36，因此RemoteCLIP输入被固定为504×504。

最后一个block不执行标准QK注意力聚合，而是：

1. 计算融合QKV投影。
2. 只提取V分支。
3. 经过attention output projection。
4. 向所有token注入class token信息。
5. 经过该block的MLP残差。

输出还包括指定中间层的：

```text
clip_mid_features: List[[B, D_native, 36, 36]]
```

这些中间层feature当前只用于debug，不进入主模型，因此会detach。主`remoteclip_feat_map`在图像编码器可训练时不会detach。

## 9. RemoteCLIP文本分支

### 9.1 32个固定模板

每个类别使用配置中的32个遥感prompt模板：

```text
C个class_names × 32 templates
  → C×32条完整文本
  → tokenize
  → RemoteCLIP text transformer
  → text projection
  → template_clip_text [C, 32, 768]
```

类别名称默认会去除首尾空白，并把下划线、连字符规范为空格。

### 9.2 修复后的梯度规则

通用`encode_text()`保留冻结式推理语义。模板编码不再调用这个带`torch.no_grad()`和detach的接口，而是通过`encode_tokenized()`与`encode_embeds()`执行。

模板路径是否建立计算图由两项共同决定：

```text
全局梯度是否开启：torch.is_grad_enabled()
文本编码器是否存在requires_grad=True的参数
```

不能用`self.training`判断，因为RemoteCLIP文本编码器在微调时也有意保持`eval()`。

结果：

* 训练阶段且文本参数可训练：保留完整计算图。
* 文本编码器完全冻结：显式无梯度并允许缓存。
* 验证阶段：外层`torch.no_grad()`关闭计算图。

### 9.3 文本micro-batch与activation checkpoint

扁平模板数量为`C × 32`。为了避免微调文本Transformer后显存突然增大，使用：

```python
text_prompt_batch_size=64
text_prompt_use_checkpoint=True
```

`text_prompt_batch_size`表示每次送入文本Transformer的完整prompt数量，不是类别数，也不是token数。

可训练时，每个文本micro-batch使用：

```python
checkpoint(..., use_reentrant=False)
```

non-reentrant checkpoint允许输入为不需要梯度的整数token，同时仍能追踪闭包中模型参数的梯度。编码完所有micro-batch后沿第0维拼接，再reshape回`[C, 32, 768]`。

## 10. CLIP相似度图与score embedding

RemoteCLIP文本和图像feature先分别做L2归一化，然后计算点积相似度：

```text
normalized template text [C, 32, 768]
× normalized image feature [B, 768, 36, 36]
→ clip_score_maps_36 [B, C, 32, 36, 36]
→ 乘固定缩放值20.0
```

这里点积表示768个对齐维度逐项相乘后相加；值越大，表示该空间位置与对应类别模板越相似。固定缩放值20.0用于放大相似度数值范围。

32个模板通道经过多尺度score encoder：

```text
[B, C, 32, 36, 36]
  → reshape为[B*C, 32, 36, 36]
  → 1×1卷积融合模板通道，32→256 (stem)
  → 三个并行深度可分离卷积分支
     ├─ 3×3，dilation=1，256→128
     ├─ 3×3，dilation=2，256→128
     └─ 3×3，dilation=3，256→128
  → 拼接为384通道
  → 1×1卷积，384→256 (fuse)
  → 与stem输出残差相加
  → GroupNorm + GELU
  → reshape为[B, C, 256, 36, 36]
  → clip_score_embed_36
```

三个分支分别提供3×3、5×5和7×7等效感受野。不使用模板门控，所有模板确定性参与1×1模板融合。score embedding输出仍为256通道。

项目当前不存在192通道的score embedding主路径；所有相关模块统一使用256通道。

## 11. SAM3 FPN注入encoder feature

SAM3 FPN为图像共享feature，没有类别维度：

```text
sam_fpn_72 [B, 256, 72, 72]
  → bilinear下采样到36×36
  → broadcast到C个类别
  → [B*C, 256, 36, 36]
```

它与下采样后的encoder feature基线拼接：

```text
cat([base_feature_36, sam_fpn_36])
  → [B*C, 512, 36, 36]
  → Conv(512→256, 3×3) + GroupNorm + GELU
  → Conv(256→256, 3×3) + GroupNorm + GELU
  → feature_fpn_delta_36
```

残差注入使用直接scale形式：

```text
feature_36 = base_feature_36
           + feature_fpn_res_scale × feature_fpn_delta_36
```

`feature_fpn_res_scale`是初始值为0的可训练标量。因此当所有refiner残差系数为0时，`feature_36 == base_feature_36`，确保初始化时的恒等路径。

FPN不再注入score_embed。score_embed_36直接等于clip_score_embed_36，保持纯净。

## 12. Encoder refiner

### 12.1 输入与尺度变换

输入：

```text
encoder_features_72 [B, C, 256, 72, 72]
score_embed_36      [B, C, 256, 36, 36]
sam_text_mean       [B, C, 256]
```

SAM3 encoder feature先做双线性下采样得到基线：

```text
encoder_features_72
  → base_feature_36 [B, C, 256, 36, 36]
```

SAM3 FPN与`base_feature_36`拼接后经卷积产生残差更新，加上`feature_fpn_res_scale`调制得到初始`feature_36`。`feature_36`和`score_embed_36`经过默认4层`EncoderRefinerLayer`联合更新。训练时可对每层使用activation checkpoint。

### 12.2 每层顺序（pre-norm + LayerScale）

```text
ClassScoreAttention：
  feature、score和text分别pre-norm
  → attention产生双路更新
  → 独立LayerScale
  → 残差相加

regular WindowScoreAttention：
  feature和score分别pre-norm
  → attention产生双路更新
  → 独立LayerScale
  → 残差相加

shifted WindowScoreAttention：
  feature和score分别pre-norm
  → attention产生双路更新
  → 独立LayerScale
  → 残差相加

feature和score各自执行：
  pre-norm
  → FFN
  → 独立LayerScale
  → 残差相加
```

每层具有八个独立LayerScale标量：

```text
class_feature_scale      class_score_scale
regular_feature_scale    regular_score_scale
shifted_feature_scale    shifted_score_scale
ffn_feature_scale        ffn_score_scale
```

LayerScale默认初始值为0.1。更新形式为：

```text
feature = feature + scale * feature_update
score   = score   + scale * score_update
```

注意：
- q、k和value都来自pre-norm输入。
- attention模块内部不再执行LayerNorm。
- 子层残差后不做post-norm。
- 不再使用 `1 + learnable_scale`，所有残差均为 `state + scale × update`。
- 每层不再有独立的output LayerNorm。
- 四层refiner全部结束后，仅score执行一次最终LayerNorm；feature不再有最终LayerNorm，确保零初始化时feature保持恒等。

### 12.3 ClassScoreAttention

它在每个空间位置上沿类别维度做attention，解决“同一位置更可能属于哪个类别”的竞争与信息交换。

输入由外层完成pre-norm后传入：

```text
q/k input = concat(pre-norm feature, pre-norm sam_text_mean, pre-norm score_embed)
```

三个部分各256维，拼接后为768维。`sam_text_mean`会broadcast到所有36×36空间位置。

value也来自pre-norm输入：

```text
v_feature来自pre-norm feature
v_score来自pre-norm score_embed
```

两路共享同一套类别attention权重，但使用不同的value和输出投影，分别得到`feature_update`与`score_update`。attention模块本身不做残差、LayerNorm和LayerScale，这些操作由外层`EncoderRefinerLayer`统一完成。

### 12.4 WindowScoreAttention

它在每个类别内部做局部空间attention，解决局部区域的结构和边界传播问题。

默认配置：

```text
regular window：window_size=12，shift_size=0
shifted window：window_size=12，shift_size=6
num_heads=8
```

q/k输入为（外层pre-norm后的feature和score）：

```text
concat(pre-norm feature, pre-norm score_embed)
```

两个部分各256维，拼接后为512维。value也来自pre-norm输入，分为feature和score两路，并共享attention权重。

shifted window通过循环平移和attention mask连接相邻常规窗口，同时阻止平移后本不相邻的位置错误混合。

### 12.5 相对位置偏置

每个WindowScoreAttention维护：

```text
relative_position_bias_table
```

当窗口边长`ws=12`时，一个轴上的相对位移范围为`-11 ... 11`，共有23种；二维组合共有`23 × 23 = 529`种。因此8个attention head对应的表形状为：

```text
[529, 8]
```

attention score由query-key点积、缩放、相对位置偏置以及shift mask共同决定，随后经过softmax变成权重。

## 13. 36×36更新量上采样到72×72

上采样采用纯更新量残差方式，不含任何卷积、归一化或激活：

```text
feature_delta_36 = feature_36 - base_feature_36
  → bilinear上采样到72×72
  → feature_delta_72 [B, C, 256, 72, 72]

refined_encoder_features_72 = encoder_features_72 + feature_delta_72
```

这里`base_feature_36`是进入refiner前的encoder feature 36基线（即`encoder_features_72`下采样后的张量），不是FPN注入后的feature。以`base_feature_36`为差分基线，保证FPN的直接残差更新也会被包含在`feature_delta_36`中并传递到72×72输出。

当所有refiner残差系数为0时：
- `feature_36 == base_feature_36`
- `feature_delta_36 == 0`
- `feature_delta_72 == 0`
- `refined_encoder_features_72 == encoder_features_72`

即初始化时refiner输出严格等于原始encoder feature。

## 14. 写回SAM3并生成最终logits

refiner输出按最初的类别chunk重新切分。对每个chunk：

1. clone原始`encoder_hidden_states`。
2. 只替换第一个空间层对应的图像token区域。
3. prompt token、prompt mask、位置编码和其他结构保持原样。
4. 调用冻结的SAM3 segmentation head。
5. 把chunk输出整理为`[B, C_chunk, H_out, W_out]`。

最后沿类别维拼接：

```text
final_logits [B, C, H_out, W_out]
```

segmentation head参数虽然冻结，但它的前向不能整体放进`torch.no_grad()`，因为loss仍需通过segmentation head的运算返回到`refined_encoder_features_72`和前面的可训练模块。

## 15. 训练loss

训练只监督`final_logits`：

```text
total_loss = final_bce_weight × loss_final_bce
           + final_dice_weight × loss_final_dice
```

其中：

* `loss_final_bce`是按“图像—类别”二元mask计算的BCE。
* `loss_final_dice`是只对当前图像中实际出现类别计算的Dice loss。
* 当前`final_bce_weight=1.0`。
* 当前`final_dice_weight=0.0`，所以默认训练实际只使用BCE。

### 15.1 二值目标

对每个类别通道分别构造：

```text
target[b, c, y, x] = 1，当label_map[b, y, x]等于类别c
target[b, c, y, x] = 0，其他情况
```

`b`表示图像索引，`c`表示类别索引，`y/x`表示像素坐标。

### 15.2 present与absent类别对

如果某类别出现在当前图像的非ignore区域，它是present pair；否则是absent pair。

当前基础配置：

```text
bce_absent_class_weight=0.05
bce_valid_pixel_weight=1.0
bce_ignore_pixel_weight=0.05
```

实际权重规则：

| 图像—类别关系 | valid像素 | ignore像素 | pair整体权重 |
| ------- | ------: | -------: | -------: |
| present |     1.0 |     0.05 |      1.0 |
| absent  |     1.0 |        0 |     0.05 |

这意味着：

* present类别在ignore区域仍受到轻微的泄漏抑制。
* absent类别只在valid区域学习“不要预测该类别”。
* absent类别整体贡献被降到0.05，避免负类数量过多主导训练。

## 16. 推理、背景类与评估

### 16.1 基础预测与逐类别相对过滤

```text
final_logits
  → sigmoid
  → raw_final_score_map（未过滤的原始sigmoid分数，永久保留）
  → 每张图、每个类别独立做空间min-max缩放得到relative_score
  → relative_score低于class_relative_prob_thd的位置在原始分数上置0
  → final_score_map（过滤后分数，保留位置仍是原始sigmoid数值）
  → argmax(final_score_map, class_dimension) → final_pred
```

`sigmoid`把每个类别的独立logit变为0到1之间的分数。相对过滤由adapter配置控制：

```python
adapter_cfg=dict(
    class_relative_prob_thd=0.5,  # None表示关闭逐类别相对过滤
    class_relative_eps=1e-6,      # 判定掩码数值跨度是否小到无法稳定缩放
)
```

规则：

* `raw_final_score_map`与`final_score_map`是两套不同输出：前者是未过滤的sigmoid分数，后者是相对低分位置已置0、保留位置仍为原始数值的分数。缩放后的relative_score只用于生成保留掩码，绝不作为输出分数。
* min/max只在每个`[b, c]`掩码的`H×W`空间内计算，不跨batch、不跨类别。
* 掩码数值跨度不超过`class_relative_eps`时视为近似常数，整张保留，不做过滤。
* 不引入softmax类别竞争；`final_pred`始终由过滤后的`final_score_map`做argmax得到。
* 普通推理与TTA共用`SemanticSegAdapter.build_infer_score_outputs()`统一生成这三个输出。
* 训练的`output_mode="final"`路径只返回`final_logits`，不经过任何阈值处理。

### 16.2 背景配置

每个数据集通过：

```python
background_cfg=dict(
    enabled=False,
    class_id=0,
    class_name=None,
    exclude_from_forward=False,
)
```

描述背景类。

| 模式                                         | 行为                                   |
| ------------------------------------------ | ------------------------------------ |
| `enabled=False`                            | 没有显式背景，普通argmax，不做低置信度背景过滤。          |
| `enabled=True, exclude_from_forward=False` | 背景参与前向；分数小于等于阈值的像素设为背景id。            |
| `enabled=True, exclude_from_forward=True`  | 背景不进入模型类别列表；评估时先恢复类别id，再把分数小于等于阈值的像素设为背景。 |

LoveDA验证配置使用第三种模式：背景id为0，但背景不参与模型前向。

### 16.3 标签流水线

```text
raw label
  → reduce_zero_label（可选）
  → eval_label_map（保留评估类别空间）
  → background exclusion（可选）
  → label_map（训练loss使用）
```

`eval_label_map`用于mIoU和可视化；`label_map`用于训练。

### 16.4 共享后处理

`engine/evaluator.py`中的`build_eval_semantic_pred()`同时服务evaluator和visualizer，避免指标与图片使用不同预测规则。

它优先读取`final_score_map`（逐类别相对过滤后的原始绝对分数），缺失时回退到`final_pred`，最终返回：

```text
pred_eval
eval_num_classes
eval_class_names
```

`eval_cfg.prob_thd`（当前0.1）是全局绝对分数阈值，作用于过滤后`final_score_map`的类别最大值，把小于等于阈值的像素设为背景id；使用`<=`是为了让逐类别相对过滤置0的像素在`prob_thd=0.0`时也能直接归为背景。它只在启用背景类时生效，与adapter的逐类别相对阈值职责分离。

评估指标包括：

```text
mIoU
mAcc
pixel accuracy / aAcc
每个类别的IoU和accuracy
```

## 17. 训练与验证生命周期

一次训练step：

```text
batch移到device
  → optimizer.zero_grad(set_to_none=True)
  → build_encoder_refiner_cache
  → run_encoder_refiner_from_cache
  → adapter输出final_logits
  → criterion计算loss
  → AMP scaler backward
  → unscale
  → 全模型梯度裁剪（默认max norm 0.01）
  → optimizer step
  → scheduler step
```

验证函数使用`@torch.no_grad()`，支持翻转TTA（仅`scale=1.0`，因当前refiner固定72×72/36×36尺度）。TTA翻转时同步变换`img_batch`和`raw_images`，确保SAM3和RemoteCLIP空间一致。

逐类别相对过滤是非线性操作，TTA合并顺序必须是“先平均原始分数，再统一过滤一次”：

```text
每个视图输出raw_final_score_map
  → 反翻转恢复空间方向
  → 多视图平均raw_final_score_map（跳过各视图已过滤的final_score_map，不参与平均）
  → adapter.build_infer_score_outputs()
  → 重新生成final_score_map与final_pred
```

TTA未启用时直接执行`model(batch)`，由adapter的infer分支完成一次过滤。验证结束后模型重新进入训练模式，但冻结的SAM3模块以及RemoteCLIP编码器仍被强制保持`eval()`。

当前TTA配置：

```python
tta_cfg = dict(
    enabled=False,
    scales=[1.0],               # 仅1.0，多尺度在支持动态空间尺寸前禁用
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)
```

非1.0尺度会在进入view循环前直接抛出`ValueError`。

## 18. 验证阶段RemoteCLIP缓存

RemoteCLIP模板文本feature的缓存策略：

| 场景 | 缓存行为 |
|---|---|
| RemoteCLIP文本编码器完全冻结 | 允许长期缓存detach后的模板feature |
| 训练 + 文本编码器可训练 | 禁止缓存，每个step重新编码并保留计算图 |
| 验证 + 文本编码器可训练 | 本次验证内允许缓存detach后的模板feature |

每次调用`model.train(mode)`切换训练/验证模式时，会自动清除RemoteCLIP模板缓存。这样：
1. 进入验证时清除旧cache。
2. 验证第一张图重新编码当前权重对应的模板feature并缓存。
3. 后续验证图复用该cache。
4. 验证结束调用`model.train()`时再次清除。

## 19. 验证阶段图像梯度保护

`OpenCLIPImageEncoder.encode_image_with_intermediate()`中判断是否建立梯度时同时检查`torch.is_grad_enabled()`、`self.enable_grad`和`self.has_trainable_params()`。验证阶段外层`@torch.no_grad()`会使`torch.is_grad_enabled()`返回False，因此不会错误开启RemoteCLIP图像计算图。

`Sam3Image._build_clip_image_cache()`也使用`keep_image_graph = torch.is_grad_enabled() and image_encoder_trainable`判断是否需要保留计算图。

## 20. Optimizer checkpoint恢复后的K参数保护

融合QKV参数（`in_proj_weight`/`in_proj_bias`）受Q/V mask保护，标记为`_ovrs_disable_weight_decay=True`。新建optimizer时该标记使对应参数组`weight_decay=0.0`。

`optimizer.load_state_dict()`从checkpoint恢复时会覆盖参数组超参数。为防御旧checkpoint中可能保存的`weight_decay≠0`，`CheckpointManager.load()`在调用`load_state_dict()`后立即执行`enforce_optimizer_param_group_invariants()`，重新强制受保护参数组的`weight_decay=0.0`。

## 21. 全ignore batch的optimizer step跳过

`SemanticCriterion`使用`valid_mask.sum()`（非ignore像素数）作为`num_valid`。当batch中所有像素都是`ignore_index`时，`num_valid=0`，criterion返回零loss。`Trainer.train_step()`检查`num_valid>0`才执行backward和optimizer step，不会在全ignore batch上更新参数。

## 22. Debug输出

`return_debug=True`时可获得：

| key                           | 形状                            | 用途                                 |
| ----------------------------- | ----------------------------- | ---------------------------------- |
| `final_logits`                | `[B, C, H_out, W_out]`        | 最终分割logits。                        |
| `raw_final_score_map`         | `[B, C, H_out, W_out]`        | 未过滤的sigmoid分数。                     |
| `final_score_map`             | `[B, C, H_out, W_out]`        | 逐类别相对过滤后分数，保留位置仍是原始sigmoid数值。      |
| `encoder_features`            | `[B, C, 256, 72, 72]`         | refiner之前的SAM3 encoder feature。    |
| `refined_encoder_features`    | `[B, C, 256, 72, 72]`         | 写回前的refined feature。               |
| `refiner_features_36`         | `[B, C, 256, 36, 36]`         | refiner最终图像feature。                |
| `score_embed_36`              | `[B, C, 256, 36, 36]`         | FPN注入并经过refiner更新的score embedding。 |
| `clip_score_embed_36`         | `[B, C, 256, 36, 36]`         | 进入FPN融合前的CLIP score embedding。     |
| `clip_score_maps`             | `[B, C, 32, 36, 36]`          | 原始32模板相似度图。                        |
| `template_clip_text_features` | `[C, 32, 768]`                | RemoteCLIP模板文本feature。             |
| `clip_mid_features`           | `List[[B, D_native, 36, 36]]` | RemoteCLIP中间层debug feature。        |

Debug分支中的detach只用于返回诊断副本，不应提前作用于参与最终loss的主路径。

## 19. 主要文件职责

| 文件                                    | 职责                                                            |
| ------------------------------------- | ------------------------------------------------------------- |
| `tools/train.py`                      | 读取配置、构建组件、启动训练或验证。                                            |
| `config_dataclasses.py`               | 模型、RemoteCLIP、refiner、loss和trainer配置定义。                       |
| `model_builder.py`                    | 构建SAM3/RemoteCLIP/refiner，加载权重，应用冻结与微调策略。                     |
| `configs/_base_/model/ovrs_sam3.py`   | 模型、RemoteCLIP、refiner、freeze和loss的基础配置。                        |
| `configs/_base_/dataloader/`          | 训练和评估dataloader的公共transform与collate。                           |
| `configs/datasets/train/`             | 按数据集拆分的训练集特定参数（路径、classes、background）。                         |
| `configs/datasets/eval/`              | 按数据集拆分的评估集特定参数。                                           |
| `configs/train/`                      | 训练入口，组合模型、优化器、schedule、数据集。                                  |
| `configs/test/`                       | 评估入口，组合模型和评估数据集，`eval_mode=True`。                            |
| `engine/optimizer_builder.py`         | 创建参数组、学习率、weight decay和scheduler。                             |
| `models/sam3_image.py`                | 主流程协调、类别chunk、cache构建、refiner调用、feature写回和logits拼接。           |
| `models/openclip_image_encoder.py`    | 504×504 RemoteCLIP dense图像编码与last-block value branch。         |
| `models/openclip_text_encoder.py`     | RemoteCLIP通用冻结文本接口，以及支持梯度、micro-batch和checkpoint的模板编码接口。      |
| `models/score_embeddings.py`          | 32模板相似度图、RemoteCLIP模板缓存和256通道score embedding。                 |
| `models/encoder_refiner.py`           | FPN注入、36×36 refiner主流程和72×72上采样融合。                            |
| `models/encoder_refiner_attention.py` | ClassScoreAttention、WindowScoreAttention、EncoderRefinerLayer。 |
| `models/segmentor.py`                 | 模型外层wrapper和训练/推理模式管理。                                        |
| `models/adapters/semantic_adapter.py` | 训练只返回logits；推理生成raw/过滤后score map和最终类别图，逐类别相对过滤在此实现。                              |
| `losses/semantic_criterion.py`        | present/absent加权BCE与可选Dice。                                   |
| `data/dataset.py`                     | 图像、标签、类别名称和背景配置处理。                                            |
| `data/collate.py`                     | batch padding、共享类别顺序、label map与metadata组装。                    |
| `engine/trainer.py`                   | AMP训练、梯度裁剪、checkpoint、验证和hook生命周期。                            |
| `engine/evaluator.py`                 | 背景后处理、TTA合并和语义分割指标。                                           |
| `engine/visualization.py`             | 保存预测、GT、score heatmap等可视化结果。                                  |
## 20. 关键实现不变量

修改项目时必须保持：

1. 同一batch中的所有样本共享完全相同的前向类别名称和顺序。
2. 类别chunk拼接后必须按原顺序完整覆盖全部前向类别。
3. RemoteCLIP图像与文本投影维度必须相同，当前为768。
4. RemoteCLIP dense grid必须是36×36。
5. SAM3 encoder主feature必须是72×72。
6. score embedding和SAM3 hidden dimension当前都为256。
7. 训练阶段可训练RemoteCLIP文本feature不得跨optimizer step缓存；验证阶段允许单次验证内缓存。
8. 验证阶段不得为RemoteCLIP图像分支重新开启autograd（尊重外层`torch.no_grad()`）。
9. TTA翻转时必须同步变换`img_batch`和`raw_images`。
10. `clip_mid_features`当前不参与主路径。
11. segmentation head虽然冻结，但必须允许梯度穿过其运算返回refiner。
12. 恢复checkpoint optimizer状态后必须重新强制K参数weight decay约束。
11. attention模式下融合QKV的K区域既不能有梯度，也不能受到weight decay。
12. evaluator与visualizer必须共用背景后处理逻辑。
13. score embedding的三个并行卷积分支必须保持36×36空间尺寸。
14. refiner所有attention和FFN统一采用pre-norm。
15. 每个refiner子层的feature与score更新使用独立LayerScale。
16. 每层内部不允许恢复post-norm。
17. 整个refiner只在四层全部结束后对score执行一次最终LayerNorm；feature不再有最终LayerNorm。
18. 推理中`raw_final_score_map`不得被原地修改；逐类别过滤必须用非原地操作生成`final_score_map`。
19. 相对过滤保留区域必须保留原始sigmoid分数，不得用缩放后的relative_score替换。
20. TTA必须先平均`raw_final_score_map`再统一过滤一次，不能平均各视图已过滤的`final_score_map`。
21. 所有refiner残差系数（feature_fpn_res_scale和八个LayerScale）初始值必须为0，确保初始化时refiner输出严格等于原始encoder feature。

## 21. 已知限制与当前非目标

* 当前只支持semantic task mode；hybrid模式会抛出`NotImplementedError`。
* 一个batch不能混用不同类别列表或类别顺序。
* 当前语义前向不支持非空几何prompt。
* RemoteCLIP图像输入固定为504×504和36×36 grid。
* SAM3 encoder/refiner尺度固定为72×72与36×36。
* prompt模板数量固定为32。
* `clip_mid_features`已提取但未进入主路径。
* 默认Dice权重为0，当前主要研究对象是加权BCE下的refiner与RemoteCLIP微调。
* 默认训练是iSAID、验证是LoveDA，属于跨数据集开放词汇评估设置。

## 22. 配置结构说明

### 22.1 设计原则

配置文件按职责分层：

- `_base_/model/`：模型、RemoteCLIP、refiner、freeze和loss定义。
- `_base_/optimizer/`：优化器与参数组倍率。
- `_base_/schedule/`：训练迭代数与scheduler。
- `_base_/dataloader/`：训练和评估的公共transform与collate。
- `datasets/train/`和`datasets/eval/`：按数据集和角色拆分，只包含路径、classes、background等数据集元数据。
- `train/`：训练入口，组合模型、优化器、schedule和训练/评估数据集。
- `test/`：评估入口，组合模型和评估数据集，`eval_mode=True`。

### 22.2 评估用法

```bash
python tools/train.py configs/test/loveda.py \
  --eval-only \
  --load-model-from /path/to/checkpoint.pth
```

切换数据集只需替换配置：

```bash
python tools/train.py configs/test/potsdam.py \
  --eval-only \
  --load-model-from /path/to/checkpoint.pth
```

限制少量batch：

```bash
python tools/train.py configs/test/loveda.py \
  --eval-only \
  --load-model-from /path/to/checkpoint.pth \
  --cfg-options train_cfg.val_max_iters=50
```

开启TTA：

```bash
--cfg-options tta_cfg.enabled=true
```

注意：
- eval-only当前使用`val_dataloader`。
- 正式测试默认`val_max_iters=None`，遍历完整验证集。
- 推荐使用`--load-model-from`进行纯模型权重测试。
- 测试输出按`work_dirs/test/<dataset>`分目录保存。

## 23. 语法检查

```bash
python -m py_compile config_dataclasses.py
python -m py_compile model_builder.py
python -m py_compile engine/optimizer_builder.py
python -m py_compile engine/checkpoint.py
python -m py_compile engine/evaluator.py
python -m py_compile models/openclip_image_encoder.py
python -m py_compile models/openclip_text_encoder.py
python -m py_compile models/score_embeddings.py
python -m py_compile models/encoder_refiner_attention.py
python -m py_compile models/encoder_refiner.py
python -m py_compile models/sam3_image.py
python -m py_compile models/segmentor.py
python -m py_compile losses/semantic_criterion.py
```

配置文件语法检查：

```bash
python -m py_compile $(find configs -name "*.py" -type f)
```

```bash
git diff --check
```

## 24. 修改本文的规则

出现以下变化时必须同步更新本文：

* 任一主要张量的通道数或空间尺寸变化。
* RemoteCLIP模板数量、输入尺寸或微调模式变化。
* 新增或删除feature融合路径。
* refiner attention顺序、残差方式或value分支变化。
* loss权重逻辑或背景类后处理变化。
* 冻结模块、学习率倍率或缓存规则变化。
* `clip_mid_features`开始进入主路径。
* 新增hybrid或几何prompt训练模式。

本文应描述实际执行路径，而不是仅描述最初设想。发生冲突时，以经过测试的当前代码为准，并立即修正文档。
