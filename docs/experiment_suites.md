# 论文实验：单项训练与指定权重验证

以下命令均在仓库根目录执行。先安装项目依赖，准备配置指定的数据与预训练权重：weights/sam3.pt、weights/RemoteCLIP-ViT-L-14.pt 和 assets/bpe_simple_vocab_16e6.txt.gz。无 RemoteCLIP 指导变体及两个原始基线不需要加载 RemoteCLIP 权重。

## 实验清单

| variant ID | 配置（configs/experiments/ 下） | 与 Full 的差异 |
| --- | --- | --- |
| full | main/dlrsd_loveda_full.py | 当前完整模型 |
| self_only | ablation/self_only.py | 展开提示轴严格对角屏蔽，连同类同义提示也不互看 |
| no_remoteclip_guidance | ablation/no_remoteclip_guidance.py | 两类注意力不拼接 score embedding，不构建 RemoteCLIP 或 score embedding；保留 SAM3 FPN |
| no_intra | ablation/no_intra.py | 不构建／运行类内注意力，保留类间与 SDF |
| single_scale | ablation/single_scale.py | 36×36 上一次 12×12 普通窗口，无尺度融合参数 |
| unshared_windows | ablation/unshared_windows.py | 三尺度窗口参数分别学习，保留 Softmax 融合 |
| frozen_remoteclip | ablation/frozen_remoteclip.py | 图像和文本两端冻结 |
| image_only | ablation/image_only.py | 仅图像 attention 微调 |
| text_only | ablation/text_only.py | 仅文本 attention 微调 |
| no_distillation | ablation/no_distillation.py | 蒸馏权重为零并跳过教师解码 |

四种图文组合中的共同微调即 Full。两个对比基线 semantic_only、semantic_presence 使用 configs/baselines/ 中的对应配置，仅验证，不加入训练队列。Semantic × Presence 在合并同义提示前，对每提示语义概率乘一次 presence，不融合实例掩码。所有外部论文方法继续引用文献成绩，不安排本地训练。

## 只训练一个消融

```bash
python tools/train_experiments.py configs/suites/paper_train.py \
  --only self_only --seed 42 --devices cuda:0
```

将 self_only 替换为表中任一训练 ID 即可。也可以直接运行对应配置：

```bash
python tools/train.py configs/experiments/ablation/self_only.py --device cuda:0
```

训练 suite 的 --seed 选择一个种子；默认种子为 42。Full 与全部消融统一 DLRSD 训练、LoveDA 七类验证、batch 4、10000步、基础LR 1e-4、RemoteCLIP倍率0.01、500步预热、9500步相对余弦与 W&B offline。默认输出 work_dirs/reca_sam3_paper/<variant>/seed_<seed>/train/，保存 best.pth、latest.pth 及 config.json。

恢复同一次训练：在 suite 命令末尾加 --resume。配置指纹变化时使用新实验目录；不将旧结构的训练状态迁移到新版本。--devices cuda:0 cuda:1 为不同实验分配设备，不是一个实验的数据并行。

## 在一个数据集加载指定权重

```bash
python tools/validate_experiments.py configs/suites/paper_validate.py \
  --only self_only --seed 42 \
  --checkpoint work_dirs/reca_sam3_paper/self_only/seed_42/train/best.pth \
  --datasets loveda --devices cuda:0
```

这条命令只创建一个验证任务，不执行训练，也不验证其他变体或数据集。权重旁必须保留训练生成的 config.json；验证从它恢复真实结构、实验 ID 与种子，校验指纹后严格加载模型参数。不能用 Full 的验证结构直接加载结构消融权重。

验证目录为 work_dirs/reca_sam3_paper/validation/<validation_name>/<variant>/seed_<seed>/<target>/。切换另一份权重时加 --validation-name iter4000 等不同名称，避免覆盖旧结果；相同任务中断或重复跳过使用 --resume。--checkpoint 必须配合恰好一个 --only；原始 SAM3 基线也可用它指定原始 SAM3 权重，无需训练 config.json。

```bash
python tools/validate_experiments.py configs/suites/paper_validate.py \
  --only semantic_only --checkpoint weights/sam3.pt \
  --datasets loveda --devices cuda:0
```

## iSAID 两种协议

- isaid_15_ignbg：读取原始 0..15 标签；0转255忽略，1..15减1，统计十五个前景类。
- isaid_16_bg：相同图像／标签文件，0保留为 background，1..15不平移，255仍忽略，统计十六类。background 参与前向；prediction_prob_threshold=0.5，低于阈值置零后按 background_class_id=0 回填，background_prob_threshold=0.0。规则不读取 GT。

两个配置分别位于 configs/test/isaid_15_ignbg.py 与 isaid_16_bg.py，evaluation.dataset 均为 isaid，协议 ID 和输出目录不同。用同一权重同时验证：

```bash
python tools/validate_experiments.py configs/suites/paper_validate.py \
  --only full --seed 42 \
  --checkpoint work_dirs/reca_sam3_paper/full/seed_42/train/best.pth \
  --datasets isaid_15_ignbg isaid_16_bg --devices cuda:0
```

主比较目标：dlrsd、isaid_15_ignbg、isaid_16_bg、loveda、potsdam、vaihingen、uavid、udd、vdd。八个数据集对应九个任务；UAVid本地始终七类。其余目标也可使用 configs/test/ 中实际文件名。每个目标配置的 postprocess 阈值覆盖训练期间验证阈值，不能逐消融优化。

## 批量执行与结果汇总

```bash
# 预览全部十个训练任务，不读数据、不创建训练目录
python tools/train_experiments.py configs/suites/paper_train.py --dry-run
# 一次执行全部训练变体
python tools/train_experiments.py configs/suites/paper_train.py
# 填写 paper_validate.py selections 后批量验证
python tools/validate_experiments.py configs/suites/paper_validate.py
# 汇总真实独立评测记录
python tools/collect_experiments.py work_dirs/reca_sam3_paper/validation \
  --output paper/generated/data/results_long.csv
```

Full 和两个基线默认评测九个主比较目标，其他九个消融默认评测 LoveDA、Potsdam、UAVid；单种子全部选定后共54个验证任务。所有变体用 LoveDA 选择一份权重，再用于其全部目标，不能按目标数据集选最优 checkpoint。

--dry-run 同样适用于验证命令；训练模型的指定权重及 config.json 仍须存在，才能检查结构。它只生成任务预览，不能代替真实模型运行。标准结果保留真实 dataset、protocol ID、种子、checkpoint哈希、完整性和指标；metrics_summary.csv 额外保留评测目标名，避免两个 iSAID 协议混淆。CPU合成检查不产生论文成绩。
