# 批量训练与独立验证

在仓库根目录执行。两个入口分别负责训练和验证，训练队列不会自动跑八个目标数据集。训练期间仍每 1000 步验证 LoveDA，保存曲线、`best.pth`、`latest.pth` 和周期权重，供手动检查。

## 公共训练设置

`configs/experiments/_base_/dlrsd_loveda.py` 统一继承 DLRSD→LoveDA 协议、模型、损失、数据增强、优化器及 `full_10k.py`。主入口与十个消融入口只覆盖差异项。

| 设置 | 值 |
| --- | --- |
| Batch size / optimizer steps | 4 / 10000 |
| AdamW 基础学习率 | 2e-6 |
| RemoteCLIP 学习率倍率 / 预热目标学习率 | 0.01 / 2e-8 |
| 线性预热 | 500 步，从目标学习率的 0.1 倍开始 |
| 相对余弦衰减 | 9500 步，最低为各参数组目标学习率的 0.01 倍 |
| 最终学习率：主模块 / RemoteCLIP | 2e-8 / 2e-10 |
| 保存 / LoveDA 验证间隔 | 每 1000 步；保留 10 份周期权重 |
| AMP / GradClip / W&B | 开启 / 0.01 / offline |

新训练使用 `work_dirs/paper_batch4_10k/`。改变训练预算或结构后使用新根目录；旧 20000 步实验不作恢复训练。

## 实验集合

| variant_id | 差异 | 独立验证集合 |
| --- | --- | --- |
| full | 完整模型；图文均 attention 微调 | 八数据集 |
| simple_adapter | 提示条件 72×72 特征经共享残差 1×1→GN→GELU→3×3→GN，移除 RemoteCLIP、聚合、SDF | 八数据集 |
| diagonal_cross_class | 两次类间注意力均屏蔽展开提示轴的非对角项，别名也不交换 | 八数据集 |
| no_intra_class | 绕过两条流的全部类内块 | LoveDA、Potsdam、UAVid |
| no_score_prior | 初始语义状态为零，保留 RemoteCLIP 中间层视觉指导 | LoveDA、Potsdam、UAVid |
| simple_fusion | 聚合特征上采样后与原始提示条件特征、FPN 拼接，768→256→256 卷积 | LoveDA、Potsdam、UAVid |
| single_orientation | SDF 仅 0°，保留支路、投影、归一化与残差 | LoveDA、Potsdam、UAVid |
| no_distillation | 蒸馏权重为 0 | LoveDA、Potsdam、UAVid |
| freeze_both | RemoteCLIP 图文均冻结 | LoveDA、Potsdam、UAVid |
| freeze_text | 文本冻结，图像 attention 微调 | LoveDA、Potsdam、UAVid |
| freeze_image | 图像冻结，文本 attention 微调 | LoveDA、Potsdam、UAVid |
| semantic_only | 原始 SAM3 语义流程；只验证 | 八数据集 |
| semantic_presence | 原始 SAM3 完整语义及实例流程；语义概率乘该提示 presence 概率一次，再合并别名；实例掩码不融合 | 八数据集 |

两个原始 SAM3 基线加载同一份 `weights/sam3.pt`，不构造 RemoteCLIP/适配模块，不生成训练权重，结果标为 training-free、step 0。presence 路径恢复原始六层、200-query 解码器及 instance mask head；缺少必需预训练参数会报错，不能用随机参数代替原始基线。

八数据集配置来自 `configs/test/{dlrsd,isaid,loveda,potsdam,vaihingen,uavid,udd,vdd}.py`。UAVid 使用当前 master 的七类合并车辆协议，包括 background 提示。完整模型只训练一次，其权重与结果在主表、结构消融表和冻结表复用。没有通用 CLIP 消融；历史 0.25 蒸馏/4K 调试入口不进入当前论文队列。

## 训练

```bash
# 只查看计划，不创建目录，不导入 GPU 模型
python tools/train_experiments.py configs/suites/paper_train.py --dry-run

# 默认单 GPU 顺序完成 11 项训练
python tools/train_experiments.py configs/suites/paper_train.py

# 中断后：跳过配置一致且完整的任务，未完成任务从 latest.pth 恢复
python tools/train_experiments.py configs/suites/paper_train.py --resume

# 可选：一张卡同时只跑一个独立实验
python tools/train_experiments.py configs/suites/paper_train.py --devices cuda:0 cuda:1

# 可选：仅跑指定变体
python tools/train_experiments.py configs/suites/paper_train.py --only full freeze_text
```

每个子任务使用独立进程，失败会保留日志并继续其他任务；汇总存在失败时入口返回非零退出码。运行中断使用 `--resume`；改变配置不得覆盖旧目录。数据、预训练文件路径可在公共配置或各数据集配置修改，修改后先 dry-run 检查。

目录为 `<root>/<variant>/seed_<seed>/train/`，包含原有 `config.json`、metrics、checkpoint、标准结果、离线 W&B，以及 `console.log`、`suite_status.json`。训练入口维护根目录 `checkpoints.json` 与 `train_summary.json`，并首次生成 `validation_template.py`；恢复训练不会覆盖用户已经填写的模板。

## 手选权重后验证

编辑 `configs/suites/paper_validate.py` 的 `selections`，或编辑生成的 `<root>/validation_template.py`。例如：

```python
suite = dict(
    # 保留文件中原有的公共字段、数据集组和其他 selections。
    selections=[
        dict(variant="full", seed=42,
             checkpoint="work_dirs/paper_batch4_10k/full/seed_42/train/iter_0005000.pth"),
        # 每个变体填写一个 checkpoint；None 表示尚未选择，不自动选 best/latest。
    ],
)
```

上例是字段片段；应在完整模板内编辑。每个 seed/变体只能填一份权重，验证会在该项对应的所有数据集上复用这份权重。可指定 `best.pth`、`latest.pth` 或任一周期权重；权重旁必须有实际训练保存的 `config.json`。

```bash
python tools/validate_experiments.py configs/suites/paper_validate.py --dry-run
python tools/validate_experiments.py configs/suites/paper_validate.py
python tools/validate_experiments.py configs/suites/paper_validate.py --resume

# 若编辑的是生成模板，则将上面的配置参数换成：
python tools/validate_experiments.py work_dirs/paper_batch4_10k/validation_template.py
```

验证从训练快照恢复架构及损失配置，仅替换目标数据加载器、评测协议与输出目录；权重严格加载，并核对 checkpoint 与训练快照的指纹。不会把消融权重套进默认完整模型。改选权重时修改 `validation_name`，保存为新一批验证，防止覆盖或混合不同权重的成绩。

输出为 `<root>/validation/<validation_name>/<variant>/seed_<seed>/<dataset>/`，包括实际评测配置、控制台日志、选择的权重路径和 SHA256、标准 `results/*.json`。根目录另有 `validate_summary.json`、`metrics_summary.json/csv`；指标仍为 fraction 单位，不能再次除以 100。现有结果收集和论文导出工具继续读取标准 JSON，不会自动把汇总 CSV 填进论文。

保留当前 LoveDA 开发验证策略：手选权重不等于没有使用目标标签。论文需要如实区分开发域与独立目标域。默认 seed 42 是运行设置，不是已经完成的正式实验。效率表仍需在统一硬件与测量条件下另行采集真实测量，批量验证不写估算耗时或参数数值。

## 验证范围

CPU 回归覆盖配置、11 项计划、手选权重与八域配置恢复、恢复/跳过/失败隔离、注意力屏蔽、零初始化视觉梯度、adapter/fusion 梯度、presence 单次调制及原始实例解码器/掩码头。合成检查不会进入正式结果目录。尚未在用户数据、正式预训练权重和 GPU 上执行完整训练/评测。
