# 当前论文评测与组表协议（2026-11-18）

## 数据角色与模型选择

固定八列：DLRSD、iSAID、LoveDA、Potsdam、Vaihingen、UAVid、UDD5、VDD。当前使用OVRSISBench发布包的DLRSD官方5601 train/1401 val划分，17类、无背景。LoveDA val用于开发与checkpoint选择，不是独立目标域；其余六域的独立性以未用于模型/超参数开发为前提。每变体/seed选一份权重，在所有数据集复用，不逐目标挑最高分。

本地类别数17/15/7/6/6/7/5/7。UAVid合并moving/static car为car=5，human=6，使用prepared-uavid7-v2和uavid_7class_merged_car_val_resize1008_v2，旧八类结果不混用。LoveDA背景参与评测、不作前向提示；Potsdam/Vaihingen的clutter和UAVid/UDD5/VDD背景参与前向与评测；iSAID背景忽略。LoveDA实际reduce_zero_label=True，正式运行前核对原始值域、转换与逐类union。通道、裁块及视频划分仍按数据配置和真实样本清单核实，不因同名认定外部协议matched。

主指标mIoU，另存mAcc/逐类IoU。LoveDA只用七类含背景。未产生的结果显示--，不计算混合协议综合均值。跨数据集与严格未见类别的主张分开。

## 当前执行集合

公共预算：batch4、10000步、AdamW基础LR2e-6、RemoteCLIP倍率0.01、500步预热和9500步相对余弦；每1000步保存/验证LoveDA，W&B offline。12个训练入口包含Full、adapter、对角类间、encoder_only_update、无类内、无初始先验、简单融合、单方向、无蒸馏和三种冻结变体。原始Semantic-only与Semantic×Presence仅验证，共用原始SAM3权重，step0，无DLRSD适配。

encoder_only_update保留初始语义状态的Q/K与梯度、普通/移位特征窗口，跳过语义更新及其专用中间层指导；无初始先验变体仍保留RemoteCLIP中间层指导。两者含义不同。训练与验证命令、完整入口和checkpoint格式见[../../docs/experiment_suites.md](../../docs/experiment_suites.md)。当前seed42不表示已完成正式运行；正式seed集合仍待锁定。

## 四张正文表

| 文件 | 内容 |
| --- | --- |
| main_rskt8 | 两个原始SAM3基线、adapter、Full及七行文献；Adaptation data只指额外适配 |
| ablation_core | Full及既定聚合、双流、融合、监督变体，包含Encoder feature updates only |
| ablation_freeze | 图文四种冻结/attention tuning组合及实测可训练参数；Full行共享权重 |
| efficiency | Semantic-only、Semantic×Presence、adapter、Full、单方向SDF的真实端到端开销 |

表内不显示实验编号；无T04、通用CLIP消融、预测过滤表或Table S1/S2。不增加新实验。本次正文修订不改四表、实验章节或选择器。

## 结果溯源与导出

标准结果与config.json、metrics.jsonl、eval_id、checkpoint路径/sha256关联。显式提供eval_ids、expected_seeds、checkpoint_sha256_by_seed、eval_protocol_id、eval_split；不挑最大/最新记录，同seed多权重不平均。Full在主表/核心/冻结表复用相同选择，同变体跨目标域保持权重及seed集合一致。单seed不报标准差，多seed用样本标准差ddof=1。训练变体training_dataset为dlrsd，原始基线为none。

效率统一硬件、精度、输入、batch、提示数、chunk、缓存、预热与同步。presence基线包含语义和实例流程；Full仅学生推理。批量验证不自动生成效率测量。正式组表要求必需结果完整且protocol_locked=true；draft只允许计划缺项，不允许非法值或身份冲突。详见[../scripts/README.md](../scripts/README.md)。

## 文献及待补证据

来源见[comparison_audit_20261117.md](comparison_audit_20261117.md)和[literature_fill_audit.md](literature_fill_audit.md)。GSNet官方LD50K与RSKT报告iSAID分开；原报告与第三方复现分开。保留CSV原PDF指纹和protocol_status，七行文献不按统一条件排名。四个LoveDA单源结果仍unresolved，按既有许可标记展示，不升级为matched。

所有待实验主张、图计划、数据和参考文献缺口集中在[manuscript_status.md](manuscript_status.md)。
