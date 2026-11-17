
# 当前论文评测与组表协议（2026-11-17）

本稿只保留四张正文表，不生成数据协议补充表或预测过滤补充表。详细工程记录保留在本文件，不作为新的表格输入论文。

## 数据与模型选择

固定列顺序：DLRSD、iSAID、LoveDA、Potsdam、Vaihingen、UAVid、UDD5、VDD。
训练使用 OVRSISBench DLRSD 发布包官方成员：5601 train / 1401 val。17类，不删零，不设置背景。LoveDA val 仍用于当前 checkpoint 选择，是开发域；尚未切换源域选择，不能因为本轮重排表格就声称目标标签未使用。其余六项为独立目标域，前提是未用于模型或超参数开发。每次训练每个种子只选择一份权重，在所有目标域复用。

本地类别数分别为17/15/7/6/6/7/5/7。UAVid使用合并车辆七类：background、building、road、tree、low vegetation、car、human；原始车辆ID5/6合并为5，human由7变为6。数据版本为prepared-uavid7-v2，协议为uavid_7class_merged_car_val_resize1008_v2；旧八类结果不并入新七类结果，也不据此自动认定外部文献协议matched。背景处理详见正文及具体数据配置。当前 LoveDA 配置实际 reduce_zero_label=True，先对原始标签做零值处理，再进行背景排除；DESIGN.md 中写False的内容不作为已核实标签映射，运行前须结合转换脚本和原始值域确认。

主要指标mIoU；mAcc和逐类IoU继续保存在标准结果，不另增表。LoveDA本方法仅七类含背景，须检查真实class_names与逐类union。无真实结果时一律--。主表不显示综合均值；正式导出仍需锁定完整协议与明确选择器，删除平均列不等于解除协议校验。

## 四张表与实验计划

- main_rskt8：两个无适配SAM3基线、adapter、Full及七个文献参考行；Adaptation data只描述额外适配，不描述基础模型全部预训练资源。
- ablation_core：Full、adapter、对角类间注意力、无类内、无初始语义先验、简单融合、单方向、无蒸馏。删除Generic CLIP。
- ablation_freeze：图文均冻结、仅文本冻结、仅图像冻结、均attention微调。Full共享同一结果与checkpoint。其余模块训练与模型选择规则保持一致。
- efficiency：语义、语义乘presence、adapter、Full、单方向；统一测量条件。presence基线包含完整语义与实例流程。

文稿表内不显示计划编号；CSV内部plan_id只用于追踪。数值选择器使用variant_id，不提供旧编号兼容。T04和预测过滤分组退出当前稿件计划。结构/损失/冻结变体分别重训练；新增基线只评测原始SAM3预训练权重；现已提供语义及语义乘presence入口与完整消融配置，但未执行正式实验。

## 原生结果与溯源

标准结果与config.json、metrics.jsonl、checkpoint sha256保持一一对应。通过现有collect工具获取长表，显式指定eval_ids、预定expected_seeds、每seed权重哈希、eval_protocol_id和eval_split。不自动挑最好/最新记录；同seed多checkpoint不能平均。无适配SAM3行training_dataset明确为none，训练变体为dlrsd，不能为适配选择器而伪造训练源。

Full在主表、核心消融和冻结表复用相同选择；同variant在不同表的同数据集选择完全一致。同一训练run跨目标域必须使用相同seed集合和每seed权重。效率可训练参数从实际测量JSON读取，不估算。正式模式需完整必需结果与protocol_locked=true；draft只容许计划缺失，不容许非法数值。

单种子不报标准差，多种子报样本标准差。正式seed集合尚未锁定；默认42不是已经完成的实验。

## 文献证据

52个主表文献值的本轮核对见comparison_audit_20261117.md及对应JSON。历史CSV保留原PDF哈希、页码、版本与protocol_status；七个参考行不按统一条件排名。四个LoveDA单源结果仍unresolved，依既有用户许可加标记展示。方法引用、数字来源和官方权重训练源分别说明，尤其GSNet官方LD50K与RSKT报告iSAID不混淆。


## 当前工程执行协议

训练为batch 4、10000步、AdamW LR 2e-6，RemoteCLIP倍率0.01，预热500步，相对余弦9500步；每1000步保存并验证LoveDA，保留10份周期权重。损失、增强及开发域选择角色保持原设置。训练队列11项，独立验证手选每项checkpoint；main组八域、ablation组LoveDA/Potsdam/UAVid。两项training-free SAM3基线共享原始预训练文件并记录step 0。当前seed 42仍为默认运行设置，不代表已完成或锁定正式多种子实验。

验证按训练快照恢复变体并严格加载权重，校验checkpoint指纹；同一次选择跨域复用。结果各目录保存路径/哈希/协议与标准JSON，改选权重须新validation_name。当前master的Potsdam、UAVid、UDD、VDD背景参与前向；LoveDA仍排除背景提示。批量入口不采集或推算效率表数据，也不自动选最佳目标分数。详见docs/experiment_suites.md。


## 仅更新特征流的新增消融

`encoder_only_update`令`model.aggregation.update_streams="feature_only"`，所有类间/类内块均跳过语义更新，仅更新encoder feature；类内仍含普通/移位两个窗口步骤。score embed保持本次前向的RemoteCLIP初值，参与Q/K且不detach；专用CLIP中间层语义指导不参与。单独重训练，沿用batch 4/10000步/LR 2e-6及相同损失，评测LoveDA/Potsdam/UAVid；队列增加为12项。该项与无初始先验消融有不同含义，衡量动态语义更新及附属指导路径的联合贡献。
