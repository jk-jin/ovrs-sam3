# 实验与评测协议（2026-11-24）

## 数据与权重选择

实现标识 `reca-sam3-paper-ablation-v1`。RemoteCLIP中间层配置、提取与缓存已删除；默认Full设计不变，十个训练变体和两个评测基线已提供代码与配置。

DLRSD 官方 5601 张训练图像用于适配，1401 张验证图像用于源域评测。LoveDA 七类验证用于 checkpoint 选择，是开发域。其余 iSAID、Potsdam、Vaihingen、UAVid、UDD5、VDD 为目标域。消融重点展示 LoveDA、Potsdam、UAVid；LoveDA 不作为独立目标域证据。

每个变体在相同预算内按 LoveDA 七类 mIoU 选一份权重，用于该种子的全部域及不同表格。主模型只训练一次，在主表、核心消融、图文微调与逐类表中复用同一选择。禁止逐目标数据集挑最佳 checkpoint。

## 十个训练变体

| 优先级 | ID | 变体 |
| --- | --- | --- |
| 1 | full | 当前主模型：两端 attention 微调、完整提示交互、共享三尺度窗口、score embedding、蒸馏 |
| 2 | self_only | 每个展开提示只关注自身，严格对角屏蔽，连同类同义提示也不互看 |
| 3 | no_remoteclip_guidance | 两类注意力均不拼接 score embedding；移除无效 RemoteCLIP 路径，保留 SAM3 FPN 指导 |
| 4 | no_intra | 每块跳过全部类内注意力；保留类间和 SDF |
| 5 | single_scale | 仅36×36上的一次12×12普通窗口；无尺度融合，直接残差 |
| 6 | unshared_windows | 保留三尺度和Softmax融合，窗口参数每层三套独立学习 |
| 7 | frozen_remoteclip | RemoteCLIP 图像与文本两端冻结 |
| 8 | image_only | 仅图像 attention 微调，文本冻结 |
| 9 | text_only | 仅文本 attention 微调，图像冻结 |
| 10 | no_distillation | 蒸馏权重0，跳过教师解码 |

图文微调表的 attention/attention 行即 Full；四组合不额外增加一个 Full。没有移位窗口或中间层引导消融。训练预算统一 batch 4、10000步、基础LR 1e-4、RemoteCLIP倍率0.01、500步预热与9500步相对余弦、梯度裁剪0.01。各消融外的增强、损失及融合保持相同。

首轮按当前默认 seed 42 开展；正式多种子集合仍需预先锁定，各变体使用相同集合。单种子不写标准差，多种子使用样本标准差。当前没有登记完成的训练或测量。

## 两个原始 SAM3 基线

仅自跑 `semantic_only` 和 `semantic_presence`，两者复用相同原始权重、training-free、step 0。Presence 只在每提示语义概率上乘一次；执行原始实例查询以生成 presence，不融合实例掩码。已有文献行继续作为原设置参考，不自动安排其本地训练。

## 标签与推理

全部配置类别参与前向，背景／clutter不再排除。原始标签转换与类别空间保持现有配置：LoveDA原始1..7转为0..6，原始0转255。Potsdam/Vaihingen道路同义提示、VDD屋顶同义提示均固定，所有变体采用一致提示。合并同义提示后过滤；各数据集阈值保持当前配置，不能逐变体重新优化：LoveDA 0.5，Potsdam/Vaihingen 0.0，UAVid 0.3，UDD5 0.4，VDD 0.5；DLRSD及当前iSAID-IgnBG配置无背景，预测概率过滤不生效。背景回填阈值为0.0。输入统一Resize到1008×1008，TTA关闭。

## 表格与溯源

五张表均在实验章节引用。核心结构表7行，图文组合表4行，共10个不同训练变体；效率表包括两个基线、Full、单尺度、独立权重。LoveDA逐类表只使用本地三方法的相同评测选择。

正式填数要求实际 eval_id、seed、checkpoint SHA-256、数据划分与协议，且评测完整。文献与本地分区，不混算综合平均；原报告来源不改写。效率统一设备、精度、输入、提示数量、批量、预热、同步和缓存条件，在相同推理条件下实际测量。缺失值为 --，不是0。

## 比较与双协议结果选择

文献引用、iSAID 两列、UAVid 七／八类与未知来源展示统一遵循[comparison_policy.md](comparison_policy.md)，不在此重复维护。主表本地结果的 iSAID 选择键为 iSAID-15-IgnBG 与 iSAID-16-BG；标准结果仍为 eval_dataset=isaid，以不同 eval_protocol_id 区分，复用同一种子／权重。

含背景验证入口已实现：configs/test/isaid_16_bg.py，背景参与前向，prediction_prob_threshold=0.5，background_prob_threshold=0.0；15类入口为isaid_15_ignbg.py，背景忽略。两种协议读取相同原始标签，采用各自固定映射。实际工程与实验任务只维护于[manuscript_status.md](manuscript_status.md)。
