# 当前评测协议

当前实现为 `reca-sam3-feature-blocks-v3`。DLRSD 官方 5601 张训练图像用于适配，1401 张验证图像用于源域评测；LoveDA 七类验证用于模型选择。其他目标为 iSAID、Potsdam、Vaihingen、UAVid、UDD5、VDD。保持当前数据配置中的类别映射、背景规则与阈值。

训练预算为 batch 4、10000 步、基础 LR 2e-6、RemoteCLIP 倍率 0.01、500 步预热与 9500 步相对余弦；每 2000 步保存/验证，W&B offline。新 Full 单独训练、单独保存，不使用旧结构权重恢复。

同一次训练只选择一份权重，用于全部八域及不同表格。`paper_validate.py` 中 `None` 表示待选，不自动挑选 best/latest。两个原始 SAM3 比较只使用原始权重，training-free、step 0，无 DLRSD 适配。

| 表格 | 当前内容 |
| --- | --- |
| main_rskt8 | 两个原始 SAM3 比较、新 Full 与外部文献参考行 |
| efficiency | Semantic-only、Semantic × Presence、新 Full 的真实开销 |

旧消融表与旧计划已删除。选择器需使用新结构真实结果的 eval_id、seed 和 checkpoint 哈希，不自动沿用旧 Full 数值。缺失成绩显示 `--`；文献原报告与本地结果分区，保留来源和类别口径标记。

正式导出要求完整标准结果及锁定协议；draft 仅提供缺失占位，不生成虚构数值。效率测量需统一设备、精度、输入、提示数、批量、预热、同步与缓存状态。
