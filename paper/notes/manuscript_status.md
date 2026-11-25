# 当前未完成工作

实验设计与比较规则分别以[evaluation_protocol.md](evaluation_protocol.md)和[comparison_policy.md](comparison_policy.md)为准；表格源数据为 data/table_spec.json。这里只记录仍需落实的事项，不累积已解决问题。

## 实验运行

- 正式锁定种子集合与权重选择规则，完成十个训练变体、两个原始基线、iSAID双协议、逐类结果与开销测量。当前只有CPU行为检查和任务预览，没有登记正式native成绩；表格保持 --。

## 论文与图

- 重绘 overview、aggregation 与 SDF 的 draw.io 源文件并重新导出 PDF：正文、聚合公式与图注已同步为两个完整聚合块、固定语义 Q/K 引导、共享三尺度普通窗口及单方向 SDF；当前图内仍有旧双状态更新、分立类间／类内块、中间层引导、移位窗口或四方向融合，尚未与新正文一致。aggregation 细节图目前未接入正文，重绘后再补充引用。
- 将定性图的旧 Adapter 列替换为 Semantic-only、Semantic × Presence 两个实际基线，并在真实示例齐备后接入正文。
- 准备真实影像、GT、预测、局部放大与失败案例，使用与定量评测一致的权重和后处理。

外部模型的代码版本、历史预处理运行绑定或付费全文获取不再列为引用前置任务；文献按来源报告并保留已知协议说明。没有运行的实验不能写成性能结论。
