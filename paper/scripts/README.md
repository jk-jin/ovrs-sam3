
# 论文四表生成器

从仓库根目录运行`python paper/scripts/build_experiment_tables.py --draft`。只用Python标准库，生成四张正文表和build_report.json，不生成协议/后处理或补充表。

## 数值来源

文献只从literature_results.csv按显式result_id选择；保持原报告精度、单位与来源状态。四个既有获准LoveDA值只在参考区加dagger，仍unresolved。主表含Adaptation data，不显示均值。数值来源核验记录是comparison_audit_20261117.json，不代替文献CSV。

本方法与两个原始SAM3基线从现有collect工具生成的fraction长表选择，不能填入示例或fixture。每个selector必须有eval_ids、expected_seeds、checkpoint_sha256_by_seed、eval_protocol_id、eval_split。基线training_dataset为none，训练变体为dlrsd。正式模式只接受完成且全量的独立evaluation记录；开发域LoveDA的成绩应在另行eval-only后显式选择，不把训练验证日志自动当测试。

- 主表：main_rows的native_selection，按数据集选择。
- 核心/冻结：统一ablation_native_selection，按variant_id再按数据集选择，Full键为full。表内不显示plan_id。
- 可训练参数：freeze_parameter_selection，按variant_id选择标准JSON的真实efficiency字段。
- 开销：efficiency_selection按semantic_only、semantic_presence、simple_adapter、full、single_orientation选择列。
- 同variant跨表同数据集必须使用完全相同selector；同variant跨数据集必须使用相同seed集合和权重。Full行在冻结表不新训或重新选权重。
- 预定seed必须完整，不挑最大/最新，不平均同seed多checkpoint。单seed不显示标准差，多seed样本标准差ddof=1。LoveDA必须具有完整七类名称/编号和非零union。

效率selector包括真实eval_id、checkpoint_sha256、json_path（位于efficiency下）和正display_scale。列键total_parameters_m、trainable_parameters_m、time_ms、peak_memory_gb只是显示选择键，不定义另一套结果schema。数值从result_path指向的真实JSON提取。计时统一硬件、精度、输入、batch、提示、chunk、预热、同步与缓存，presence基线包含完整语义和实例路径。

## 构建与验证

draft仅容许计划缺失生成--，不容许非法数值或身份冲突。不带--draft要求全部必需原生/消融/冻结/开销数据齐备且protocol_locked=true；参考行缺项不阻止，因原论文没有报告而留--。

测试：`PYTHONDONTWRITEBYTECODE=1 python paper/scripts/test_build_experiment_tables.py`。合成记录仅在内存/临时目录；不进入论文结果目录。根工具的原生schema不放宽。export_config.json同步四个本地主表方法身份，供实际结果产生后使用。


## 表格版式

四表统一生成`tabularx{\linewidth}`并采用可伸缩数值列；表注也使用`\linewidth`，保证跨栏/单栏的左右边界一致。main.tex提供tabularx包和居中的C列，主表跨栏、其他表单栏；长表头显式分行。不要手工缩放已生成表格或恢复自然宽度tabular。单栏表允许here/top/bottom/page，跨栏表允许top/bottom/page。main中的唯一浮动边界位于结论之后、参考文献之前，避免在结论前集中清空队列造成正文末尾空栏；不为各小节强制分页。
