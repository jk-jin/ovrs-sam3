# 当前论文表格生成

从仓库根目录执行：

```bash
python paper/scripts/test_build_experiment_tables.py
python paper/scripts/build_experiment_tables.py --draft
```

输入为 `paper/data/table_spec.json`、文献 CSV 和真实标准结果汇总 CSV。当前生成主结果与效率两张表；旧消融与冻结策略行已清空，相应表不再生成。

主表 `native_selection` 按数据集显式填写实际 eval_id、种子、权重哈希与协议。Full 必须来自 `reca-sam3-multiscale-windows` 新训练，同一权重用于全部数据集。效率使用 `efficiency_selection` 选择 Semantic-only、Semantic × Presence 和 Full 的真实测量，不复用旧结构结果。

不带 `--draft` 时要求必需本地结果/效率及协议锁完整。draft 只允许缺失值显示 `--`，非法数值、身份冲突、口径冲突仍报错。文献参考行保留原始来源协议及类别限制。
