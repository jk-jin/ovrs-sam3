# ReCA-SAM3 可编辑论文图

四份原生 draw.io 图稿与同名矢量 PDF 已对齐当前模型。图内文字和图注使用英文；没有独立 SDF 图，总体框架仅保留 SDF 模块。所有源图由可编辑形状、文字和连接器组成，无外层虚线框或整图顶部大标题；聚合细节保留两个子图标题。

## 文件与正文接入

| 图 | 可编辑源文件 | 论文图片 | 正文状态 |
| --- | --- | --- | --- |
| 动机 | [motivation.drawio](src/motivation.drawio) | [motivation.pdf](motivation.pdf) | 引言中 Fig. 1，三种信息流对比 |
| 总体框架 | [overview.drawio](src/overview.drawio) | [overview.pdf](overview.pdf) | 方法中 Fig. 2，两个完整聚合块与训练参考路径 |
| 聚合／多尺度窗口 | [aggregation.drawio](src/aggregation.drawio) | [aggregation.pdf](aggregation.pdf) | 方法中 Fig. 3，(a) 单块更新，(b) 共享三尺度窗口 |
| 定性对比模板 | [qualitative_comparison.drawio](src/qualitative_comparison.drawio) | [qualitative_comparison.pdf](qualitative_comparison.pdf) | 未接入正文，等待真实样本和预测替换占位 |

## 当前图示定义

- 动机图区分匹配表征上的代价聚合、解码后的逐类掩码融合，以及本文解码前的跨提示特征聚合；不使用假想预测展示性能收益。
- 总体图的 Block 1 与 Block 2 均含类间注意力和多尺度窗口注意力，块后执行最终 LayerNorm。语义引导在两个块中复用，紫色箭头仅表示 Q/K 引导；绿色 FPN 引导用于窗口 Q/K。原始条件特征与 FPN72 接 SDF，高分辨率 FPN 接冻结解码器。训练参考路径用红色虚线区分，推理不运行教师。
- 聚合图 (a) 将条件特征的 LayerNorm 输出同时用于 Q/K 与 V；语义引导单独归一化，仅经拼接参与 Q/K。图中两个圆形加号分别对应类间残差和多尺度空间残差。
- 聚合图 (b) 的三路输入均来自类间更新后的同一份特征，分辨率为 36×36、24×24、12×12；普通窗口均为 12×12，对应 9、4、1 个窗口。最小分支仅在其 12×12 网格上做全图注意力。
- 三个绿色窗口模块在同一个聚合块内共用预归一化、Q/K/V 投影、输出投影和相对位置偏置，不同块独立参数化。引导按尺度缩放，仅接 Q/K，值来自条件特征。
- 三路输出是特征增量，先对齐到 36×36，再经三个可学习标量的 Softmax 权重求和；组合图 (a) 在融合后执行一次空间残差。尺度权重没有图像输入，也不表示逐像素动态门控。
- 图中没有语义状态更新、移位窗口、CLIP 中间层引导或 SDF 方向旋转。

定性模板列为 Input、Ground truth、SAM 3 Semantic-only、SAM 3 Semantic × Presence、ReCA-SAM3，数据集行为 LoveDA、Potsdam、UAVid，保留局部放大槽。全部图像／掩码槽都是明确标注的占位，不能作为实验结果；接入正文前需替换为相同样本、裁块和类别色表的真实结果，并纳入失败案例。

## 配色与编辑

| 元素 | 风格 |
| --- | --- |
| SAM 3、条件特征与更新路径 | 浅蓝，冻结模块以文字注明 |
| RemoteCLIP | 浅黄，标注 Attention tuning |
| 语义引导 | 浅紫，紫色 Q/K 旁路 |
| 窗口注意力、FPN 引导和 SDF | 浅绿 |
| 投影与缩放 | 浅橙 |
| 输入类别与定性表头 | 浅青 |
| 训练参考与蒸馏 | 红色虚线 |

源图采用 Liberation Serif，PDF 已嵌入常规与粗体字体。源图文字显式分行，几何与 PDF 保持一致；PDF 中图形、窗口网格和文字均为矢量。更改字号、形状或连接后，应按论文实际跨栏宽度检查网格、连线和文字。核对结果与文件哈希维护于 ../notes/source_snapshot.json。

## 保存源图后生成 PDF

安装 draw.io Desktop，将 drawio 加入 PATH，或用环境变量 DRAWIO 指定可执行文件。论文只使用 PDF，不额外导出 SVG。在 paper/ 中运行：

```bash
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

.latexmkrc 根据 figure_manifest.json 检查源图与 PDF 哈希，内容改变时调用 draw.io 导出，再编译论文；内容未变时直接使用已提交 PDF。源图改变但 draw.io 不可用时停止构建，避免误用旧图；不启动后台监视器。

也可在仓库根目录导出清单内的图：

```bash
python3 paper/scripts/export_figures.py
python3 paper/scripts/export_figures.py --force
```

更新图时同步提交 .drawio、PDF 和清单。定性图准备齐全后再增加正文引用；不要直接把占位模板当作定性实验。
