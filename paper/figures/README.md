# ReCA-SAM3 可编辑论文图

四份原生 draw.io 图稿与同名矢量 PDF 已对齐当前模型。图内文字和图注使用英文；没有独立 SDF 图，总体框架仅保留 SDF 模块。overview 按用户确认的预览重绘为三个白底虚线分区，标题位于白色留白，内部功能组使用浅色背景；其余图维持原设计。源图由可编辑形状、文字、图标和连接器组成。

## 文件与正文接入

| 图 | 可编辑源文件 | 论文图片 | 正文状态 |
| --- | --- | --- | --- |
| 动机 | [motivation.drawio](src/motivation.drawio) | [motivation.pdf](motivation.pdf) | 引言中 Fig. 1，三种信息流对比 |
| 总体框架 | [overview.drawio](src/overview.drawio) | [overview.pdf](overview.pdf) | 方法中 Fig. 2，三板块、特征叠片与灰色训练参考栏 |
| 聚合／多尺度窗口 | [aggregation.drawio](src/aggregation.drawio) | [aggregation.pdf](aggregation.pdf) | 方法中 Fig. 3，(a) 单块更新，(b) 共享三尺度窗口 |
| 定性对比模板 | [qualitative_comparison.drawio](src/qualitative_comparison.drawio) | [qualitative_comparison.pdf](qualitative_comparison.pdf) | 未接入正文，等待真实样本和预测替换占位 |

## 当前图示定义

- 动机图区分匹配表征上的代价聚合、解码后的逐类掩码融合，以及本文解码前的跨提示特征聚合；不使用假想预测展示性能收益。
- 总体图分为类别条件特征生成、RemoteCLIP 引导聚合、融合与分割三个板块。编码器采用梯形及神经网络图标，雪花表示冻结、火焰表示 attention 微调；特征、文本提示和 FPN 分别采用叠片、卡片和金字塔表示。主要背景为白色，仅内部功能组使用浅蓝、浅奶油、浅紫、浅桃与浅绿。训练蒸馏栏使用灰色虚线、灰色文字及低对比度特征，不使用红色。
- Block 1 与 Block 2 均含类间注意力和多尺度窗口注意力，两个块以错位叠框及独立权重标签表示，块后执行最终 LayerNorm。前后块同时在水平和垂直方向错位，后块露出顶部、右侧和底部轮廓，不能让两个右边界重合。类间特征采用沿垂直深度分离的四层透视平面，前景平面标出同一空间位置，避免沿斜边平移而压成一张斜片；类间示意展示同一位置在示例提示间的联系；类内示意展示三个普通窗口尺度及共享参数。语义引导在两个块中复用，仅表示 Q/K 引导，绿色 FPN 引导只用于窗口 Q/K，条件特征提供 V 并接受残差。
- 原始条件特征与 FPN72 接 SDF，高分辨率 FPN 接冻结解码器。同名 ``to (b)/(c)'' 与 ``from (a)'' 端口表示相同 SAM 3 张量的复用，用于避免额外的跨图长线，不表示新增模型支路。输入图像只接图像编码路径，文本提示接 SAM 3 提示条件路径与 RemoteCLIP 文本编码器。
- overview 的 RGB、RemoteCLIP 响应、学生响应、预测与教师响应五处为占位组；特征叠片及提示交互图是示意，不是实测热力图。替换时可删除对应占位组并插入真实图像对象；图注须保留可视化对象的真实含义。
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
| overview 训练参考与蒸馏 | 中性灰色虚线、文字与低对比度图形 |

overview 采用 Liberation Sans，其余图保留 Liberation Serif；PDF 嵌入常规与粗体字体。overview 的图标、占位组和特征叠片均为原生可编辑对象，没有将整张预览 PNG 嵌入源图。内部功能卡片的内容须留出可见内边距，通常约 15–20 个画布单位；小型标量卡片约 8–12 个单位。检查标题、底部共享权重标签及右侧输出占位，不能与容器边框重合；不要用缩小文字代替必要的纵向留白。overview 当前画布为 2000×1000。文字显式分行，几何与 PDF 保持一致；PDF 中图形、窗口网格和文字均为矢量。更改后应按实际跨栏宽度检查网格、连线和文字。核对结果与文件哈希维护于 ../notes/source_snapshot.json。

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
