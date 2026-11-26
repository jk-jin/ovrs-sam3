# ReCA-SAM3 可编辑论文图

四份原生 draw.io 图稿与同名矢量 PDF 已对齐当前模型。图内文字和图注使用英文；没有独立 SDF 图，总体框架仅保留 SDF 模块。源图由可编辑形状、文字、图标和连接器组成。

后续新绘图和重绘统一遵循 **[论文绘图规范](STYLE_GUIDE.md)**。该文件集中维护布局、色值、字体、留白、特征叠片、Block 层次、图像使用和验收要求；本页维护文件入口与当前模型的图示定义。视觉参考为已修正间距的 overview；motivation 已按同一风格重绘为四子图，聚合图与定性模板保留现有图稿。

## 文件与正文接入

| 图 | 可编辑源文件 | 论文图片 | 正文状态 |
| --- | --- | --- | --- |
| 动机 | [motivation.drawio](src/motivation.drawio) | [motivation.pdf](motivation.pdf) | 引言中 Fig. 1，2×2 四种适配位置对比 |
| 总体框架 | [overview.drawio](src/overview.drawio) | [overview.pdf](overview.pdf) | 方法中 Fig. 2，三板块、特征叠片与灰色训练参考栏 |
| 聚合／多尺度窗口 | [aggregation.drawio](src/aggregation.drawio) | [aggregation.pdf](aggregation.pdf) | 方法中 Fig. 3，(a) 单块更新，(b) 共享三尺度窗口 |
| 定性对比模板 | [qualitative_comparison.drawio](src/qualitative_comparison.drawio) | [qualitative_comparison.pdf](qualitative_comparison.pdf) | 未接入正文，等待真实样本和预测替换占位 |

## 当前图示定义

- 动机图采用 2×2 白底子图：(a) CAT-Seg／RSKT-Seg 的匹配代价聚合；(b) SegEarth-OV3 的解码响应融合；(c) ProC-SAM3／ROSE-SAM3 的提示与响应适配；(d) ReCA-SAM3 的解码前条件特征聚合。(c) 中两条离线路径是替代策略，ROSE 依据无标签图像冻结响应选提示，不能画成二者串联。
- 动机图用响应图片槽与蓝色条件特征叠片区分更新对象，保留八处明确图片槽；无标签校准数据仅用图标示意，不伪造实际响应。冻结模块用雪花，(b)/(c) 的免训练与 (d) 的监督适配明确区分；SDF 的跳连及空间细节在图注说明省略。各子图按主要适配位置归类，不表示范式互斥，也不声称已有方法未考虑类别证据。
- 总体图分为类别条件特征生成、RemoteCLIP 引导聚合、融合与分割三个板块。主推理流程与训练蒸馏栏分开，视觉表达按 STYLE_GUIDE.md 执行。
- Block 1 与 Block 2 均含类间注意力和多尺度窗口注意力，不同完整块独立参数化，块后执行最终 LayerNorm。类间示意展示同一位置在示例提示间的联系；类内示意展示三个普通窗口尺度及共享参数。语义引导在两个块中复用，仅表示 Q/K 引导，绿色 FPN 引导只用于窗口 Q/K，条件特征提供 V 并接受残差。
- 原始条件特征与 FPN72 接 SDF，高分辨率 FPN 接冻结解码器。同名 ``to (b)/(c)'' 与 ``from (a)'' 端口表示相同 SAM 3 张量的复用，用于避免额外的跨图长线，不表示新增模型支路。输入图像只接图像编码路径，文本提示接 SAM 3 提示条件路径与 RemoteCLIP 文本编码器。
- overview 的 RGB、RemoteCLIP 响应、学生响应、预测与教师响应五处为占位组；特征叠片及提示交互图是示意，不是实测热力图。替换时可删除对应占位组并插入真实图像对象；图注须保留可视化对象的真实含义。
- 聚合图 (a) 将条件特征的 LayerNorm 输出同时用于 Q/K 与 V；语义引导单独归一化，仅经拼接参与 Q/K。图中两个圆形加号分别对应类间残差和多尺度空间残差。
- 聚合图 (b) 的三路输入均来自类间更新后的同一份特征，分辨率为 36×36、24×24、12×12；普通窗口均为 12×12，对应 9、4、1 个窗口。最小分支仅在其 12×12 网格上做全图注意力。
- 三个绿色窗口模块在同一个聚合块内共用预归一化、Q/K/V 投影、输出投影和相对位置偏置，不同块独立参数化。引导按尺度缩放，仅接 Q/K，值来自条件特征。
- 三路输出是特征增量，先对齐到 36×36，再经三个可学习标量的 Softmax 权重求和；组合图 (a) 在融合后执行一次空间残差。尺度权重没有图像输入，也不表示逐像素动态门控。
- 图中没有语义状态更新、移位窗口、CLIP 中间层引导或 SDF 方向旋转。

定性模板列为 Input、Ground truth、SAM 3 Semantic-only、SAM 3 Semantic × Presence、ReCA-SAM3，数据集行为 LoveDA、Potsdam、UAVid，保留局部放大槽。全部图像／掩码槽都是明确标注的占位，不能作为实验结果；接入正文前需替换为相同样本、裁块和类别色表的真实结果，并纳入失败案例。

## 绘图规范与编辑

所有视觉规则及提交前验收表见 [STYLE_GUIDE.md](STYLE_GUIDE.md)。模型语义以本页、源码和有效配置为准；图的风格不能改变信息流含义。可编辑源图、PDF 和清单须同步，实际核对结果记录在 [source_snapshot.json](../notes/source_snapshot.json)。

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
