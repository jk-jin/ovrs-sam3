# ReCA-SAM3 可编辑论文图

五图按照确认的五图预览重绘：浅色分区、梯形编码器、特征堆叠、黑色前向流、紫色状态流，以及红色虚线训练蒸馏路径。源文件保留原生形状、文字、分组和连接器；论文仅加载同名 PDF。五张图均已删除外层虚线框和顶部整图标题，编号与图题由LaTeX的caption统一提供；内部子图/分区名称保留。

| 图 | 可编辑源文件 | 论文图片 | 画布宽×高 |
| --- | --- | --- | --- |
| 动机 | [motivation.drawio](src/motivation.drawio) | [motivation.pdf](motivation.pdf) | 1448×250 |
| 总体框架 | [overview.drawio](src/overview.drawio) | [overview.pdf](overview.pdf) | 1448×504 |
| 双流聚合 | [aggregation.drawio](src/aggregation.drawio) | [aggregation.pdf](aggregation.pdf) | 768×340 |
| SDF | [sdf.drawio](src/sdf.drawio) | [sdf.pdf](sdf.pdf) | 675×336 |
| 定性对比 | [qualitative_comparison.drawio](src/qualitative_comparison.drawio) | [qualitative_comparison.pdf](qualitative_comparison.pdf) | 1448×302 |

本轮保留五图的模块关系、横向顺序与颜色，增加实际模块高度、纵向间距和定性图片槽高度；文字字号、圆形加号、窗口网格、图标及输入/输出影像保持比例。Figure 1保留三栏内容，Figure 2–5沿用确认的参考设计：总体图上方为SAM 3/FPN/条件特征路径，中间为双流交替聚合，下方单独列出训练蒸馏；聚合细节图包含类别token、普通/移位窗口与跨块顺序；SDF展开两条分支中的四个旋转、共享卷积、逆对齐均值和残差；定性图保留左侧五列三行与右侧占位说明、共享色标，图片槽从144×34调整为144×68。

图中所有遥感影像、GT和预测仍为明确标注的占位图，不是实验结果。选中图片槽后用Edit Image替换；图像已嵌入源文件，无外部占位模板依赖。Figure 1中的矢量图标与内嵌图片保留在源文件内部；仓库不再保存或导出独立SVG文件。

## 保存源图后自动生成PDF

本机需安装draw.io Desktop，并确保`drawio`可执行命令在PATH中；也可用环境变量`DRAWIO`指定可执行文件路径。Python只需标准库。

在`paper/`目录运行：

```bash
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

`.latexmkrc`会在每次执行编译命令时先检查五份源图；内容变化时直接调用draw.io导出PDF，再交给XeLaTeX编译论文。导出只使用PDF格式，不生成中间SVG。若源图和PDF的SHA-256均与清单一致，则跳过导出；因此未修改图时可以直接使用仓库内PDF，不依赖本机draw.io。源图已修改却没有draw.io时，编译会报错，避免误用旧PDF。

保存`.drawio`后重新运行上述命令即可看到更新；此配置不启动后台文件监视器。若编辑器的构建工具直接调用`xelatex`，请把构建命令改为`latexmk`，否则不会执行自动导出。

也可以在仓库根目录单独导出：

```bash
python3 paper/scripts/export_figures.py
# 强制重新导出全部五张图
python3 paper/scripts/export_figures.py --force
```

手动导出使用draw.io的File → Export As → PDF，选当前页并裁切到内容，保存到本目录对应PDF。自动脚本使用相同的原生渲染器，保留2像素边距；导出成功才替换旧PDF。更新图时同步保存`.drawio`、PDF和`figure_manifest.json`。清单中的哈希用于内容检测，不依赖文件时间戳。

## 配色与连线

| 元素 | 风格 |
| --- | --- |
| SAM 3、条件特征、特征流 | 浅蓝；冻结模块带雪花 |
| RemoteCLIP | 浅黄梯形；火焰表示attention tuning |
| 类别指导、语义流 | 浅紫；紫色状态连接线 |
| 类内块、SDF | 浅绿；SDF两条分支分别为浅紫和浅绿分区 |
| 卷积、投影、采样 | 浅橙 |
| 输入类别与定性表头 | 浅青 |
| 训练参考与蒸馏 | 红色虚线，与前向路径分开 |

SDF两条分支参数独立，各分支内部四个旋转共享卷积权重；先逆旋转对齐、求均值再残差相加。双流更新顺序为语义流先、特征流后；Q/K组合两种状态，V来自当前更新状态。Q、K、V分别表示query、key、value。总体图的原始条件特征、FPN72和FPN144/288分别连接相应融合/解码模块，teacher与student输出共同指向蒸馏损失。

五图沿用跨栏加载，保持各自比例。定性列为Input / Ground truth / SAM 3 / Adapter / ReCA-SAM3；数据集行为LoveDA、Potsdam、UAVid。替换真实结果时使用相同样本、相同裁块、各数据集共享类别配色，并包含失败案例。
