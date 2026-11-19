# ReCA-SAM3 可编辑论文图

五图采用同一套浅色渐变分区、对比更明确的纯色模块、虚线边框、编码器梯形和特征堆叠。论文加载矢量PDF；draw.io源文件包含原生形状、独立文字、已连接端点的箭头和可替换图像，不是把预览PNG贴进画布。文字跟随所属模块，特征堆叠使用原生分组。

| 图 | 可编辑源文件 | 矢量图 | 画布宽×高 |
| --- | --- | --- | --- |
| 动机 | [motivation.drawio](src/motivation.drawio) | [PDF](motivation.pdf) / [SVG](motivation.svg) | 1260×295 |
| 总体框架 | [overview.drawio](src/overview.drawio) | [PDF](overview.pdf) / [SVG](overview.svg) | 1440×565 |
| 双流聚合 | [aggregation.drawio](src/aggregation.drawio) | [PDF](aggregation.pdf) / [SVG](aggregation.svg) | 1320×480 |
| SDF | [sdf.drawio](src/sdf.drawio) | [PDF](sdf.pdf) / [SVG](sdf.svg) | 1300×440 |
| 定性对比 | [qualitative_comparison.drawio](src/qualitative_comparison.drawio) | [PDF](qualitative_comparison.pdf) / [SVG](qualitative_comparison.svg) | 1160×415 |

五图都使用跨栏宽度、按原比例缩放。SDF改为跨栏，保证分支和方向模块的文字可读；不把详细结构挤进单栏。图题与图内文字为英文。

## 修改与导出

1. 在draw.io / diagrams.net中打开对应`.drawio`，修改模块文字、颜色、连线或图像。英文常规字体为Times New Roman；现有矢量导出使用度量相近的TeX Gyre Termes并嵌入字形。
2. 选中图像槽，用Edit Image替换。所有影像、GT、预测和局部放大槽目前均为同一份中性占位图，且明确写PLACEHOLDER；没有真实模型预测。源图自包含内嵌图像，替换共享`image_placeholder.svg`不会自动修改已嵌入的副本。
3. 通过File → Export As → PDF导出，选择当前页、Crop裁切到内容、关闭网格和阴影，保存为本目录同名PDF。保持少量边框空白，不导出整张A4空画布；可同时导出同名SVG。选项说明见[官方PDF导出文档](https://www.drawio.com/docs/manual/export/export-to-pdf/)。
4. 保存编辑后的`.drawio`并同步同名PDF/SVG；源文件是后续编辑依据，导出文件只用于显示。图的主体比例改变后，再编译`paper/main.tex`检查页内大小和浮动位置。

## 颜色与结构

大框使用从上至下的浅色渐变，小框采用同一语义色系中更饱和的纯色；两者不共用填充色。外层总框使用白色到冷灰的轻渐变，内部区域按功能保留色系，文字和连线使用深色，避免渐变干扰阅读。draw.io中的渐变是原生`fillColor` / `gradientColor`样式，仍可直接编辑。色值与文件对应关系见`figure_manifest.json`。

| 色系 | 小模块纯色 | 大区域渐变（上 → 下） | 主要用途 |
| --- | --- | --- | --- |
| 蓝 | `#BCD3ED` | `#F8FBFF` → `#E5EDF8` | SAM3特征与相关路径 |
| 黄 | `#FFE29A` | `#FFFCF1` → `#FFF0CF` | RemoteCLIP编码与文本模板 |
| 紫 | `#D4BEED` | `#FCFAFF` → `#EDE5F8` | 类别指导、语义状态及交互 |
| 绿 | `#C6DCA9` | `#FAFCF5` → `#E9F1DC` | 类内空间交互与细节分支 |
| 橙 | `#F1C49F` | `#FFF9F3` → `#F9E6D6` | 投影、卷积与训练区域 |
| 青 | `#BBDDD6` | `#F4FCFA` → `#E0F1EB` | 类别文本及辅助特征 |

SAM3是否冻结仍以frozen标识为准，RemoteCLIP的attention tuning仅表示选择的参数子集可训练。珊瑚色虚线表示训练参考与蒸馏；黑实线为前向数据流，紫色线为语义/指导流。定性图的五列标题依次使用青、绿、蓝、橙、紫，数据集行标签使用冷灰，所有真实图片槽仍为中性灰占位图。

总体图中原始72×72条件特征、FPN72和FPN144/288分别连接SDF或冻结解码器。模板相似度和dense RemoteCLIP特征共同进入guidance embedding。双流先更新语义再更新特征；SDF先统一上采样，再形成独立投影的两个分支。参考输出与学生logits共同进入蒸馏损失，损失不反向充当解码器输入。完整模型图没有presence/实例推理分支。

定性图的列为Input / Ground truth / SAM3 / Adapter / ReCA-SAM3，三个样例行是LoveDA、Potsdam、UAVid。完成实验后用相同样本、相同裁块、同数据集共享图例替换，并保留失败案例；此布局不预设任何模型效果。
