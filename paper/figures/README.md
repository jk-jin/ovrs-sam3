# ReCA-SAM3 可编辑论文图

方法的机制与流程说明暂用中文，公式及符号说明保留英文；图内文字与图注继续使用英文。独立 SDF 图已从正文、源文件、PDF 和导出清单删除，融合模块仍保留在模型与总体框架中，其投影、残差卷积与相加由正文和公式说明。

## 当前文件与重绘范围

| 图 | 可编辑源文件 | 论文图片 | 当前状态 |
| --- | --- | --- | --- |
| 动机 | [motivation.drawio](src/motivation.drawio) | [motivation.pdf](motivation.pdf) | 已接入正文；保留信息流对比，统一术语 |
| 总体框架 | [overview.drawio](src/overview.drawio) | [overview.pdf](overview.pdf) | 已接入正文；图内仍为旧聚合结构，待重绘 |
| 聚合细节 | [aggregation.drawio](src/aggregation.drawio) | [aggregation.pdf](aggregation.pdf) | 尚未接入正文；旧双状态更新与移位窗口需要重绘 |
| 定性对比 | [qualitative_comparison.drawio](src/qualitative_comparison.drawio) | [qualitative_comparison.pdf](qualitative_comparison.pdf) | 尚未接入正文；替换旧 Adapter 列，等待真实结果 |

当前只完成方法文字中文化和独立 SDF 图删除，以下为待确认的重绘建议，不代表图稿已经更新。

## 推荐内容分工

1. **动机图**：保留 CLIP 代价聚合、逐类 SAM 3 查询与本文解码前跨提示交互的区别。只展示由结构确认的信息流，不用假想预测证明性能提升。
2. **总体框架图**：上方为冻结 SAM 3 条件特征与 FPN，下方为 RemoteCLIP 图文编码及 score embedding；中部画两个完整 Aggregation Block，各包含 Cross-class Attention 和 Multiscale Window Attention。语义引导旁路进入两个块，仅连查询和键。末端用一个 SDF 框表示融合，保留原始条件特征与 FPN72 输入；高分辨率 FPN 接解码器。教师与蒸馏置于独立 Training only 区域。
3. **聚合块与多尺度共享窗口图**：推荐一个跨栏组合图，左侧子图说明单个块的类间残差与多尺度类内残差，右侧子图展开多尺度部分。若缩印后网格或文字不清楚，再拆成两幅独立图；两处不重复完整 Q/K/V 结构。
4. **定性对比图**：列为 Input、Ground truth、SAM 3 Semantic-only、SAM 3 Semantic × Presence、ReCA-SAM3；数据集行为 LoveDA、Potsdam、UAVid。统一样本、裁块和类别色表，保留局部放大与失败案例。

组合方案共四幅图；将多尺度图独立成图则共五幅。图号由 LaTeX 生成，细节图准备完毕后再接入正文。

## 多尺度共享窗口子图布局

- 单个块完成类间更新后的 36×36 条件特征，从同一输入分为 36×36、24×24、12×12 三条平行支路。后两路标出 Resize，避免误画为串行更新。
- 三路均采用 12×12 普通窗口，分别有 3×3、2×2、1×1 个窗口。最小尺度标注 Global on 12×12 grid，避免暗示原分辨率上的稠密全局注意力。
- 同色窗口模块以虚线括号标注 Shared parameters within each block。共享范围包括 LayerNorm、Q/K/V 与输出投影、相对位置偏置；不同聚合块独立参数化。
- 语义引导和 SAM 3 FPN 引导按尺度缩放后接 Q/K，V 来自对应尺度的条件特征。窗口输出标为 Feature increment，较小尺度的增量上采样至 36×36。
- 三路增量乘尺度权重后求和，最后与类间更新后的输入做一次残差相加。权重以 Three learned logits → Softmax 表示，在该块全部提示、通道和空间位置间共享；不画成图像生成的逐像素门控。
- 不绘制语义状态更新、移位窗口、循环平移或 CLIP 中间层路径。普通窗口的相对位置偏置保留，可在图注中说明。

## 配色与缩印

| 元素 | 风格 |
| --- | --- |
| SAM 3、条件特征与更新路径 | 浅蓝，冻结模块带雪花 |
| RemoteCLIP | 浅黄，火焰表示 attention tuning |
| 语义引导 | 浅紫，紫色旁路箭头，只接 Q/K |
| 窗口注意力与 SDF 框 | 浅绿 |
| 卷积、投影与缩放 | 浅橙 |
| 输入类别与定性表头 | 浅青 |
| 训练参考与蒸馏 | 红色虚线 |

继续使用原生 draw.io 矢量形状、特征堆叠与窗口网格，不加整图外层虚线框和顶部大标题。按论文实际宽度检查字号、线宽和网格可辨性；用箭头说明更新对象、共享范围与残差位置，不重复堆砌正文公式。现有影像、GT 与预测槽仍是占位，不代表真实实验结果。

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

更新图时同步提交 .drawio、PDF 和清单。独立多尺度图创建后再加入清单及正文；组合图方案直接更新现有 aggregation 图。
