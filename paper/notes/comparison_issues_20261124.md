# 数据集协议与比较结果：当前结论

日期：2026-11-24。当前决定：删除原Table II以及GFMR、ConceptBank、PRS-Seg的全部正文引用；不再纳入比较。Zotero导出库中的未调用记录保留，不会出现在编译后的参考文献列表中。主表保留25行、156个文献单元格；未改任何模型或数据集评测配置。

## 已明确解决

- LoveDA标准定义：原始0为no-data，设为255忽略；原始1..7减1得到0..6，原始1对应background，仍计入七类mIoU。只能转换一次，预先已处理标签不能再减1。
- SegEarth-OV3、ProC-SAM3的当前公开配置／数据类可核实上述LoveDA处理。ROSE通过MMSeg LoveDADataset使用相同默认处理；Pi-Seg注册七类并读取预处理masks_png2。
- UAVid存在明确7/8类差别：RSKT注册moving car/static car两个类；SegEarth-OV3和ROSE数据类将其合并car。文献混列表不能证明两个数字可直接排名。
- iSAID存在明确评测目标差别：本项目15前景且忽略GT背景；SegEarth-OV3/ProC公开配置为含background的16类，不忽略原始0。
- ROSE当前提示文件由官方get_cls_idx函数解析得到26个查询、16个类。README仍写25；此前“当前发布库只有25查询”的说法不再成立。原始文件与解析器URL、commit及指纹见protocol_evidence_20261124.json。
- GLoP数据可以明确作为ROSE Table I的二次引用：既注明方法原论文，也注明真实取数来源。标成GLoP-SAM3 (ROSE report)，不称亲自复现或已核对GLoP全文。
- GFMR、ConceptBank、PRS-Seg已退出比较，相关协议及重复ConceptBank条目不再是本稿比较部分的待办。VLex-SAM3、ConceptSeg继续排除。

## 推荐处理方式（尚未改变评测实现）

| 项目 | 文献汇总表 | 若要做严格同条件比较 |
| --- | --- | --- |
| UAVid | 保留来源值，分成UAVid-7与UAVid-8子列或逐格标7/8；不混排优劣、不混算平均 | 项目主协议保留7类；用相同图像划分把旧模型的8类预测和GT同时合并car后重算。已有8类混淆矩阵也可合并对应行列；只有mIoU标量不能换算7类成绩 |
| iSAID | 分成iSAID-15-IgnBG与iSAID-16-BG子列或逐格标注；不按训练/免训练名称猜协议 | 保留项目当前15类忽略背景协议，并补一套16类含背景评测用于直接对齐SAM3论文。额外评测无需重训，但需给模型明确背景预测通道／固定回填机制，不能用GT填背景。另一选择是将SAM3基线在相同GT背景忽略规则下重测 |
| ActiveSAM | 保留ActiveSAM (ProC-SAM3 reproduction)，引用ActiveSAM方法与ProC取数来源 | 当前官方代码复测应另列Ours reproduction，固定代码commit、SAM3权重、提示、类签名、阈值、数据映射；不能将旧遥感分数改标成最新版本 |
| ROSE-SAM3 | 保留论文Table I原报告值，不因为当前提示数恰好相同就宣称复现一致 | 固定286396d6832ba93ce9d0dbdcd37d02fabdc0540a，记录提示文件哈希、26个实际查询、权重和后处理；向作者确认README是否过期、该文件是否对应论文出表，再区分论文行与自跑行 |
| GLoP-SAM3 | 可整行转引已读ROSE表格，方法名引用GLoP，表注／正文引用ROSE | 有原文全文后再核对原始表及协议。无需把拿到全文当作引用转引成绩的前置条件，但不能隐去二手来源 |

## 尚需处理的证据问题

1. **RSKT历史LoveDA成绩的运行绑定**：标准定义已明确，但当前公开register_LoveDA.py仍列no-data+7语义类，读取masks_png，ignore_label=255；没有在该评测入口看到减1。离线预处理可能解释这一差别。不能断言论文实际用了8类，也不能断言全部历史数字已经按7类核实。需要该次运行的标签唯一值、预处理记录或逐类IoU/union。当前LoveDA来源标记继续保留。
2. **UAVid与iSAID同条件数值**：只有原报告标量无法完成协议转换。需要预测图、原始混淆矩阵，或运行统一评测；还必须确认图像划分、类别对应和忽略区域相同。
3. **ROSE文档/代码一致性**：已排除“当前文件少一个查询”的推断，但匹配26这个数量不能证明具体文本、权重、融合系数及分数与论文完全相同；README25仍需作者澄清。
4. **ActiveSAM复现版本**：当前官方实现有EPG/ECD与无标签COCO类签名；已有遥感分数来自另一论文的复现，出表版本没有固定，不能写成当前官方RS分数。
5. **GLoP原始协议**：引用来源可以明确解决，源论文表格、类别与后处理是否逐项一致在全文取得前仍未确认。

## 为什么不能简单“去掉背景”或“平均两种car”

- iSAID忽略GT背景时，背景像素上的全部预测都不进入统计；含背景评测时，在这些位置误检船或飞机会增加对应前景类的假阳性。仅从16类平均中扣除背景IoU并不会消除这些假阳性。因此“15前景均值但仍统计GT背景”与本项目15类忽略背景仍是不同协议。
- UAVid合并car后，原来moving与static互相错分的像素会变为正确，且各类并集不同，必须合并GT和预测后重新算IoU，不能平均两个car IoU再换分母。

## 引用写法

推荐表注：Results for GLoP-SAM3 are quoted from Table I of ROSE-SAM3; the original full text was not consulted.
方法归属引用GLoP原论文；数值来源引用实际已读的ROSE。转引不是ROSE作者复现，不使用reproduced这个标签。期刊全文需要登录与是否可以透明转引是两回事。

## 范围外

方法章节与架构图仍有旧设计描述；本次不重构模型、方法章节或评测实现。
