# 对比方法与文献来源核对稿（2026-11-24）

当前选择已用于论文组表；待解决事项见 [comparison_issues_20261124.md](comparison_issues_20261124.md)。排除VLex-SAM3、ConceptSeg。数值为mIoU (%)，--为未报告或待跑。

## 主比较：两组

| Method | Data / condition | DLRSD | iSAID | LoveDA | Potsdam | Vaihingen | UAVid | UDD5 | VDD | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Training on remote sensing segmentation data** | | | | | | | | | | |
| SCAN | DLRSD | 52.42 | 44.28 | 23.17 | 27.45 | 15.23 | 20.28 | 34.14 | 29.24 | R |
| SAN | DLRSD | 86.45 | 49.56 | 25.33 | 37.25 | 39.22 | 23.53 | 37.23 | 35.83 | R |
| SED | DLRSD | 87.68 | 51.23 | 24.55 | 29.35 | 39.02 | 21.33 | 35.73 | 32.53 | R |
| CAT-Seg | DLRSD | 88.68 | 53.34 | 28.64 | 35.75 | 42.30 | 25.73 | 40.24 | 39.14 | R |
| OVRS | DLRSD | 88.85 | 52.65 | 31.53 | 36.44 | 43.50 | 24.13 | 40.82 | 37.23 | R |
| GSNet | DLRSD | 86.02 | 53.73 | 32.52 | 37.85 | 44.13 | 24.22 | 40.92 | 37.34 | R |
| RSKT-Seg | DLRSD | 93.49 | 54.32 | 33.23 | 38.44 | 42.69 | 25.72 | 42.10 | 39.69 | R |
| Pi-Seg | DLRSD | -- | 53.50 | 39.00 | 35.33 | 44.18 | 33.58 | 59.08 | 47.73 | I |
| UGDMoE | DLRSD | 93.35 | 54.38 | 34.17 | 38.21 | 44.92 | 26.85 | 43.78 | 40.12 | U |
| SkySense-O | Sky-SA | -- | 43.92 | 38.30 | 54.10 | 51.64 | -- | -- | -- | H |
| ReCA-SAM3 | DLRSD | -- | -- | -- | -- | -- | -- | -- | -- | N |
| **Training-free** | | | | | | | | | | |
| SCLIP | None | -- | 16.10 | 30.40 | 39.60 | 35.90 | 31.40 | 38.70 | 37.90 | S |
| ClearCLIP | None | -- | 18.20 | 32.40 | 42.00 | 36.20 | 36.20 | 41.80 | 39.30 | S |
| ProxyCLIP | None | -- | 21.80 | 34.30 | 49.00 | 47.50 | 35.80 | 40.80 | 47.80 | S |
| CorrCLIP | None | -- | 25.50 | 36.90 | 51.90 | 47.00 | 38.30 | 46.10 | 47.30 | S |
| SegEarth-OV | None | -- | 21.7 | 36.9 | 47.1 | 29.1 | 42.5 | 50.6 | 45.3 | E |
| ConInfer | None | -- | 20.08 | 39.33 | 49.99 | 31.37 | 46.40 | 46.86 | 50.29 | C |
| ActiveSAM (ProC-SAM3 reproduction) | None | -- | 28.30 | 41.10 | 40.60 | 45.10 | 53.30 | 72.10 | 62.20 | P |
| SegEarth-OV3 | None | -- | 27.6 | 47.4 | 57.8 | 60.8 | 54.7 | 71.7 | 64.5 | S |
| RSGPNet (own paper) | None | -- | -- | 47.80 | 59.70 | 61.10 | -- | 71.00 | -- | J |
| GLoP-SAM3 (ROSE report) | None | -- | 28.00 | 46.20 | 57.90 | 62.20 | 55.40 | 72.60 | 64.70 | O |
| ProC-SAM3 | None | -- | 29.5 | 46.1 | 58.8 | 64.5 | 60.0 | 73.4 | 67.5 | P |
| ROSE-SAM3 | Unlabeled | -- | 35.10 | 48.70 | 58.90 | 65.30 | 59.90 | 73.70 | 67.50 | O |
| SAM 3 Semantic-only | None | -- | -- | -- | -- | -- | -- | -- | -- | N |
| SAM 3 Semantic × Presence | None | -- | -- | -- | -- | -- | -- | -- | -- | N |

## 不同适配／支持条件参考

| Method | Data / condition | DLRSD | iSAID | LoveDA | Potsdam | Vaihingen | UAVid | UDD5 | VDD | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Test-time adaptation** | | | | | | | | | | |
| GFMR (full) | Online GN | -- | 29.9 | 48.3 | 58.4 | 64.3 | 55.3 | 72.7 | 66.2 | F |
| **Support-assisted frozen inference** | | | | | | | | | | |
| ConceptBank (v1, GT support) | GT support | -- | 35.4 | 49.4 | 60.5 | 63.0 | -- | -- | -- | B |
| **Frozen inference; prototype supervision unresolved** | | | | | | | | | | |
| PRS-Seg | Unspecified | -- | -- | 48.1 | 53.8 | 59.7 | -- | -- | -- | A |

## 数值来源

- **R**: [arXiv:2509.12040v3](https://arxiv.org/pdf/2509.12040v3), PDF p.6, Table 1; SHA-256 `6fe6b20ccb7f9fc3d2303da2511792a1cb7d0da650adb2fbc188b6ebf6b28701`.
- **I**: [arXiv:2604.15652v3](https://arxiv.org/pdf/2604.15652v3), PDF p.5, Table 1; SHA-256 `83e679e4efedbcdebb371d339ec696a77562d108f2983a053f87b72b0653f85b`.
- **H**: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_SkySense-O_Towards_Open-World_Remote_Sensing_Interpretation_with_Vision-Centric_Visual-Language_Modeling_CVPR_2025_paper.pdf), PDF p.6, Table 2; SHA-256 `0d82cbffff30737597b785d2d0a4da03c127d61d826b23e477b3feb2cecdcc58`.
- **S**: [arXiv:2512.08730v2](https://arxiv.org/pdf/2512.08730v2), PDF p.6, Table 1; SHA-256 `b7cf3e68414a1eadd34709263f4ee137725b813766491c79bcf8645473bed930`.
- **C**: [arXiv:2603.29271v1](https://arxiv.org/pdf/2603.29271v1), PDF p.6, Table 1; SHA-256 `da2728ada151caeb4915bb745757d6fabf12289ebce53b8ff80bb93c45085881`.
- **P**: [arXiv:2606.21863v2](https://arxiv.org/pdf/2606.21863v2), PDF p.4, Table I; SHA-256 `f21de49839da3a8c7b8e29bbd5be74686d6d5830b44359dba0908c82b89ee11d`.
- **J**: [arXiv:2606.28410v1](https://arxiv.org/pdf/2606.28410v1), PDF p.6, Table II; SHA-256 `1b4c9ea42fc7f3ccebad4be11e99d7c9f95989633510b01513d1cfaf3b314418`.
- **O**: [arXiv:2611.28112v1](https://arxiv.org/pdf/2611.28112v1), PDF p.7, Table I; SHA-256 `109d1df4bb393a0b3423ec657abd65bef7e6143b13765db5b051d91341165b5d`.
- **F**: [arXiv:2610.04300v1](https://arxiv.org/pdf/2610.04300v1), PDF p.15, Table 1; SHA-256 `317e68e5ada8c1b82b7cb90ace7d7de509b483808ab67d39938e170a08267233`.
- **B**: [arXiv:2602.06333v1 (GT support)](https://arxiv.org/pdf/2602.06333v1), PDF p.6, Table 2; SHA-256 `06a4d763f732938e674c619e53fda34b7321387ae05fbb259cceb226f6e7a39a`.
- **U**: [UGDMoE publisher HTML Table 2](https://www.mdpi.com/2072-4292/18/9/1349), not a downloaded PDF.
- **A**: [PRS-Seg publisher abstract](https://ieeexplore.ieee.org/abstract/document/11739897), not an inspected full-text table.
- **E**: SegEarth-OV CVPR Table 1; source fingerprint retained in literature_results.csv.
- **N**: local experiment pending.

## 方法与引用

- **SCAN**: Open-Vocabulary Segmentation with Semantic-Assisted Calibration; [liu2024openvocabularysegmentationsemanticassistedcalibration](https://arxiv.org/abs/2312.04089)
- **SAN**: Side Adapter Network for Open-Vocabulary Semantic Segmentation; [xu2023adapternetworkopenvocabularysemantic](https://arxiv.org/abs/2302.12242)
- **SED**: SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation; [xie2024sedsimpleencoderdecoderopenvocabulary](https://arxiv.org/abs/2311.15537)
- **CAT-Seg**: CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation; [choCATSegCostAggregation2024](https://doi.org/10.1109/CVPR52733.2024.00394)
- **OVRS**: Open-Vocabulary High-Resolution Remote Sensing Image Semantic Segmentation; [caoOVRSHighResolution2025](https://ieeexplore.ieee.org/document/10962188/)
- **GSNet**: Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation; [yeOpenVocabularyRemoteSensing2025](https://doi.org/10.1609/aaai.v39i9.33022)
- **RSKT-Seg**: Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing; [liExploringEfficientOpenVocabulary2026](https://doi.org/10.1609/aaai.v40i8.37521)
- **Pi-Seg**: Perturbation-Regularized Open-Vocabulary Remote Sensing Segmentation with Unified Multi-Domain Evaluation; [liPerturbationRegularizedOpenVocabularyRemote2026](https://doi.org/10.48550/arXiv.2604.15652)
- **UGDMoE**: UGDMoE: An Uncertainty-Guided Mixture-of-Experts Decoder for Open-Vocabulary Remote Sensing Segmentation; [quUGDMoE2026](https://doi.org/10.3390/rs18091349}, url = {https://www.mdpi.com/2072-4292/18/9/1349)
- **SkySense-O**: SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling; [zhuSkySenseOOpenWorldRemote2025a](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_SkySense-O_Towards_Open-World_Remote_Sensing_Interpretation_with_Vision-Centric_Visual-Language_Modeling_CVPR_2025_paper.pdf)
- **SCLIP**: SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference; [wang2024scliprethinkingselfattentiondense](https://arxiv.org/abs/2312.01597)
- **ClearCLIP**: ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference; [lan2024clearclipdecomposingcliprepresentations](https://arxiv.org/abs/2407.12442)
- **ProxyCLIP**: ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation; [lan2024proxyclipproxyattentionimproves](https://arxiv.org/abs/2408.04883)
- **CorrCLIP**: CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation; [zhang2025corrclipreconstructingpatchcorrelations](https://arxiv.org/abs/2411.10086)
- **SegEarth-OV**: SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images; [liSegEarthOVTrainingFreeOpenVocabulary2025](https://doi.org/10.1109/CVPR52734.2025.00986)
- **ConInfer**: ConInfer: Context-Aware Inference for Training-Free Open-Vocabulary Remote Sensing Segmentation; [chen2026coninfercontextawareinferencetrainingfree](https://arxiv.org/abs/2603.29271)
- **ActiveSAM (ProC-SAM3 reproduction)**: ActiveSAM: Fast and Accurate Open-Vocabulary Semantic Segmentation with Frozen SAM 3; [tienActiveSAMFastAccurate2026](https://doi.org/10.48550/arXiv.2606.16996)
- **SegEarth-OV3**: SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images; [liSegEarthOV3ExploringSAM2026](https://doi.org/10.48550/arXiv.2512.08730)
- **RSGPNet (own paper)**: RSGPNet: Geometric Prompting for Remote Sensing Open-Vocabulary Semantic Segmentation; [wangRSGPNetGeometricPrompting2026](https://doi.org/10.1109/JSTARS.2026.3725927)
- **GLoP-SAM3 (ROSE report)**: GLoP-SAM3: A Dual Postprocessing Framework With Global--Local Prior Fusion and Reliability Gating for Open-Vocabulary Remote Sensing Segmentation; [liuGLoPSAM3DualPostprocessing2026](https://doi.org/10.1109/TGRS.2026.3713962)
- **ProC-SAM3**: Prompt-Calibrated SAM 3 for Open-Vocabulary Remote-Sensing Semantic Segmentation; [songPromptCalibratedSAM32026](https://doi.org/10.1109/LGRS.2026.3713378)
- **ROSE-SAM3**: ROSE-SAM3: Response-Oriented Prompt Selection for Open-Vocabulary Remote Sensing Segmentation; [songROSESAM3ResponseOrientedPrompt2026](https://doi.org/10.48550/arXiv.2611.28112)
- **GFMR (full)**: A Geometric-Transformation Feature-Adaptive Manifold Restoration Method for Open-Vocabulary Semantic Segmentation of Remote Sensing Images; [wangGeometricTransformationFeatureAdaptiveManifold2026](https://doi.org/10.48550/arXiv.2610.04300)
- **ConceptBank (v1, GT support)**: Taming SAM3 in the Wild: A Concept Bank for Open-Vocabulary Segmentation; [peiTamingSAM3Wild2026](https://doi.org/10.48550/arXiv.2602.06333)
- **PRS-Seg**: PRS-Seg: Open-Vocabulary Remote Sensing Segmentation with Prototype Calibration and Boundary Refinement; [tangPRSSegOpenVocabularyRemote2026](https://doi.org/10.1109/ICCS71313.2026.11739897)
