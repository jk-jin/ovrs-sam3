# ReCA-SAM3 命名迁移对照

本次工程整理把项目自研模块、配置、张量、日志和文档统一为 ReCA-SAM3 命名。SAM3 / OpenCLIP 原生模块和预训练权重加载所依赖的内部名称保持不变。

## 1. 模块与文件

| 旧名称 | 新名称 | 新文件 |
| --- | --- | --- |
| `SAM3Segmentor` | `ReCASAM3` | `models/reca_sam3.py` |
| `ClipScoreEmbedding` | `RemoteCLIPGuidanceEmbedding` | `models/remoteclip_guidance.py` |
| `ClassConditionedEncoderRefiner` | `RemoteCLIPGuidedAggregation` | `models/feature_aggregation.py` |
| `RefinerInterBlock` | `CrossClassAggregationBlock` | `models/aggregation_attention.py` |
| `RefinerIntraBlock` | `IntraClassAggregationBlock` | `models/aggregation_attention.py` |
| `ClassScoreAttention` | `CrossClassAttention` | `models/aggregation_attention.py` |
| `WindowScoreAttention` | `WindowSpatialAttention` | `models/aggregation_attention.py` |
| `DecoderInputFusion` | `SemanticDetailFusion` | `models/semantic_detail_fusion.py` |
| 四方向残差函数 | `aggregate_orientations` | `models/semantic_detail_fusion.py` |

## 2. 配置

| 旧字段 | 新字段 |
| --- | --- |
| `encoder_refiner_cfg` | `model.aggregation` |
| `openclip_cfg` | `model.remoteclip` |
| `freeze_cfg` | `model.freeze` |
| `criterion_cfg` | 根 `loss` |
| `fusion_layers` | `aggregation.num_blocks` |
| `hidden_dim` | `aggregation.feature_dim` |
| `score_embed_dim` | `aggregation.semantic_dim` |
| `refiner_hw` | `aggregation.aggregation_hw` |
| `encoder_hw` | `aggregation.encoder_hw` |
| `final_balanced_bce_weight` | `loss.seg_bce_weight` |
| `final_dice_weight` | `loss.seg_dice_weight` |
| `sam3_mask_distill_weight` | `loss.reference_distill_weight` |
| `sam3_mask_distill_boundary_width` | `loss.reference_boundary_width` |
| `present_ignore_suppress_weight` | `loss.present_ignore_suppression_weight` |
| `sam3_mask_prob_thd` | `evaluation.postprocess.reference_prob_threshold` |
| `final_mask_prob_thd` | `evaluation.postprocess.prediction_prob_threshold` |
| `eval_cfg.prob_thd` | `evaluation.background_prob_threshold` |
| `openclip_text_finetune` | `freeze.remoteclip_text_finetune` |
| `openclip_image_finetune` | `freeze.remoteclip_image_finetune` |

新增根配置：`experiment`、`training`、`evaluation`、`runtime`、`logging`。
`runtime` 是 seed/device/AMP/work_dir 的唯一来源；`configs/_base_/tracking.py` 已删除，日志配置移入 `configs/_base_/logging.py`。

## 3. 张量与输出键

| 旧名称 | 新名称 |
| --- | --- |
| `clip_score_maps_36` | `clip_template_scores_36` |
| `clip_score_embed_36` | `clip_guidance_36` |
| `score_embed_36` | `semantic_state_36` |
| `feature_36` | `feature_state_36` |
| `base_feature_36` | `initial_feature_36` |
| `refiner_features_36` | `aggregated_features_36` |
| `cross_attended_encoder_features_72` | `prompt_conditioned_features_72` |
| `clip_mid_36` | `clip_spatial_guidance_36` |
| `sam_fpn_36` | `sam3_spatial_guidance_36` |
| `fused_feature_72` | `decoder_features_72` |
| `sam3_direct_logits` | `sam3_reference_logits` |
| `raw_sam3_prompt_score_map` | `raw_reference_prompt_score_map` |
| `raw_sam3_score_map` | `raw_reference_score_map` |

保留：`final_logits`、`raw_prompt_score_map`、`raw_final_score_map`、`final_score_map`、`sam3_teacher_logits`。

## 4. 损失日志

| 旧键 | 新键 |
| --- | --- |
| `loss_final_bce` | `loss_seg_bce` |
| `loss_final_dice` | `loss_seg_dice` |
| `loss_sam3_mask_distill_bce` | `loss_reference_distill_bce` |
| `loss_sam3_mask_distill_weighted` | `loss_reference_distill_weighted` |
| `loss_present_ignore_suppress_bce` | `loss_present_ignore_suppression_bce` |
| `loss_present_ignore_suppress_weighted` | `loss_present_ignore_suppression_weighted` |

`total_loss` 不变。

## 5. 训练入口与实验目录

| 旧入口 | 新入口 |
| --- | --- |
| `configs/train/isaid_loveda_full.py` | `configs/experiments/main/isaid_loveda_full.py` |
| `configs/train/isaid_loveda_exp.py` | `configs/experiments/dev/isaid_loveda_4k.py` |
| `configs/test/_base_/ovrs_sam3_eval.py` | `configs/test/_base_/reca_sam3_eval.py` |

旧训练权重与旧结果目录不再兼容：

* 参数路径随模块、配置和子模块重命名整体变化，旧 checkpoint 不能 `--resume-from`。
* 不提供旧参数映射、自动转换或兼容别名。
* 新实验必须使用新的 work directory。
* checkpoint 容器 schema 仍为 4，但模型参数路径已改变。
* 新训练目录中的 `config.json` 与 `results/*.json` 使用新格式；旧 `metrics.jsonl` 不会被自动当作最终成绩。

## 6. 论文导出

新工具：

```bash
python tools/collect_experiments.py work_dirs/paper --output paper/generated/data/results_long.csv
python tools/import_results.py --input external/baseline.csv --mapping external/mapping.json \
  --protocol external/protocol.json --input-unit percent --output-dir work_dirs/paper/imported
python tools/export_paper_results.py --root work_dirs/paper \
  --spec paper/export_config.json --output paper/generated
```

旧 `tools/collect_experiments.py` 按 JSONL 最高验证点猜测成绩的行为已删除；标准结果以 `results/<eval_id>.json` 为准。
