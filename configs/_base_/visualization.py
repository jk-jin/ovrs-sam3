visualization = dict(
    enabled=True,
    save_dir="visualizations",
    save_stage="val",
    alpha=0.45,

    save_original=True,
    save_prediction=True,
    save_ground_truth=True,
    save_semantic_prediction=True,

    save_score_summary=True,
    save_score_heatmaps=True,
    heatmap_colormap="turbo",

    save_sam3_direct_segmentation=True,
    sam3_direct_seg_threshold=0.5,

    vis_prob=0.05,
    max_samples_per_epoch=100,
    vis_seed=42,

    image_folder_pattern="image_{image_id:06d}",
    ignore_index=255,
)