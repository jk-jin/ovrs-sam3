visualization = dict(
    enabled=True,
    save_dir="visualizations",
    save_stage="val",
    alpha=0.45,

    # Basic semantic visualization.
    save_original=True,
    save_prediction=True,
    save_ground_truth=True,
    save_semantic_prediction=True,

    # Semantic prior / final branch score analysis.
    save_score_summary=True,
    save_score_heatmaps=True,
    heatmap_colormap="turbo",

    # Final mixer mask layer visualization.
    save_final_mixer_mask_layers=True,
    save_final_mixer_layer_heatmaps=True,
    save_final_mixer_layer_predictions=True,
    save_final_mixer_layer_overlays=True,

    # None means save all classes.
    # If too many images are produced, change this to 8 or 16.
    max_final_mixer_layer_heatmap_classes=None,

    # CLIP image-text argmax visualization.
    save_clip_argmax_prediction=True,

    # Frozen SAM3 direct segmentation visualization.
    save_sam3_direct_segmentation=True,
    sam3_direct_seg_threshold=0.5,

    # Presence visualization.
    save_presence_scores=True,
    save_presence_layers=True,

    # Sampling control.
    vis_prob=0.05,
    max_samples_per_epoch=100,
    vis_seed=42,

    image_folder_pattern="image_{image_id:06d}",
    ignore_index=255,
)