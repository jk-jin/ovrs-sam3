_base_ = [
    "./ovrs_sam3_isaid_loveda_base.py",
]

# -------------------------------------------------------------------------
# Full training config: 20000 steps, eval every 2000 steps.
# Local visualization enabled, W&B enabled (scalars only, no images).
# Checkpoints saved every 2000 steps.
# -------------------------------------------------------------------------

train_cfg = dict(
    max_iters=20000,
    save_interval=2000,
    eval_interval=2000,

    # batch_size=1, so this validates about 500 LoveDA images.
    val_max_iters=500,

    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.01,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=10,
    auto_resume=False,
    device="cuda",
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=1000,
        end=1000,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=19000,
        eta_min=1e-6,
    ),
]

# Local visualization only.
# Current WandbHook does not upload images.
visualization = dict(
    enabled=True,
    save_dir="visualizations",
    save_stage="val",
    alpha=0.45,

    save_original=True,
    save_prediction=True,
    save_raw_final_prediction=False,
    save_ground_truth=True,
    save_semantic_prediction=True,

    # Keep score visualizations off by default for full training
    # to avoid writing too many large files.
    save_score_summary=False,
    save_score_heatmaps=False,
    heatmap_colormap="turbo",

    save_sam3_direct_segmentation=False,

    vis_prob=0.02,
    max_samples_per_epoch=20,
    vis_seed=42,

    image_folder_pattern="image_{image_id:06d}",
    ignore_index=255,
)

experiment_tracking = dict(
    metrics_jsonl=dict(
        enabled=True,
        filename="metrics.jsonl",
        train_interval=20,
        val_interval=1,
        priority=80,
    ),
    wandb=dict(
        enabled=True,
        project="ovrs-sam3",
        name="full_isaid_loveda_final_v1",
        group="full_isaid_loveda",
        tags=[
            "full",
            "isaid-train",
            "loveda-val",
            "final-design",
            "fixed-32-templates",
            "score-dim-64",
        ],
        mode="online",
        train_interval=20,
        log_val_iter=False,
        priority=90,
        name_from_config_keys=[],
        name_prefix=None,
    ),
)