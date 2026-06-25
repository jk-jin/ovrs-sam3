_base_ = [
    "./ovrs_sam3_isaid_loveda_base.py",
]

# -------------------------------------------------------------------------
# Short experiment config: 4000 steps, eval every 1000 steps.
# W&B enabled, local visualization disabled.
# Used as the base for all sweep experiments.
# -------------------------------------------------------------------------

train_cfg = dict(
    max_iters=4000,
    save_interval=1000,
    eval_interval=1000,
    val_max_iters=500,

    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.01,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=5,
    auto_resume=False,
    device="cuda",
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=400,
        end=400,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=3600,
        eta_min=1e-6,
    ),
]

visualization = dict(
    enabled=False,
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
        name=None,
        group="exp_isaid_loveda",
        tags=[
            "exp",
            "isaid-train",
            "loveda-val",
            "refiner-36x36",
            "fixed-32-templates",
            "score-dim-256",
        ],
        mode="online",
        train_interval=20,
        log_val_iter=False,
        priority=90,
        name_prefix="exp",
        name_from_config_keys=[],
    ),
)