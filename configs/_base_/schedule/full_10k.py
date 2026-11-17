train_cfg = dict(
    max_iters=10000,
    save_interval=1000,
    eval_interval=1000,
    val_max_iters=None,

    log_window_size=20,
    grad_clip_norm=0.01,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=10,
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=500,
        end=500,
    ),
    dict(
        type="RelativeCosineAnnealingLR",
        T_max=9500,
        min_lr_ratio=0.01,
        end=10000,
    ),
]
