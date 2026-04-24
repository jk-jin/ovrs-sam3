param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=40000,
    eta_min=1e-6,
)

train_cfg = dict(
    max_iters=40000,
    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.1,
    save_interval=2000,
    eval_interval=2000,
    monitor='semantic.miou',
    monitor_mode='max',
    max_keep_ckpts=10,
    auto_resume=False,
    device='cuda',
)