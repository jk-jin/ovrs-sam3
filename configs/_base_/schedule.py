param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=12,
    eta_min=1e-6,
)

train_cfg = dict(
    max_epochs=12,
    log_interval=20,
    use_amp=True,
    grad_clip_norm=0.1,
    save_interval=1,
    device='cuda',
)
