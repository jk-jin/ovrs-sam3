_base_ = [
    "../../_base_/model/ovrs_sam3.py",
    "../../_base_/runtime.py",
    "../../_base_/evaluation.py",
    "../../_base_/tracking.py",
    "../../_base_/visualization.py",
]

model = dict(
    eval_mode=True,
)

train_cfg = dict(
    max_iters=0,
    log_window_size=20,
    use_amp=True,
    grad_clip_norm=None,
    save_interval=0,
    eval_interval=0,
    val_max_iters=None,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=1,
    auto_resume=False,
    device="cuda",
)
