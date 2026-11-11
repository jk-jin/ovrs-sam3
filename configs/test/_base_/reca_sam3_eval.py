_base_ = [
    "../../_base_/model/reca_sam3.py",
    "../../_base_/loss/semantic.py",
    "../../_base_/runtime.py",
    "../../_base_/evaluation.py",
    "../../_base_/logging.py",
    "../../_base_/visualization.py",
]

model = dict(
    eval_mode=True,
)

experiment = dict(
    name="eval",
    method_id="reca_sam3",
    method_name="ReCA-SAM3",
    variant_id="full",
    implementation_id="reca-sam3-paper-v1",
    kind="evaluation",
    factors=dict(),
)

train_cfg = dict(
    max_iters=0,
    log_window_size=20,
    grad_clip_norm=None,
    save_interval=0,
    eval_interval=0,
    val_max_iters=None,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=1,
)

visualization = dict(
    enabled=True,
    vis_prob=0.02,
    max_samples_per_epoch=100,
)
