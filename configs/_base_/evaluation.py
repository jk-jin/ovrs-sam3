eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.1,
)

tta_cfg = dict(
    enabled=False,
    scales=[1.0],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)
