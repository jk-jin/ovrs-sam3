evaluation = dict(
    ignore_index=255,
    background_prob_threshold=0.0,
    metric_groups=[],
    geometry=dict(
        input_hw=[1008, 1008],
        keep_ratio=False,
    ),
    tta=dict(
        enabled=False,
        scales=[1.0],
        flip_modes=["none", "h", "v"],
        size_divisor=14,
    ),
    postprocess=dict(
        reference_prob_threshold=0.1,
        prediction_prob_threshold=0.0,
    ),
)
