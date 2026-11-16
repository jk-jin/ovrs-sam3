loss = dict(
    ignore_index=255,
    seg_bce_weight=1.0,
    present_class_weight=0.8,
    seg_dice_weight=0.0,
    eps=1e-6,
    reference_distill_start_iter=1000,
    reference_distill_end_iter=12000,
    reference_distill_start_weight=0.5,
    reference_distill_end_weight=0.0,
    reference_boundary_width=3,
)
