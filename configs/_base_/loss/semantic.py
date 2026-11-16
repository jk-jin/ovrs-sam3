loss = dict(
    ignore_index=255,
    seg_bce_weight=1.0,
    absent_class_weight=0.2,
    seg_dice_weight=0.0,
    eps=1e-6,
    reference_distill_weight=0.5,
    reference_boundary_width=3,
)
