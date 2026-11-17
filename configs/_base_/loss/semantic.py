loss = dict(
    ignore_index=255,
    seg_bce_weight=1.0,
    present_class_weight=0.8,
    seg_dice_weight=0.0,
    eps=1e-6,
    reference_distill_weight=0.5,
)
