_base_ = [
    "./sam3_semantic.py",
]

# Train dataloader is inherited from sam3_semantic.py:
#   train: iSAID
#
# This config only replaces val_dataloader with LoveDA val split:
#   val: LoveDA
#
# It also uses a short sweep schedule:
#   max_iters = 2000
#   warmup    = 200
#   eval      = every 1000 iters

loveda_classes = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agricultural",
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type="data.dataset.OVSemanticSegDataset",
        img_dir="data/datasets/loveDA/img_dir/val",
        ann_dir="data/datasets/loveDA/ann_dir/val",
        classes=loveda_classes,
        img_suffix=".png",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=True,
        return_raw_image=True,
        transforms=[
            dict(type="ToTensor"),
            dict(type="ConvertImageDtype", dtype="float32", scale=True),
            dict(type="Resize", size=(1008, 1008), keep_ratio=False),
            dict(
                type="Normalize",
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ],
    ),
    collate_fn=dict(
        type="data.collate.OVSemanticCollator",
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)

# LoveDA labels after reduce_zero_label=True:
#   original 0 no-data      -> 255 ignore
#   original 1 background   -> 0
#   original 2 building     -> 1
#   ...
# so bg_idx=0 is correct.
eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.0,
    bg_idx=0,
    use_score_map=True,
)

visualization = dict(
    enabled=False,
)

# Sweep schedule.
# Do not use the full 1000-step warmup for short sweep trials.
train_cfg = dict(
    max_iters=2000,
    save_interval=1000,
    eval_interval=1000,

    # Fast cross-dataset validation during sweeps.
    # batch_size=1, so this means 100 LoveDA val images.
    # Set to None or 0 for full LoveDA validation.
    val_max_iters=500,

    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.01,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=20,
    auto_resume=False,
    device="cuda",
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=200,
        end=200,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=1800,
        eta_min=1e-6,
    ),
]