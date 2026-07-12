_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/isaid.py",
]

# -------------------------------------------------------------------------
# Final model: OVRS-SAM3 + frozen RemoteCLIP + 32 fixed templates
# Train: iSAID
# Val:   LoveDA
# -------------------------------------------------------------------------

model = dict(
    task_mode="semantic",
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="weights/sam3.pt",
    load_from_hf=False,
    device="cuda",
    eval_mode=False,
    compile=False,
    prompt_chunk_size=8,

    openclip_cfg=dict(
        enabled=True,
        model_name="ViT-L-14",
        pretrained="weights/RemoteCLIP-ViT-L-14.pt",
        default_output="feat_map",
        image_size=504,
        image_intermediate_layers=[7, 15],

        prompt_templates=[
            "a remote sensing image of {}.",
            "a satellite image of {}.",
            "an aerial image of {}.",
            "a high-resolution overhead image of {}.",
            "a top-down view of {}.",
            "a bird's-eye view image of {}.",
            "a remote sensing scene containing {}.",
            "a satellite scene containing {}.",
            "an aerial scene containing {}.",
            "a high-resolution remote sensing scene of {}.",
            "a land cover region of {} in a satellite image.",
            "a land use area of {} in an aerial image.",
            "a semantic segmentation region of {}.",
            "a labeled mask region corresponding to {}.",
            "a continuous area of {} in overhead imagery.",
            "a visible region of {} from above.",
            "the texture pattern of {} in a satellite image.",
            "the spatial pattern of {} in remote sensing imagery.",
            "the shape and boundary of {} in an aerial image.",
            "the object boundary of {} from an overhead view.",
            "a small-scale remote sensing object of {}.",
            "a large-scale remote sensing region of {}.",
            "multiple instances of {} in a satellite image.",
            "dense objects of {} in overhead imagery.",
            "sparse objects of {} in remote sensing imagery.",
            "urban remote sensing imagery showing {}.",
            "rural remote sensing imagery showing {}.",
            "natural land surface containing {}.",
            "man-made structures containing {}.",
            "a homogeneous area of {}.",
            "a complex background with {}.",
            "an object or region classified as {} in remote sensing imagery.",
        ],
        normalize_label_for_clip=True,
        text_prompt_batch_size=64,
        text_prompt_use_checkpoint=True,
    ),

    encoder_refiner_cfg=dict(
        enabled=True,
        fusion_layers=4,
        num_heads=8,
        dropout=0.1,
        hidden_dim=256,

        score_embed_dim=256,
        layer_scale_init=0.1,

        refiner_hw=36,
        encoder_hw=72,

        window_size=12,
        shift_size=6,

        use_checkpoint=True,
        early_prompt_attention=False,
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.encoder_refiner",
        ],
        frozen_modules=[],
        openclip_text_finetune="attention",
        openclip_image_finetune="attention",
    ),

    adapter_cfg=dict(),

    criterion_cfg=dict(
        ignore_index=255,
        final_bce_weight=1.0,
        final_dice_weight=0.0,
        bce_absent_class_weight=0.05,
        bce_valid_pixel_weight=1.0,
        bce_ignore_pixel_weight=0.05,
        eps=1e-6,
    ),
)

# -------------------------------------------------------------------------
# Dataloaders
# Train dataloader is inherited from configs/datasets/isaid.py.
# Val dataloader is LoveDA val split.
# -------------------------------------------------------------------------

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
)

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
        background_cfg=dict(
            enabled=True,
            class_id=0,
            class_name="background",
            exclude_from_forward=True,
        ),
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
# Background class id=0 is declared in dataset.background_cfg.
eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.1,
)

# -------------------------------------------------------------------------
# Optimizer
# Only encoder_refiner is trainable in the final design.
# -------------------------------------------------------------------------

optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                "core.encoder_refiner": dict(lr_mult=1.0, decay_mult=1.0),
                "core.clip_text_encoder": dict(lr_mult=0.01, decay_mult=1.0),
                "core.clip_image_encoder": dict(lr_mult=0.01, decay_mult=1.0),
            },
        ),
    )
)

# -------------------------------------------------------------------------
# Default schedule: short experiment-friendly defaults.
# Full training overrides these in ovrs_sam3_isaid_loveda_full.py.
# -------------------------------------------------------------------------

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=400,
        end=400,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=3600,
        eta_min=1e-6,
    ),
]

train_cfg = dict(
    max_iters=4000,
    save_interval=1000,
    eval_interval=1000,
    val_max_iters=500,

    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.01,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=5,
    auto_resume=False,
    device="cuda",
)

# Default: no local visualization in short experiments.
visualization = dict(
    enabled=False,
)

# Default: local metrics on, W&B off.
# Exp/full configs override wandb.enabled=True.
experiment_tracking = dict(
    metrics_jsonl=dict(
        enabled=True,
        filename="metrics.jsonl",
        train_interval=20,
        val_interval=1,
        priority=80,
    ),
    wandb=dict(
        enabled=False,
        project="ovrs-sam3",
        name=None,
        group=None,
        tags=[],
        mode="online",
        train_interval=20,
        log_val_iter=False,
        priority=90,
        name_from_config_keys=[],
        name_prefix=None,
    ),
)

tta_cfg = dict(
    enabled=False,
    scales=[1.0],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)