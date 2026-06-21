_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/isaid.py",
]

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
        image_intermediate_layers=[7, 15],

        prompt_templates=[
            "a remote sensing image of {}.",
            "a satellite image of {}.",
            "an aerial image of {}.",
            "an overhead image of {}.",
            "a high resolution satellite image of {}.",
            "a high resolution aerial image of {}.",
            "a top down view of {}.",
            "a bird's eye view of {}.",
            "a remote sensing scene containing {}.",
            "a satellite scene containing {}.",
            "an aerial scene containing {}.",
            "an overhead scene containing {}.",
            "a remote sensing image showing {}.",
            "a satellite image showing {}.",
            "an aerial image showing {}.",
            "an overhead image showing {}.",
            "a segmented region of {} in a remote sensing image.",
            "a semantic segmentation mask of {} in a satellite image.",
            "a land cover region of {}.",
            "a land use region of {}.",
            "a large area of {} in an aerial image.",
            "a small area of {} in a satellite image.",
            "dense {} in a remote sensing image.",
            "sparse {} in a remote sensing image.",
            "the boundary of {} in a satellite image.",
            "the texture of {} in an aerial image.",
            "the shape of {} from an overhead view.",
            "the pattern of {} in a remote sensing scene.",
            "{} in urban remote sensing imagery.",
            "{} in rural remote sensing imagery.",
            "{} on the ground surface from above.",
            "{} visible from satellite imagery.",
        ],

        normalize_label_for_clip=True,
    ),

    template_guided_refiner_cfg=dict(
        enabled=True,

        hidden_dim=256,
        num_prompt_templates=32,

        lowres_hw=18,
        lowres_layers=4,

        highres_hw=72,
        highres_layers=2,

        num_heads=8,
        dropout=0.1,

        window_size=9,
        shift_size=4,

        use_checkpoint=True,
        early_prompt_attention=False,
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.template_guided_refiner",
        ],
        frozen_modules=[],
        openclip_text_finetune="frozen",
        openclip_image_finetune="frozen",
    ),

    adapter_cfg=dict(),

    criterion_cfg=dict(
        ignore_index=255,

        final_bce_weight=1.0,
        final_dice_weight=0.0,

        bce_absent_class_weight=0.0,

        bce_valid_pixel_weight=1.0,
        bce_ignore_pixel_weight=0.05,

        eps=1e-6,
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
)

eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.0,
    bg_idx=0,
    use_score_map=True,
)

optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                "core.template_guided_refiner": dict(
                    lr_mult=4.0,
                    decay_mult=1.0,
                ),

                # RSKT-Seg: CLIP_MULTIPLIER = 0.01
                "core.clip_text_encoder": dict(
                    lr_mult=0.01,
                    decay_mult=0.0,
                ),
                "core.clip_image_encoder": dict(
                    lr_mult=0.01,
                    decay_mult=0.0,
                ),
            },
        ),
    )
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=1000,
        end=0,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=19000,
        eta_min=1e-6,
    )
]

train_cfg = dict(
    max_iters=20000,
    save_interval=1000,
    eval_interval=20000,
    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.01,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=20,
    auto_resume=False,
    device="cuda",
)

tta_cfg = dict(
    enabled=False,
    scales=[0.75, 1.0, 1.25],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)
