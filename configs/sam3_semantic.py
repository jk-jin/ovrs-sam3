_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/loveda.py",
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
            "an aerial image of {}.",
            "a satellite image of {}.",
            "an overhead view of {}.",

            "an overhead remote sensing image of {}.",
            "a high-resolution aerial image of {}.",
            "a top-down satellite view of {}.",
            "a remote sensing scene containing {}.",
        ],
        num_prompt_templates=8,
        normalize_label_for_clip=True,
    ),

    final_mixer_cfg=dict(
        enabled=True,

        fusion_layers=4,
        num_heads=8,
        dropout=0.1,

        dynamic_prompt_cfg=dict(
            tokens_per_template=4,
        ),

        lowres_cfg=dict(
            hidden_dim=256,
            score_embed_dim=32,
            window_size=8,
            shift_size=4,
        ),

        upsampler_cfg=dict(
            class_chunk_size=4,
            decoder_channels=[256, 128, 96, 64, 32],
            sam_guidance_channels=[32, 24, 16, 8],

            clip_guidance_channels=[32, 24],
            clip_guidance_stage_indices=[0, 1],

            upsample_mode="bilinear",
            norm="group_norm",
            act="gelu",
        ),
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.final_mixer",
        ],
        frozen_modules=[],
    ),

    adapter_cfg=dict(),

    criterion_cfg=dict(
        ignore_index=255,

        final_bce_weight=1.0,
        final_dice_weight=0.0,

        # 0.0 = absent classes not supervised for mask BCE.
        # Set to 0.01 / 0.05 for mild absent-class suppression.
        bce_absent_class_weight=0.0,

        # BCE pixel weights:
        # valid pixels keep full supervision;
        # ignore pixels get weaker suppression to avoid over-penalizing unlabeled regions.
        bce_valid_pixel_weight=5.0,
        bce_ignore_pixel_weight=0.1,

        eps=1e-6,
    ),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
)

eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.5,
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
                "core.final_mixer": dict(
                    lr_mult=4.0,
                    decay_mult=1.0,
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
    grad_clip_norm=0.1,
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