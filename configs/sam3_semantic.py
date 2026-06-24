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

        prompt_template="a remote sensing image of {}.",
        normalize_label_for_clip=True,
    ),

    encoder_refiner_cfg=dict(
        enabled=True,

        ablation_variant="baseline",

        num_query_tokens=32,
        fusion_layers=4,
        num_heads=8,
        dropout=0.1,

        hidden_dim=256,

        clip_score_embed_dim=128,
        clip_score_conv_kernel=7,

        # Experiment 1: score embed text source
        score_embed_source="learned_query",
        fixed_score_templates=[
            "a satellite image of {}.",
            "an aerial image of {}.",
            "a remote sensing image of {}.",
            "a high resolution satellite view of {}.",
            "a top-down aerial view of {}.",
            "an overhead view of {}.",
            "a geospatial image showing {}.",
            "a satellite scene containing {}.",

            "a remote sensing segmentation region of {}.",
            "a land cover area of {}.",
            "a land use region of {}.",
            "a mapped region corresponding to {}.",
            "a visible object region of {} from above.",
            "a distinct terrain region of {}.",
            "a spatial area labeled as {}.",
            "a geographic region occupied by {}.",

            "the texture pattern of {} in a satellite image.",
            "the color and texture of {} from an aerial view.",
            "the shape of {} seen from above.",
            "the boundary of {} in remote sensing imagery.",
            "the dense spatial pattern of {}.",
            "the sparse spatial pattern of {}.",
            "the large scale appearance of {}.",
            "the small scale appearance of {}.",

            "urban remote sensing imagery of {}.",
            "rural remote sensing imagery of {}.",
            "a natural landscape region of {}.",
            "a man-made structure region of {}.",
            "a transportation related area of {}.",
            "a vegetation or land surface region of {}.",
            "a water or impervious surface context containing {}.",
            "a complex aerial scene with {}.",
        ],

        # Experiment 2: score embed upsample fuse CLIP mid features
        score_upsample_fuse_clip_mid=False,
        score_mid_proj_dim=64,
        clip_mid_native_dim=1024,
        clip_mid_layer_for_36=15,
        clip_mid_layer_for_72=7,

        # Experiment 3: window attention scales
        window_attention_scales=[36, 18],

        # Experiment 4: class attention q/k context
        class_attention_context="sam_text_score",

        # Experiment 5: spatial upsample fuse SAM3 FPN
        spatial_upsample_fuse_sam_fpn=False,
        sam_fpn_fuse_proj_dim=64,

        encoder_hw=72,
        score_base_hw=18,
        window_size=9,
        shift_size=4,

        use_checkpoint=True,
        early_prompt_attention=False,
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.encoder_refiner",
        ],
        frozen_modules=[],

        # Text side:
        #   frozen      = freeze OpenCLIP text encoder
        #   attention   = train text attention q/v + positional embedding
        #   transformer = train all text transformer params
        #   full        = train all OpenCLIP text encoder params
        openclip_text_finetune="frozen",

        # Image side:
        #   frozen      = freeze OpenCLIP image encoder
        #   attention   = train visual attention q/v + visual positional embedding
        #   transformer = train all visual transformer params
        #   full        = train all OpenCLIP visual params
        openclip_image_finetune="frozen",
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
        bce_valid_pixel_weight=1.0,
        bce_ignore_pixel_weight=0.01,

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
                "core.encoder_refiner": dict(lr_mult=4.0, decay_mult=1.0),

                # OpenCLIP text q/v or full text fine-tune.
                # 1e-4 × 0.02 = 2e-6
                "core.clip_text_encoder": dict(lr_mult=0.02, decay_mult=0.0),

                # OpenCLIP image q/v or full image fine-tune.
                # Conservative default; can be swept later.
                # 1e-4 × 0.01 = 1e-6
                "core.clip_image_encoder": dict(lr_mult=0.01, decay_mult=0.0),
            },
        ),
    )
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=1000,
        end=1000,
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

        # Required for W&B sweep auto naming via --cfg-options.
        name_from_config_keys=[],
        name_prefix=None,
    ),
)

tta_cfg = dict(
    enabled=False,
    scales=[0.75, 1.0, 1.25],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)