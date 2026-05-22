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

        image_encoder_mode="maskclip",
        maskclip_skip_last_layers=1,

        prompt_templates=[
            "a remote sensing image of {}.",
            "an aerial image of {}.",
            "a satellite image of {}.",
            "an overhead view of {}.",
        ],
        num_prompt_templates=4,
        num_clip_text_latents=32,
        normalize_label_for_clip=True,
    ),

    final_mixer_cfg=dict(
        enabled=True,

        num_class_tokens=32,
        fusion_layers=4,
        num_heads=8,
        dropout=0.1,
        presence_enabled=True,

        clip_sam_upsample_cfg=dict(
            enabled=True,
            window_size=8,
            shift_size=4,
            dropout=0.1,
            gamma_init=1.0,
            gamma_max=2.0,
        ),

        dynamic_code_cfg=dict(
            source="class_token_to_sam3_text",
        ),

        mask_prior_cfg=dict(
            type="softmax",
            tau=16.0,
            multiply_presence=True,
        ),

        window_attention_cfg=dict(
            window_size=8,
            shift_size=4,
            dropout=0.1,
        ),

        mask_head_cfg=dict(
            type="attn_feature_dot_dynamic_code",
            direct_dot=True,
            class_feature_pool_stride=4,
        ),
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.global_clip_sam_feature_builder",
            "core.clip_sam_upsampler",

            "core.class_token_query_embed",
            "core.class_token_text_cross_attn",
            "core.class_token_text_cross_attn_norm",
            "core.class_token_encoder_cross_attn",
            "core.class_token_encoder_cross_attn_norm",

            "core.final_mixer",
        ],
        frozen_modules=[],
    ),

    adapter_cfg=dict(),

    criterion_cfg=dict(
        ignore_index=255,

        final_bce_weight=0.4,
        final_dice_weight=1.0,
        final_ce_weight=0.4,
        final_ignore_bce_weight=0.1,

        presence_loss_weight=1.0,
        presence_layer_loss_weights=[0.02, 0.05, 0.1, 0.2],

        mask_layer_loss_weight=1.0,
        mask_layer_weights=[0.1, 0.2, 0.4],

        bce_class_balance_clamp_min=0.2,
        bce_class_balance_clamp_max=5.0,

        ce_class_balance_clamp_min=0.2,
        ce_class_balance_clamp_max=5.0,

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
                "core.global_clip_sam_feature_builder": dict(
                    lr_mult=2.0,
                    decay_mult=1.0,
                ),

                "core.clip_sam_upsampler": dict(
                    lr_mult=4.0,
                    decay_mult=1.0,
                ),

                "core.clip_sam_upsampler.window_attn.q_proj": dict(
                    lr_mult=10.0,
                    decay_mult=0.0,
                ),
                "core.clip_sam_upsampler.window_attn.k_proj": dict(
                    lr_mult=10.0,
                    decay_mult=0.0,
                ),
                "core.clip_sam_upsampler.shifted_window_attn.q_proj": dict(
                    lr_mult=10.0,
                    decay_mult=0.0,
                ),
                "core.clip_sam_upsampler.shifted_window_attn.k_proj": dict(
                    lr_mult=10.0,
                    decay_mult=0.0,
                ),

                "core.class_token_query_embed": dict(
                    lr_mult=4.0,
                    decay_mult=0.0,
                ),
                "core.class_token_text_cross_attn": dict(
                    lr_mult=4.0,
                    decay_mult=1.0,
                ),
                "core.class_token_text_cross_attn_norm": dict(
                    lr_mult=4.0,
                    decay_mult=0.0,
                ),
                "core.class_token_encoder_cross_attn": dict(
                    lr_mult=4.0,
                    decay_mult=1.0,
                ),
                "core.class_token_encoder_cross_attn_norm": dict(
                    lr_mult=4.0,
                    decay_mult=0.0,
                ),

                "core.final_mixer": dict(
                    lr_mult=4.0,
                    decay_mult=1.0,
                ),

                "mask_feature_attn.q_proj": dict(
                    lr_mult=10.0,
                    decay_mult=0.0,
                ),
                "mask_feature_attn.k_proj": dict(
                    lr_mult=10.0,
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