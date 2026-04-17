_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/potsdam.py",
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
        extra_token_templates=[
            "a remote sensing image of {}.",
            "an aerial image of {}.",
        ],
        num_extra_tokens=2,
        text_token_gate_init=0.5,
        normalize_label_for_clip=True,
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.clip_text_proj",
            "core.clip_text_token_gate",
            "core.clip_image_proj",
            "core.clip_text_to_image_attn",
            "core.clip_text_to_image_norm",
            "core.clip_to_sam3_text_attn",
            "core.clip_to_sam3_text_norm",
        ],
        frozen_modules=[],
    ),

    criterion_cfg=dict(
        ignore_index=255,
        semantic_bce_weight=0.2,
        semantic_dice_weight=1.0,
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
    prob_thd=0.5,
    bg_idx=0,
    use_score_map=True,
)

optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                "core.clip_text_proj": dict(lr_mult=1.0, decay_mult=1.0),
                "core.clip_image_proj": dict(lr_mult=1.0, decay_mult=1.0),
                "core.clip_text_to_image_attn": dict(lr_mult=1.0, decay_mult=1.0),
                "core.clip_to_sam3_text_attn": dict(lr_mult=1.0, decay_mult=1.0),
            },
        ),
    )
)

param_scheduler = dict(
    type="CosineAnnealingLR",
    T_max=40000,
    eta_min=1e-6,
)

train_cfg = dict(
    max_iters=40000,
    save_interval=2000,
    eval_interval=2000,
    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.1,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=5,
    auto_resume=False,
    device="cuda",
)

tta_cfg = dict(
    enabled=False,
    scales=[0.75, 1.0, 1.25],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)