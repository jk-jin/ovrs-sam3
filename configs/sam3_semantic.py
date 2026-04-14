_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/potsdam.py",
]

model = dict(
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="weights/sam3.pt",
    load_from_hf=False,
    device="cuda",
    eval_mode=False,
    compile=False,

    semantic_use_instance_branch=True,
    semantic_use_semantic_branch=True,
    semantic_fusion_mode="max",

    semantic_use_presence_score=True,
    confidence_threshold=0.5,
    prompt_chunk_size=8,

    openclip_cfg=dict(
        text_encoder=dict(
            enabled=True,
            model_name="ViT-L-14",
            checkpoint_path="weights/RemoteCLIP-ViT-L-14.pt",
            extra_token_templates=[
                "a remote sensing image of {}.",
                "an aerial image of {}.",
            ],
            num_extra_tokens=2,
            text_token_gate_init=1.0,
            normalize_label_for_clip=True,
        ),
        image_encoder=dict(
            enabled=False,
            model_name="ViT-L-14",
            checkpoint_path="weights/RemoteCLIP-ViT-L-14.pt",
            default_output="feat_map",
        ),
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.clip_text_encoder.resizer",
            "core.clip_text_token_gate",
        ],
        frozen_modules=[],
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
)

eval_cfg = dict(
    ignore_index=255,
    prob_thd=0,
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
                "core.clip_text_encoder.resizer": dict(lr_mult=1.0, decay_mult=1.0),
            },
        ),
    )
)

param_scheduler = dict(
    type="CosineAnnealingLR",
    T_max=4,
    eta_min=1e-6,
)

train_cfg = dict(
    max_epochs=4,
    log_interval=20,
    use_amp=True,
    grad_clip_norm=0.1,
    save_interval=1,
    eval_interval=1,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=5,
    device="cuda",
    auto_resume=False,
)

tta_cfg = dict(
    enabled=False,
    scales=[0.75, 1.0, 1.25],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)

criterion = dict(
    semantic_bce=1.0,
    semantic_dice=1.0,
    instance_bce=1.0,
    instance_dice=1.0,
    presence_bce=0.25,
    ignore_index=255,
)