_base_ = [
    './_base_/runtime.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/visualization.py',
    './datasets/isaid.py'
]

model = dict(
    bpe_path='assets/bpe_simple_vocab_16e6.txt.gz',
    checkpoint_path='weights/sam3.pt',
    load_from_hf=False,
    device='cuda',
    eval_mode=False,
    compile=False,

    semantic_aggregation='max',

    semantic_use_instance_branch=True,
    semantic_use_semantic_branch=True,
    semantic_fusion_mode='max',

    semantic_use_presence_score=True,
    confidence_threshold=0.5,
    prompt_chunk_size=8,

    freeze_cfg=dict(
        train_adapters_only=False,
        trainable_modules=[],
        frozen_modules=[],
    ),
)

eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.2,
    bg_idx=0,
    use_score_map=True,
)

train_cfg = dict(
    max_epochs=12,
    log_interval=20,
    use_amp=True,
    grad_clip_norm=0.1,
    save_interval=1,
    eval_interval=1,
    monitor='total_loss',
    monitor_mode='min',
    max_keep_ckpts=5,
    device='cuda',
    auto_resume=False,
)

tta_cfg = dict(
    enabled=False,
    scales=[0.75, 1.0, 1.25],
    flip_modes=['none', 'h', 'v'],
    size_divisor=14,
)

criterion = dict(
    semantic=dict(
        loss_ce=1.0,
        loss_dice=0.0,
    ),
    ignore_index=255,
)