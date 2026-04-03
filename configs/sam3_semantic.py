_base_ = [
    './_base_/runtime.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/visualization.py',
    './datasets/loveda.py'
]

model = dict(
    bpe_path='assets/bpe_simple_vocab_16e6.txt.gz',
    checkpoint_path='weights/sam3.pt',
    load_from_hf=False,
    device='cuda',
    eval_mode=False,
    compile=False,

    semantic_topk=20,
    semantic_aggregation='weighted_sum',

    semantic_use_query_branch=True,
    semantic_use_semantic_branch=True,
    semantic_fusion_mode='max',

    semantic_use_presence_score=True,
    semantic_presence_reduce='max',

    prompt_chunk_size=8,

    freeze_cfg=dict(
        train_adapters_only=False,
        trainable_modules=[],
        frozen_modules=[],
    ),
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

criterion = dict(
    semantic=dict(
        loss_ce=1.0,
        loss_dice=0.0,
    ),
    ignore_index=255,
)