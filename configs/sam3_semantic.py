_base_ = [
    './_base_/runtime.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/visualization.py',
]

ld50k_classes = [
    'background',
    'agriculture land',
    'vehicle',
    'tree',
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
    prompt_chunk_size=16,   # 第5点会配套加代码
    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            'semantic_adapter',
            'core.segmentation_head',
            'core.dot_prod_scoring',
        ],
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/ld50k/img_dir/train',
        ann_dir='data/datasets/ld50k/ann_dir/train',
        classes=ld50k_classes,
        img_suffix='.png',
        seg_suffix='.png',
        ignore_index=255,
        reduce_zero_label=False,   # 第4点：显式保留 background
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008),
            dict(type='PadToSize', size=(1008, 1008), label_pad_value=255),
        ],
    ),
    collate_fn=dict(
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/ld50k/img_dir/val',
        ann_dir='data/datasets/ld50k/ann_dir/val',
        classes=ld50k_classes,
        img_suffix='.png',
        seg_suffix='.png',
        ignore_index=255,
        reduce_zero_label=False,
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008),
            dict(type='PadToSize', size=(1008, 1008), label_pad_value=255),
        ],
    ),
    collate_fn=dict(
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
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