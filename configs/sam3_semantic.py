_base_ = [
    './_base_/runtime.py',
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/visualization.py',
]

model = dict(
    bpe_path='assets/bpe_simple_vocab_16e6.txt.gz',
    checkpoint_path='weights/sam3.pt',
    load_from_hf=False,
    device='cuda',
    eval_mode=False,
    enable_segmentation=True,
    compile=False,
    return_segmentor=True,
    semantic_topk=20,
    semantic_aggregation='weighted_sum',
    freeze_cfg=dict(
        freeze_backbone=True,
        freeze_text_encoder=True,
        freeze_transformer_encoder=True,
        freeze_transformer_decoder=True,
        freeze_geometry_encoder=True,
        freeze_dot_prod_scoring=True,
        freeze_segmentation_head=False,
        train_adapters_only=False,
        trainable_name_keywords=['semantic_adapter', 'segmentation_head'],
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.mmseg_style_prompt_dataset.MMSegStylePromptDataset',
        data_root='data/datasets/ld50k',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        img_suffix='.png',
        seg_map_suffix='.png',
        split_file=None,
        classes=[
            'background',
            'agriculture land',
            'vehicle',
            'tree',
        ],
        ignore_index=255,
        reduce_zero_label=True,
        filter_empty_gt=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008, box_format='xyxy'),
            dict(type='PadToSize', size=(1008, 1008)),
        ]
    ),
    collate_fn=dict(
        type='data.collate.SAM3BatchCollator',
        pad_size_divisor=14,
        normalize_boxes=True,
        box_format='xyxy',
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.mmseg_style_prompt_dataset.MMSegStylePromptDataset',
        data_root='data/datasets/ld50k',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix='.png',
        seg_map_suffix='.png',
        split_file=None,
        classes=[
            'background',
            'agriculture land',
            'vehicle',
            'tree',
        ],
        ignore_index=255,
        reduce_zero_label=True,
        filter_empty_gt=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008, box_format='xyxy'),
            dict(type='PadToSize', size=(1008, 1008)),
        ]
    ),
    collate_fn=dict(
        type='data.collate.SAM3BatchCollator',
        pad_size_divisor=14,
        normalize_boxes=True,
        box_format='xyxy',
    ),
)

train_cfg = dict(
    task='semantic',
    max_epochs=12,
    log_interval=20,
    use_amp=True,
    grad_clip_norm=0.1,
    save_interval=1,
    device='cuda',
)

criterion = dict(
    semantic=dict(
        loss_bce=1.0,
        loss_dice=1.0,
    )
)
