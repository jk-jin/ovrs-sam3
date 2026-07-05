floodnet_classes = [
    'background', 'building flooded', 'building non flooded',
    'road flooded', 'road_non flooded', 'water', 'tree',
    'vehicle', 'pool', 'grass'
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/FloodNet/val+test/img',
        ann_dir='data/datasets/FloodNet/val+test/lbl',
        classes=floodnet_classes,
        img_suffix='.jpg',
        seg_suffix='.png',
        ignore_index=255,
        reduce_zero_label=False,
        background_cfg=dict(
            enabled=True,
            class_id=0,
            class_name="background",
            exclude_from_forward=True,
        ),
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype', dtype='float32', scale=True),
            dict(type='Resize', size=(1008, 1008), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ],
    ),
    collate_fn=dict(
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)