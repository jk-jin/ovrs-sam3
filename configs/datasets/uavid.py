uavid_classes = [
    'background',
    'building',
    'road',
    'tree',
    'low vegetation',
    'moving car',
    'static car',
    'human',
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/UAVid/img_dir/val',
        ann_dir='data/datasets/UAVid/ann_dir/val',
        classes=uavid_classes,
        img_suffix='.png',
        seg_suffix='.png',
        ignore_index=255,
        reduce_zero_label=False,
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