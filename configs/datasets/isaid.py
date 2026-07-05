isaid_classes = [
    'ship',
    'store tank',
    'baseball diamond',
    'tennis court',
    'basketball court',
    'ground track field',
    'bridge',
    'large vehicle',
    'small vehicle',
    'helicopter',
    'swimming pool',
    'roundabout',
    'soccer ball field',
    'plane',
    'harbor',
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/iSAID/img_dir/train',
        ann_dir='data/datasets/iSAID/ann_dir/train',
        classes=isaid_classes,
        img_suffix='.png',
        seg_suffix='_instance_color_RGB.png',
        ignore_index=255,
        reduce_zero_label=True,
        background_cfg=dict(
            enabled=False,
            class_id=0,
            class_name=None,
            exclude_from_forward=False,
        ),
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype', dtype='float32', scale=True),

            dict(
                type='RandomResizeByRatio',
                base_scale=(1008, 1008),
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),

            dict(
                type='RandomCrop',
                crop_size=(1008, 1008),
                cat_max_ratio=0.75,
                ignore_index=255,
                pad_if_needed=True,
                image_pad_value=0.0,
            ),

            dict(type='RandomHorizontalFlip', prob=0.5),
            dict(type='RandomVerticalFlip', prob=0.5),
            dict(type='RandomRotate90', prob=0.5),

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

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/iSAID/img_dir/val',
        ann_dir='data/datasets/iSAID/ann_dir/val',
        classes=isaid_classes,
        img_suffix='.png',
        seg_suffix='_instance_color_RGB.png',
        ignore_index=255,
        reduce_zero_label=True,
        background_cfg=dict(
            enabled=False,
            class_id=0,
            class_name=None,
            exclude_from_forward=False,
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