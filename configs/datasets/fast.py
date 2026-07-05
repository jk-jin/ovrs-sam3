fast_classes = [
    'Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'other-ship',
    'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship',
    'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'other-vehicle',
    'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor',
    'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21',
    'C919', 'A220', 'A321', 'A330', 'A350', 'other-airplane',
    'Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court',
    'Roundabout', 'Intersection', 'Bridge'
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/FAST/val/images',
        ann_dir='data/datasets/FAST/val/semlabels/gray',
        classes=fast_classes,
        img_suffix='.png',
        seg_suffix='.png',
        ignore_index=255,
        reduce_zero_label=False,
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