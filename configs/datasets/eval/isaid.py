_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/iSAID/img_dir/val",
        ann_dir="data/datasets/iSAID/ann_dir/val",
        classes=[
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
        'harbor'
        ],
        img_suffix=".png",
        seg_suffix="_instance_color_RGB.png",
        ignore_index=255,
        reduce_zero_label=True,
        background_cfg=dict(
            enabled=False,
            class_id=0,
            class_name=None,
            exclude_from_forward=False,
        ),
    ),
)
