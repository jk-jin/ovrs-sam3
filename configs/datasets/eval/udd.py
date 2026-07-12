_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/UDD/UDD5/val/src",
        ann_dir="data/datasets/UDD/UDD5/val/gt",
        classes=[
        'vegetation',
        'building',
        'road',
        'vehicle',
        'background'
        ],
        img_suffix=".JPG",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=False,
        background_cfg=dict(
            enabled=True,
            class_id=4,
            class_name='background',
            exclude_from_forward=True,
        ),
    ),
)
