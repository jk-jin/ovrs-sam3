_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/VDD/val/src",
        ann_dir="data/datasets/VDD/val/gt",
        classes=[
        'background',
        'facade',
        'road',
        'vegetation',
        'vehicle',
        'roof',
        'water'
        ],
        img_suffix=".JPG",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=False,
        background_cfg=dict(
            enabled=True,
            class_id=0,
            class_name='background',
            exclude_from_forward=True,
        ),
    ),
)
