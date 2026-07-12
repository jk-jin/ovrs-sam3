_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/potsdam/img_dir/val",
        ann_dir="data/datasets/potsdam/ann_dir/val",
        classes=[
        'road',
        'building',
        'grass',
        'tree',
        'car',
        'clutter'
        ],
        img_suffix=".png",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=False,
        background_cfg=dict(
            enabled=True,
            class_id=5,
            class_name='clutter',
            exclude_from_forward=True,
        ),
    ),
)
