_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/loveDA/img_dir/val",
        ann_dir="data/datasets/loveDA/ann_dir/val",
        classes=[
        'background',
        'building',
        'road',
        'water',
        'barren',
        'forest',
        'agricultural'
        ],
        img_suffix=".png",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=True,
        background_cfg=dict(
            enabled=True,
            class_id=0,
            class_name='background',
            exclude_from_forward=True,
        ),
    ),
)
