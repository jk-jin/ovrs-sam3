_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/UAVid/img_dir/val",
        ann_dir="data/datasets/UAVid/ann_dir/val",
        classes=[
        'background',
        'building',
        'road',
        'tree',
        'low vegetation',
        'moving car',
        'static car',
        'human'
        ],
        img_suffix=".png",
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
