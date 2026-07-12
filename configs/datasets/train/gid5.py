_base_ = ["../../_base_/dataloader/semantic_train.py"]

train_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/gid5/img_dir/train",
        ann_dir="data/datasets/gid5/ann_dir/train",
        classes=[
        'built-up',
        'farmland',
        'forest',
        'meadow',
        'water'
        ],
        img_suffix=".png",
        seg_suffix=".png",
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
