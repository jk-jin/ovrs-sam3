_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/DLRSD/img_dir/val",
        ann_dir="data/datasets/DLRSD/ann_dir/val",
        classes=[
        'airplane',
        'bare soil',
        'buildings',
        'cars',
        'chaparral',
        'court',
        'dock',
        'field',
        'grass',
        'mobile home',
        'pavement',
        'sand',
        'sea',
        'ship',
        'tanks',
        'trees',
        'water'
        ],
        img_suffix=".jpg",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=False,
        background_cfg=dict(
            enabled=False,
            class_id=0,
            class_name=None,
            exclude_from_forward=False,
        ),
    ),
)
