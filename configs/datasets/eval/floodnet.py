_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/FloodNet/val+test/img",
        ann_dir="data/datasets/FloodNet/val+test/lbl",
        classes=[
        'background',
        'building flooded',
        'building non flooded',
        'road flooded',
        'road_non flooded',
        'water',
        'tree',
        'vehicle',
        'pool',
        'grass'
        ],
        img_suffix=".jpg",
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
