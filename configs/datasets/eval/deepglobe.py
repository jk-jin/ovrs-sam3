_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/deepglobe/img_dir/train",
        ann_dir="data/datasets/deepglobe/ann_dir/train",
        classes=[
        'urban',
        'agriculture',
        'rangeland',
        'forest',
        'water',
        'barren',
        'unknown'
        ],
        img_suffix=".png",
        seg_suffix=".png",
        ignore_index=255,
        reduce_zero_label=False,
        background_cfg=dict(
            enabled=True,
            class_id=6,
            class_name='unknown',
            exclude_from_forward=True,
        ),
    ),
)
