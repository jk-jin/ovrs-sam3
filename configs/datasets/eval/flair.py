_base_ = ["../../_base_/dataloader/semantic_eval.py"]

val_dataloader = dict(
    dataset=dict(
        img_dir="data/datasets/FLAIR_test/image",
        ann_dir="data/datasets/FLAIR_test/mask",
        classes=[
        'building',
        'pervious surface',
        'impervious surface',
        'bare soil',
        'water',
        'coniferous',
        'deciduous',
        'brushwood',
        'vineyard',
        'herbaceous vegetation',
        'agricultural land',
        'plowed land',
        'other'
        ],
        img_suffix=".png",
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
