_base_ = [
    "../_base_/model/ovrs_sam3.py",
    "../_base_/optimizer/ovrs_sam3_adamw.py",
    "../_base_/schedule/full_20k.py",
    "../_base_/runtime.py",
    "../_base_/evaluation.py",
    "../_base_/tracking.py",
    "../_base_/visualization.py",
    "../datasets/train/isaid.py",
    "../datasets/eval/loveda.py",
]

work_dir = "./work_dirs/full/isaid_loveda"

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
)

eval_cfg = dict(
    metric_groups=[
        dict(
            name="loveda_no_background",
            class_ids=[1, 2, 3, 4, 5, 6],
            metrics=["miou", "macc"],
        ),
    ],
)

train_cfg = dict(
    monitor="semantic.groups.loveda_no_background.miou",
)

visualization = dict(
    enabled=True,
    save_raw_final_prediction=False,
    save_score_summary=False,
    save_score_heatmaps=False,
    save_sam3_direct_segmentation=False,
    vis_prob=0.02,
    max_samples_per_epoch=100,
)

experiment_tracking = dict(
    wandb=dict(
        enabled=True,
        name="full_isaid_loveda_final_v1",
        group="full_isaid_loveda",
        tags=[
            "full",
            "isaid-train",
            "loveda-val",
            "refiner-36x36",
            "fixed-32-templates",
            "score-dim-256",
        ],
    ),
)
