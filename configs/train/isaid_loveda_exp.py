_base_ = [
    "../_base_/model/ovrs_sam3.py",
    "../_base_/optimizer/ovrs_sam3_adamw.py",
    "../_base_/schedule/exp_4k.py",
    "../_base_/runtime.py",
    "../_base_/evaluation.py",
    "../_base_/tracking.py",
    "../_base_/visualization.py",
    "../datasets/train/isaid.py",
    "../datasets/eval/loveda.py",
]

work_dir = "./work_dirs/exp/isaid_loveda"

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
)

experiment_tracking = dict(
    wandb=dict(
        enabled=True,
        group="exp_isaid_loveda",
        tags=[
            "exp",
            "isaid-train",
            "loveda-val",
            "refiner-36x36",
            "fixed-32-templates",
            "score-dim-256",
        ],
        name_prefix="exp",
    ),
)
