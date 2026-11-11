_base_ = [
    "../../protocols/isaid_loveda_semantic.py",
    "../../_base_/schedule/exp_4k.py",
]

experiment = dict(
    name="isaid_loveda_dev_4k",
    method_id="reca_sam3",
    method_name="ReCA-SAM3",
    variant_id="full",
    implementation_id="reca-sam3-paper-v1",
    kind="debug",
    factors=dict(
        num_blocks=4,
        directional_fusion=True,
    ),
    note=(
        "Short development run. train_cfg.val_max_iters=500 means validation "
        "is partial and must not enter paper tables."
    ),
)

runtime = dict(
    work_dir="./work_dirs/dev/isaid_loveda_4k",
)

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
)

logging = dict(
    wandb=dict(
        enabled=True,
        mode="offline",
        project="reca-sam3",
        group="dev",
        tags=[
            "reca-sam3",
            "debug",
            "isaid-train",
            "loveda-val",
        ],
    ),
)
