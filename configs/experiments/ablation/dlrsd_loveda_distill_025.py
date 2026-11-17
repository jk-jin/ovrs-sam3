_base_ = ["../../protocols/dlrsd_loveda_semantic.py"]

experiment = dict(
    name="dlrsd_loveda_distill_025",
    method_id="reca_sam3",
    method_name="ReCA-SAM3",
    variant_id="distill_0.25",
    implementation_id="reca-sam3-paper-v1",
    kind="ablation",
    factors=dict(
        num_blocks=4,
        directional_fusion=True,
        present_class_weight=0.8,
        reference_distill_weight=0.25,
    ),
    note="Fixed reference distillation weight ablation on valid pixels only.",
)

loss = dict(
    reference_distill_weight=0.25,
)

runtime = dict(
    work_dir="./work_dirs/ablation/dlrsd_loveda_distill_025",
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
)

logging = dict(
    wandb=dict(
        enabled=True,
        mode="offline",
        project="reca-sam3",
        group="ablation",
        tags=[
            "reca-sam3",
            "ablation",
            "reference-distill-0.25",
        ],
    ),
)
