_base_ = ["../../protocols/dlrsd_loveda_semantic.py"]

experiment = dict(
    name="dlrsd_loveda_distill_025",
    method_id="reca_sam3",
    method_name="ReCA-SAM3",
    variant_id="distill_cosine_0.25_to_0",
    implementation_id="reca-sam3-paper-v1",
    kind="ablation",
    factors=dict(
        num_blocks=4,
        directional_fusion=True,
        present_class_weight=0.8,
        reference_distill_start_iter=1000,
        reference_distill_end_iter=12000,
        reference_distill_start_weight=0.25,
        reference_distill_end_weight=0.0,
    ),
    note="Cosine reference distillation ablation: 0.25 to 0 at iterations 1000-12000.",
)

loss = dict(
    reference_distill_start_weight=0.25,
)

runtime = dict(
    work_dir="./work_dirs/ablation/dlrsd_loveda_distill_025",
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
        group="ablation",
        tags=[
            "reca-sam3",
            "ablation",
            "reference-distill-0.25",
        ],
    ),
)
