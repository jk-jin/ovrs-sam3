_base_ = ["../../protocols/dlrsd_loveda_semantic.py"]

experiment = dict(
    name="dlrsd_loveda_full",
    method_id="reca_sam3",
    method_name="ReCA-SAM3",
    variant_id="full",
    implementation_id="reca-sam3-paper-v1",
    kind="main",
    factors=dict(
        num_blocks=4,
        directional_fusion=True,
        reference_distill_weight=0.5,
        present_ignore_suppression_weight=0.0,
    ),
    note="DLRSD train -> LoveDA val; seven-class semantic evaluation; no ignore suppression.",
)

runtime = dict(
    work_dir="./work_dirs/main/dlrsd_loveda_full",
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
        group="main",
        tags=[
            "reca-sam3",
            "dlrsd-train",
            "loveda-val",
            "64-templates",
        ],
    ),
)

visualization = dict(
    enabled=True,
    save_raw_final_prediction=False,
    save_score_summary=False,
    save_score_heatmaps=False,
    save_reference_segmentation=False,
    vis_prob=0.02,
    max_samples_per_epoch=100,
)
