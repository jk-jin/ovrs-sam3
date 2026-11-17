_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(aggregation=dict(update_streams="feature_only"))
experiment = dict(
    name="encoder_only_update",
    variant_id="encoder_only_update",
    kind="ablation",
    factors=dict(update_streams="feature_only"),
    note="Update encoder features only; retain the initial semantic prior in Q/K. "
         "Keep regular/shifted feature windows; omit semantic-update middle-layer guidance.",
)
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/encoder_only_update/seed_42/train")
