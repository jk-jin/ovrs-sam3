_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(aggregation=dict(intra_class_enabled=False))

experiment = dict(name="no_intra_class", variant_id="no_intra_class", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'intra_class_enabled': False}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/no_intra_class/seed_42/train")
