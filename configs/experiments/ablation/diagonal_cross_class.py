_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(aggregation=dict(cross_class_mode="diagonal"))

experiment = dict(name="diagonal_cross_class", variant_id="diagonal_cross_class", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'cross_class_mode': 'diagonal'}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/diagonal_cross_class/seed_42/train")
