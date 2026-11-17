_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(aggregation=dict(semantic_init="zeros"))

experiment = dict(name="no_score_prior", variant_id="no_score_prior", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'semantic_init': 'zeros'}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/no_score_prior/seed_42/train")
