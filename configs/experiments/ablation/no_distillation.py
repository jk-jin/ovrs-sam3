_base_ = ["../_base_/dlrsd_loveda.py"]

loss = dict(reference_distill_weight=0.0)

experiment = dict(name="no_distillation", variant_id="no_distillation", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.0}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/no_distillation/seed_42/train")
