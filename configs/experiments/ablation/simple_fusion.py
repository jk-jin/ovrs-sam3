_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(fusion=dict(type="concat"))

experiment = dict(name="simple_fusion", variant_id="simple_fusion", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'fusion_type': 'concat'}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/simple_fusion/seed_42/train")
