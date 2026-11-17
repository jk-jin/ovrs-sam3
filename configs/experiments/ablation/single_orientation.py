_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(fusion=dict(orientations=[0]))

experiment = dict(name="single_orientation", variant_id="single_orientation", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': False, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/single_orientation/seed_42/train")
