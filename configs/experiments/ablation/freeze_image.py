_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(freeze=dict(remoteclip_image_finetune="frozen"))

experiment = dict(name="freeze_image", variant_id="freeze_image", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'text_finetune': 'attention', 'image_finetune': 'frozen'}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/freeze_image/seed_42/train")
