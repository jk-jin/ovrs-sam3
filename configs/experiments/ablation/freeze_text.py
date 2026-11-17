_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(freeze=dict(remoteclip_text_finetune="frozen"))

experiment = dict(name="freeze_text", variant_id="freeze_text", kind="ablation",
    factors={'num_blocks': 4, 'directional_fusion': True, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'text_finetune': 'frozen', 'image_finetune': 'attention'}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/freeze_text/seed_42/train")
