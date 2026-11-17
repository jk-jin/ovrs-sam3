_base_ = ["../_base_/dlrsd_loveda.py"]

model = dict(pipeline="adapter", aggregation=dict(enabled=False), remoteclip=dict(enabled=False),
    freeze=dict(remoteclip_text_finetune="frozen", remoteclip_image_finetune="frozen"))

experiment = dict(name="simple_adapter", variant_id="simple_adapter", kind="ablation",
    factors={'num_blocks': 0, 'directional_fusion': False, 'present_class_weight': 0.8, 'reference_distill_weight': 0.5, 'pipeline': 'adapter'}, note="Controlled paper variant; new training run required.")
runtime = dict(work_dir="./work_dirs/paper_batch4_10k/simple_adapter/seed_42/train")
