_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="image_only", variant_id="image_only", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/image_only/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "image_only"]))

model = {'freeze': {'remoteclip_text_finetune': 'frozen'}}
