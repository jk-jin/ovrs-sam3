_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="text_only", variant_id="text_only", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/text_only/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "text_only"]))

model = {'freeze': {'remoteclip_image_finetune': 'frozen'}}
