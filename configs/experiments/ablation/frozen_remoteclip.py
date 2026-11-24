_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="frozen_remoteclip", variant_id="frozen_remoteclip", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/frozen_remoteclip/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "frozen_remoteclip"]))

model = {'freeze': {'remoteclip_text_finetune': 'frozen', 'remoteclip_image_finetune': 'frozen'}}
