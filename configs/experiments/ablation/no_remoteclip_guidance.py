_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="no_remoteclip_guidance", variant_id="no_remoteclip_guidance", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/no_remoteclip_guidance/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "no_remoteclip_guidance"]))

model = {'remoteclip': {'enabled': False}, 'aggregation': {'use_remoteclip_guidance': False}, 'freeze': {'remoteclip_text_finetune': 'frozen', 'remoteclip_image_finetune': 'frozen'}}
