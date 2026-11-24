_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="self_only", variant_id="self_only", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/self_only/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "self_only"]))

model = {'aggregation': {'cross_class_mode': 'self'}}
