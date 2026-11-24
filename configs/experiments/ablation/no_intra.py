_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="no_intra", variant_id="no_intra", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/no_intra/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "no_intra"]))

model = {'aggregation': {'intra_class_enabled': False}}
