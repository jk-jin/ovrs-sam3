_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="single_scale", variant_id="single_scale", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/single_scale/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "single_scale"]))

model = {'aggregation': {'window_scales': [36]}}
