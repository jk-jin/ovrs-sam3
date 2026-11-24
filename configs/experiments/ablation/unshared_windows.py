_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="unshared_windows", variant_id="unshared_windows", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/unshared_windows/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "unshared_windows"]))

model = {'aggregation': {'share_window_weights_across_scales': False}}
