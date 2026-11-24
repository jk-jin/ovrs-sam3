_base_ = ["../main/dlrsd_loveda_full.py"]

experiment = dict(name="no_distillation", variant_id="no_distillation", kind="ablation", note="Single-factor ablation of Full.")

runtime = dict(work_dir="./work_dirs/reca_sam3_paper/no_distillation/seed_42/train")

logging = dict(wandb=dict(group="ablation", tags=["reca-sam3", "no_distillation"]))

loss = {'reference_distill_weight': 0.0}
