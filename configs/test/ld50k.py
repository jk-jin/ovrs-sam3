_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/ld50k.py",
]

experiment = dict(
    name="test_ld50k",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/ld50k",
)

evaluation = dict(
    dataset="ld50k",
    split="val",
    data_version="prepared-v1",
    protocol_id="ld50k_val_resize1008_v1",
)
