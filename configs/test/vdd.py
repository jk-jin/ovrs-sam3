_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/vdd.py",
]

experiment = dict(
    name="test_vdd",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/vdd",
)

evaluation = dict(
    dataset="vdd",
    split="val",
    data_version="prepared-v1",
    protocol_id="vdd_val_resize1008_v1",
)
