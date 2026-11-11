_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/udd.py",
]

experiment = dict(
    name="test_udd",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/udd",
)

evaluation = dict(
    dataset="udd",
    split="val",
    data_version="prepared-v1",
    protocol_id="udd_val_resize1008_v1",
)
