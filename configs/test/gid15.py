_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/gid15.py",
]

experiment = dict(
    name="test_gid15",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/gid15",
)

evaluation = dict(
    dataset="gid15",
    split="val",
    data_version="prepared-v1",
    protocol_id="gid15_val_resize1008_v1",
)
