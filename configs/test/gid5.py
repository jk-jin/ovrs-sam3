_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/gid5.py",
]

experiment = dict(
    name="test_gid5",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/gid5",
)

evaluation = dict(
    dataset="gid5",
    split="val",
    data_version="prepared-v1",
    protocol_id="gid5_val_resize1008_v1",
)
