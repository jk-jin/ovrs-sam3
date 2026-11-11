_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/uavid.py",
]

experiment = dict(
    name="test_uavid",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/uavid",
)

evaluation = dict(
    dataset="uavid",
    split="val",
    data_version="prepared-v1",
    protocol_id="uavid_val_resize1008_v1",
)
