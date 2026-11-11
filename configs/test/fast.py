_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/fast.py",
]

experiment = dict(
    name="test_fast",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/fast",
)

evaluation = dict(
    dataset="fast",
    split="val",
    data_version="prepared-v1",
    protocol_id="fast_val_resize1008_v1",
)
