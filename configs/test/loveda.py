_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/loveda.py",
]

experiment = dict(
    name="test_loveda",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/loveda",
)

evaluation = dict(
    dataset="loveda",
    split="val",
    data_version="prepared-v1",
    protocol_id="loveda_7class_resize1008_v1",
)
