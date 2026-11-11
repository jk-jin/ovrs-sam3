_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/vaihingen.py",
]

experiment = dict(
    name="test_vaihingen",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/vaihingen",
)

evaluation = dict(
    dataset="vaihingen",
    split="val",
    data_version="prepared-v1",
    protocol_id="vaihingen_val_resize1008_v1",
)
