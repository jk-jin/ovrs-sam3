_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/potsdam.py",
]

experiment = dict(
    name="test_potsdam",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/potsdam",
)

evaluation = dict(
    dataset="potsdam",
    split="val",
    data_version="prepared-v1",
    protocol_id="potsdam_val_resize1008_v1",
)
