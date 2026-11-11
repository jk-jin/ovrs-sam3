_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/isaid.py",
]

experiment = dict(
    name="test_isaid",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/isaid",
)

evaluation = dict(
    dataset="isaid",
    split="val",
    data_version="prepared-v1",
    protocol_id="isaid_val_resize1008_v1",
)
