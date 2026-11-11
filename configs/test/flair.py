_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/flair.py",
]

experiment = dict(
    name="test_flair",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/flair",
)

evaluation = dict(
    dataset="flair",
    split="test",
    data_version="prepared-v1",
    protocol_id="flair_test_resize1008_v1",
)
