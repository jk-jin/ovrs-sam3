_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/floodnet.py",
]

experiment = dict(
    name="test_floodnet",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/floodnet",
)

evaluation = dict(
    dataset="floodnet",
    split="val+test",
    data_version="prepared-v1",
    protocol_id="floodnet_val+test_resize1008_v1",
)
