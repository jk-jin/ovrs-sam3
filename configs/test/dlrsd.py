_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/dlrsd.py",
]

experiment = dict(
    name="test_dlrsd",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/dlrsd",
)

evaluation = dict(
    dataset="dlrsd",
    split="val",
    data_version="ovrsisbench-dlrsd-official-v1",
    protocol_id="dlrsd_ovrsisbench_official_val_resize1008_v1",
)
