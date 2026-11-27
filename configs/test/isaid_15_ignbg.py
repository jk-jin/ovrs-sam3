_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/isaid_15_ignbg.py",
]

experiment = dict(
    name="test_isaid_15_ignbg",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/isaid_15_ignbg",
)

evaluation = dict(
    postprocess=dict(
        prediction_prob_threshold=0.0,
    ),
    dataset="isaid",
    split="val",
    data_version="prepared-v1",
    protocol_id="isaid_15_ignbg_val_resize1008_v1",
)
