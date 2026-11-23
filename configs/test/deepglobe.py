_base_ = [
    "./_base_/reca_sam3_eval.py",
    "../datasets/eval/deepglobe.py",
]

experiment = dict(
    name="test_deepglobe",
    kind="evaluation",
)

runtime = dict(
    work_dir="./work_dirs/test/deepglobe",
)

evaluation = dict(
    postprocess=dict(
        prediction_prob_threshold=0.5,
    ),
    dataset="deepglobe",
    split="train",
    data_version="prepared-v1",
    protocol_id="deepglobe_train_resize1008_v1",
)
