_base_ = ["./isaid_15_ignbg.py", "../datasets/eval/isaid_16_bg.py"]

experiment = dict(name="test_isaid_16_bg")
runtime = dict(work_dir="./work_dirs/test/isaid_16_bg")
evaluation = dict(
    dataset="isaid", protocol_id="isaid_16_bg_val_resize1008_v1",
    background_prob_threshold=0.0,
    postprocess=dict(prediction_prob_threshold=0.5),
)
