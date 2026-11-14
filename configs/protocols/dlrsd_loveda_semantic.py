_base_ = [
    "../_base_/model/reca_sam3.py",
    "../_base_/loss/semantic.py",
    "../_base_/optimizer/reca_sam3_adamw.py",
    "../_base_/schedule/full_20k.py",
    "../_base_/runtime.py",
    "../_base_/evaluation.py",
    "../_base_/logging.py",
    "../_base_/visualization.py",
    "../datasets/train/dlrsd.py",
    "../datasets/eval/loveda.py",
]

training = dict(
    regime="supervised_adaptation",
    dataset="dlrsd",
    split="train",
    protocol_id="dlrsd_scene_stratified_80_20_seed42_train_v1",
)

evaluation = dict(
    dataset="loveda",
    split="val",
    data_version="prepared-v1",
    protocol_id="loveda_7class_resize1008_v1",
    geometry=dict(
        input_hw=[1008, 1008],
        keep_ratio=False,
    ),
)
