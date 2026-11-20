_base_ = ["./_paper.py"]
# Fill the new full model checkpoint explicitly; None leaves it pending.
suite = dict(
    validation_name="selected_weights",
    selections=[dict(variant="full", seed=42, checkpoint=None)],
    baselines=[
        dict(variant="semantic_only", config="configs/baselines/semantic_only.py", datasets="main"),
        dict(variant="semantic_presence", config="configs/baselines/semantic_presence.py", datasets="main"),
    ],
)
