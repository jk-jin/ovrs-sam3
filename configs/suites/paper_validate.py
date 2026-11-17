_base_ = ["./_paper.py"]
# Fill each checkpoint explicitly; None leaves that training run pending.
# Paths are relative to repository root; use the same full checkpoint for all tables.
suite = dict(validation_name="selected_weights", selections=[{'variant': 'full', 'seed': 42, 'checkpoint': None},
 {'variant': 'simple_adapter', 'seed': 42, 'checkpoint': None},
 {'variant': 'diagonal_cross_class', 'seed': 42, 'checkpoint': None},
 {'variant': 'encoder_only_update', 'seed': 42, 'checkpoint': None},
 {'variant': 'no_intra_class', 'seed': 42, 'checkpoint': None},
 {'variant': 'no_score_prior', 'seed': 42, 'checkpoint': None},
 {'variant': 'simple_fusion', 'seed': 42, 'checkpoint': None},
 {'variant': 'single_orientation', 'seed': 42, 'checkpoint': None},
 {'variant': 'no_distillation', 'seed': 42, 'checkpoint': None},
 {'variant': 'freeze_both', 'seed': 42, 'checkpoint': None},
 {'variant': 'freeze_text', 'seed': 42, 'checkpoint': None},
 {'variant': 'freeze_image', 'seed': 42, 'checkpoint': None}],
    baselines=[
        dict(variant="semantic_only", config="configs/baselines/semantic_only.py", datasets="main"),
        dict(variant="semantic_presence", config="configs/baselines/semantic_presence.py", datasets="main"),
    ])
