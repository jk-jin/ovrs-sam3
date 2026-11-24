_base_ = ["./_paper.py"]

# Fill selected weights here, or use --only VARIANT --checkpoint PATH --seed 42.
suite = {'validation_name': 'selected_weights',
 'selections': [{'variant': 'full', 'seed': 42, 'checkpoint': None},
                {'variant': 'self_only', 'seed': 42, 'checkpoint': None},
                {'variant': 'no_remoteclip_guidance', 'seed': 42, 'checkpoint': None},
                {'variant': 'no_intra', 'seed': 42, 'checkpoint': None},
                {'variant': 'single_scale', 'seed': 42, 'checkpoint': None},
                {'variant': 'unshared_windows', 'seed': 42, 'checkpoint': None},
                {'variant': 'frozen_remoteclip', 'seed': 42, 'checkpoint': None},
                {'variant': 'image_only', 'seed': 42, 'checkpoint': None},
                {'variant': 'text_only', 'seed': 42, 'checkpoint': None},
                {'variant': 'no_distillation', 'seed': 42, 'checkpoint': None}],
 'baselines': [{'variant': 'semantic_only',
                'config': 'configs/baselines/semantic_only.py',
                'datasets': 'main'},
               {'variant': 'semantic_presence',
                'config': 'configs/baselines/semantic_presence.py',
                'datasets': 'main'}]}
