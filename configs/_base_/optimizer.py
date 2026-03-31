optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                'semantic_adapter': dict(lr_mult=1.0, decay_mult=1.0),
                'instance_adapter': dict(lr_mult=1.0, decay_mult=1.0),
                'segmentation_head': dict(lr_mult=1.0, decay_mult=1.0),
                'prompt_mlp': dict(lr_mult=0.5, decay_mult=1.0),
            },
        ),
    )
)
