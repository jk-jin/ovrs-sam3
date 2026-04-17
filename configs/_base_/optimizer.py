optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={},
        ),
    )
)