optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                "core.encoder_refiner": dict(lr_mult=1.0, decay_mult=1.0),
                "core.clip_text_encoder": dict(lr_mult=0.01, decay_mult=1.0),
                "core.clip_image_encoder": dict(lr_mult=0.01, decay_mult=1.0),
            },
        ),
    )
)
