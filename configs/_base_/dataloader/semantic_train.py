train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type="data.dataset.OVSemanticSegDataset",
        return_raw_image=True,
        transforms=[
            dict(type="ToTensor"),
            dict(type="ConvertImageDtype", dtype="float32", scale=True),

            dict(
                type="ResizeShortestEdge",
                short_edge=1008,
            ),

            dict(
                type="RandomCrop",
                crop_size=(1008, 1008),
                cat_max_ratio=1.0,
                ignore_index=255,
                pad_if_needed=True,
                image_pad_value=0.0,
            ),

            dict(type="ColorAugSSD"),
            dict(type="RandomHorizontalFlip", prob=0.5),

            dict(
                type="Normalize",
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ],
    ),
    collate_fn=dict(
        type="data.collate.OVSemanticCollator",
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)
