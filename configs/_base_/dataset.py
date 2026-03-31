# Replace `your_project` with your actual package name.
# This template now ships with:
# - data/dataset.py  -> JsonPromptSegDataset
# - data/collate.py  -> SAM3BatchCollator
#
# Example annotation schema is documented in JsonPromptSegDataset.

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.JsonPromptSegDataset',
        data_root='data/train',
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008, box_format='xyxy'),
        ],
    ),
    collate_fn=dict(
        type='data.collate.SAM3BatchCollator',
        pad_size_divisor=14,
        normalize_boxes=True,
        box_format='xyxy',
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.JsonPromptSegDataset',
        data_root='data/val',
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008, box_format='xyxy'),
        ],
    ),
    collate_fn=dict(
        type='data.collate.SAM3BatchCollator',
        pad_size_divisor=14,
        normalize_boxes=True,
        box_format='xyxy',
    ),
)
