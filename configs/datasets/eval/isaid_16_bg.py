_base_ = ["./isaid_15_ignbg.py"]

# Raw labels: 0 background, 1..15 foreground, 255 ignored. Never shift twice.

val_dataloader = dict(dataset=dict(classes=['background', 'ship', 'store tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'swimming pool', 'roundabout', 'soccer ball field', 'plane', 'harbor'], reduce_zero_label=False, background_cfg=dict(enabled=True, class_id=0, class_name="background")))
