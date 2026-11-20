# Each block updates encoder features with cross/regular/shifted attention.
suite = dict(
    root="work_dirs/feature_blocks_v3",
    seeds=[42],
    devices=["cuda:0"],
    experiments=[dict(
        variant="full",
        config="configs/experiments/main/dlrsd_loveda_full.py",
        datasets="main",
    )],
    dataset_groups=dict(main=[
        "dlrsd", "isaid", "loveda", "potsdam", "vaihingen", "uavid", "udd", "vdd",
    ]),
)
