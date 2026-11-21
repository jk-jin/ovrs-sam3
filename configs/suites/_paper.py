# Each block updates encoder features with cross-class and two multiscale window updates.
suite = dict(
    root="work_dirs/multiscale_windows",
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
