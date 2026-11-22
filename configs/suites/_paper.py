# Each block adds a cross-class update followed by the sum of three regular-window updates.
suite = dict(
    root="work_dirs/multiscale_windows_v3",
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
