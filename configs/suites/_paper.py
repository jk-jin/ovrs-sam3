# Each block learns separate residual weights for its cross-class and three window updates.
suite = dict(
    root="work_dirs/reca_sam3",
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
