seed = 42
work_dir = "./work_dirs/ovrs_sam3"

default_scope = "sam3"
log_level = "INFO"

default_hooks = dict(
    logger=dict(
        interval=20,
        val_interval=50,
        print_metric_tables=True,
        print_per_class_metrics=True,
    )
)
