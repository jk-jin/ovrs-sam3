visualization = dict(
    enabled=True,
    save_dir='visualizations',
    save_stage='val',
    alpha=0.45,

    save_original=True,
    save_prediction=True,
    save_ground_truth=True,
    save_dense_prediction=True,

    max_samples=None,
    image_folder_pattern='image_{image_id:06d}',
    ignore_index=255,
)