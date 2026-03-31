# Toggle validation visualization saving.
# Put this file into configs/_base_/visualization.py and add it to `_base_` if needed.

visualization = dict(
    enabled=True,
    save_dir='visualizations',    # relative paths are resolved under work_dir
    save_stage='val',             # val | train | all
    alpha=0.45,
    threshold=0.5,
    save_original=True,
    save_prediction=True,
    save_ground_truth=True,
    pred_color=(255, 0, 0),
    gt_color=(0, 255, 0),
    max_samples=None,             # e.g. 50 for quick debug
    image_folder_pattern='image_{image_id:06d}',
    prompt_in_filename=True,
)
