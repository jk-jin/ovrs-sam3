# -*- coding: utf-8 -*-
"""
Safer iSAID dataset converter for MMSegmentation-style projects.

What this version fixes:
1) Keeps the official iSAID class order / RGB palette / file suffix.
2) Validates that every converted annotation only contains labels in {0..15, 255}.
3) Detects unknown RGB colors in source masks and maps them to ignore_index=255,
   or raises immediately in --strict mode.
4) Handles RGB / RGBA / paletted / grayscale masks more robustly.
5) Tries to collect only real dataset PNG files under an extracted `images/` folder
   before falling back to recursive globbing, reducing accidental pickup of unrelated PNGs.

Official reference:
https://github.com/open-mmlab/mmsegmentation/blob/main/tools/dataset_converters/isaid.py
https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/isaid.py
"""

import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ORIGINAL_SOURCE = (
    'https://github.com/open-mmlab/mmsegmentation/blob/main/'
    'tools/dataset_converters/isaid.py'
)

# Official iSAID palette / class ids in MMSegmentation.
ISAID_PALETTE = {
    0: (0, 0, 0),
    1: (0, 0, 63),
    2: (0, 63, 63),
    3: (0, 63, 0),
    4: (0, 63, 127),
    5: (0, 63, 191),
    6: (0, 63, 255),
    7: (0, 127, 63),
    8: (0, 127, 127),
    9: (0, 0, 127),
    10: (0, 0, 191),
    11: (0, 0, 255),
    12: (0, 191, 127),
    13: (0, 127, 191),
    14: (0, 127, 255),
    15: (0, 100, 155),
}

VALID_LABELS = set(ISAID_PALETTE.keys()) | {255}
IGNORE_INDEX = 255
PNG_COMPRESS_LEVEL = max(0, min(9, int(os.environ.get('OVRS_CONVERTER_PNG_COMPRESS', '1'))))


def _pack_color(color):
    return (np.uint32(color[0]) << 16) | (np.uint32(color[1]) << 8) | np.uint32(color[2])


_ISAID_PACKED_KEYS = np.array(sorted(_pack_color(c) for c in ISAID_PALETTE.values()), dtype=np.uint32)
_ISAID_PACKED_VALS = np.array(
    [
        next(idx for idx, rgb in ISAID_PALETTE.items() if _pack_color(rgb) == key)
        for key in _ISAID_PACKED_KEYS
    ],
    dtype=np.uint8,
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pad_image(arr, target_h, target_w, pad_val):
    if arr.ndim == 3:
        out = np.full((target_h, target_w, arr.shape[2]), pad_val, dtype=arr.dtype)
        out[:arr.shape[0], :arr.shape[1], :] = arr
    else:
        out = np.full((target_h, target_w), pad_val, dtype=arr.dtype)
        out[:arr.shape[0], :arr.shape[1]] = arr
    return out


def _unique_nonzero_rows(arr):
    if arr.size == 0:
        return np.empty((0, arr.shape[1]), dtype=arr.dtype)
    arr = np.ascontiguousarray(arr)
    view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    uniq = np.unique(view)
    return uniq.view(arr.dtype).reshape(-1, arr.shape[1])


def color_mask_to_index(arr, treat_alpha_zero_as_ignore=True):
    """Convert source annotation to label ids.

    Returns:
        label_map: HxW uint8
        unknown_colors: list[(r,g,b)] that were not in official palette
    """
    if arr.ndim == 2:
        # Already a grayscale index map.
        label = arr.astype(np.uint16, copy=False)
        label[(label > 15) & (label != 255)] = 255
        return label.astype(np.uint8), []

    if arr.ndim != 3:
        raise ValueError(f'Unsupported annotation shape: {arr.shape}')

    alpha = None
    if arr.shape[2] >= 4:
        alpha = arr[:, :, 3]
        arr = arr[:, :, :3]
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    flat = arr.reshape(-1, 3).astype(np.uint32, copy=False)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    pos = np.searchsorted(_ISAID_PACKED_KEYS, packed)

    out = np.full((packed.shape[0],), IGNORE_INDEX, dtype=np.uint8)
    in_range = pos < _ISAID_PACKED_KEYS.size
    matched = in_range.copy()
    matched[in_range] = _ISAID_PACKED_KEYS[pos[in_range]] == packed[in_range]
    out[matched] = _ISAID_PACKED_VALS[pos[matched]]

    if alpha is not None and treat_alpha_zero_as_ignore:
        alpha_flat = alpha.reshape(-1)
        out[alpha_flat == 0] = IGNORE_INDEX
        matched[alpha_flat == 0] = True

    unknown = flat[~matched]
    unknown_colors = [tuple(int(x) for x in row.tolist()) for row in _unique_nonzero_rows(unknown)]
    label = out.reshape(arr.shape[0], arr.shape[1])
    return label, unknown_colors


def validate_label_values(label, file_hint):
    uniques = np.unique(label)
    invalid = [int(v) for v in uniques.tolist() if int(v) not in VALID_LABELS]
    if invalid:
        raise ValueError(
            f'Invalid label ids in {file_hint}: {invalid}. '
            f'Valid ids must be 0..15 or 255(ignore_index).'
        )
    return [int(v) for v in uniques.tolist()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmsegmentation-style layout with validation')
    parser.add_argument('dataset_path', help='iSAID folder path')
    parser.add_argument('--tmp_dir', help='Kept only for CLI compatibility; unused in this version')
    parser.add_argument('-o', '--out_dir', help='Output path')
    parser.add_argument('--patch_width', default=896, type=int,
                        help='Width of the cropped image patch')
    parser.add_argument('--patch_height', default=896, type=int,
                        help='Height of the cropped image patch')
    parser.add_argument('--overlap_area', default=384, type=int,
                        help='Overlap area between two patches')
    parser.add_argument('--crop_test', action='store_true',
                        help='Also crop test images. Official mmseg script does not crop test by default.')
    parser.add_argument('--strict', action='store_true',
                        help='Raise an error as soon as an unknown RGB color is seen in a source label.')
    parser.add_argument('--skip_output_validation', action='store_true',
                        help='Skip scanning converted train/val masks after conversion.')
    return parser.parse_args()


def _extract_dir_for_archive(archive_path):
    parent = osp.dirname(archive_path)
    stem = osp.splitext(osp.basename(archive_path))[0]
    return osp.join(parent, f'{stem}_extracted')


def _extract_archive_to_sibling(archive_path):
    extract_root = _extract_dir_for_archive(archive_path)
    if osp.isdir(extract_root) and any(True for _ in os.scandir(extract_root)):
        return extract_root
    if osp.isdir(extract_root):
        shutil.rmtree(extract_root)
    ensure_dir(extract_root)
    print(f'[Info] Extracting {osp.basename(archive_path)} -> {extract_root}')
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)
    return extract_root


def _positions(length, patch, step):
    last = max(length - patch, 0)
    positions = list(range(0, last + 1, step))
    if not positions:
        positions = [0]
    if positions[-1] != last:
        positions.append(last)
    return positions


def iter_patch_ranges(img_h, img_w, patch_h, patch_w, overlap):
    step_x = patch_w - overlap
    step_y = patch_h - overlap
    if step_x <= 0 or step_y <= 0:
        raise ValueError('patch size must be larger than overlap')
    for y_str in _positions(img_h, patch_h, step_y):
        y_end = min(y_str + patch_h, img_h)
        for x_str in _positions(img_w, patch_w, step_x):
            x_end = min(x_str + patch_w, img_w)
            yield int(y_str), int(y_end), int(x_str), int(x_end)


def _save_png(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, compress_level=PNG_COMPRESS_LEVEL)


def _worker_crop_image(task):
    src_path, out_dir, mode, patch_h, patch_w, overlap = task
    img = np.asarray(Image.open(src_path).convert('RGB'))
    img_h, img_w, _ = img.shape
    if img_h < patch_h or img_w < patch_w:
        img = pad_image(img, max(img_h, patch_h), max(img_w, patch_w), 0)
    img_h, img_w = img.shape[:2]
    stem = osp.splitext(osp.basename(src_path))[0]
    saved = 0
    for y_str, y_end, x_str, x_end in iter_patch_ranges(img_h, img_w, patch_h, patch_w, overlap):
        patch = img[y_str:y_end, x_str:x_end, :]
        name = f'{stem}_{y_str}_{y_end}_{x_str}_{x_end}.png'
        _save_png(patch, osp.join(out_dir, 'img_dir', mode, name))
        saved += 1
    return {'file': osp.basename(src_path), 'saved': saved}


def _worker_copy_image(task):
    src_path, out_dir, mode = task
    dst = osp.join(out_dir, 'img_dir', mode, osp.basename(src_path))
    shutil.copy2(src_path, dst)
    return {'file': osp.basename(src_path), 'saved': 1}


def _worker_crop_label(task):
    src_path, out_dir, mode, patch_h, patch_w, overlap, strict = task
    raw = np.asarray(Image.open(src_path))
    label, unknown_colors = color_mask_to_index(raw)
    if strict and unknown_colors:
        raise ValueError(
            f'Unknown RGB colors found in {src_path}: {unknown_colors[:10]}'
            + (' ...' if len(unknown_colors) > 10 else '')
        )

    source_uniques = validate_label_values(label, src_path)

    img_h, img_w = label.shape
    if img_h < patch_h or img_w < patch_w:
        label = pad_image(label, max(img_h, patch_h), max(img_w, patch_w), IGNORE_INDEX)
    img_h, img_w = label.shape

    base = osp.splitext(osp.basename(src_path))[0]
    stem = base.split('_instance_color_RGB')[0].split('_')[0]

    saved = 0
    patch_uniques = set()
    for y_str, y_end, x_str, x_end in iter_patch_ranges(img_h, img_w, patch_h, patch_w, overlap):
        patch = label[y_str:y_end, x_str:x_end]
        validate_label_values(patch, f'{src_path}:{y_str}:{y_end}:{x_str}:{x_end}')
        patch_uniques.update(int(v) for v in np.unique(patch).tolist())
        name = f'{stem}_{y_str}_{y_end}_{x_str}_{x_end}_instance_color_RGB.png'
        _save_png(patch, osp.join(out_dir, 'ann_dir', mode, name))
        saved += 1

    return {
        'file': osp.basename(src_path),
        'saved': saved,
        'source_uniques': source_uniques,
        'patch_uniques': sorted(patch_uniques),
        'unknown_colors': unknown_colors,
    }


def _get_num_workers():
    env = os.environ.get('OVRS_CONVERTER_WORKERS')
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 4))


def _run_tasks(tasks, worker_fn, prefix):
    results = []
    if not tasks:
        return results
    workers = _get_num_workers()
    print(f'  {prefix}: {len(tasks)} files, workers={workers}')
    if workers <= 1 or len(tasks) == 1:
        for i, task in enumerate(tasks, start=1):
            res = worker_fn(task)
            results.append(res)
            print(f"    [{i}/{len(tasks)}] {res['file']} -> {res['saved']}")
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        future_map = {executor.submit(worker_fn, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            res = future.result()
            results.append(res)
            print(f"    [{idx}/{len(tasks)}] {res['file']} -> {res['saved']}")
    return results


def _collect_pngs_from_extracted_dirs(extracted_dirs):
    pngs = []
    # Prefer the official extracted structure: <extract_root>/images/*.png
    for root in extracted_dirs:
        direct = sorted(glob.glob(osp.join(root, 'images', '*.png')))
        if direct:
            pngs.extend(direct)
            continue
        nested = sorted(glob.glob(osp.join(root, '**', 'images', '*.png'), recursive=True))
        if nested:
            pngs.extend(nested)
            continue
        # Fallback for non-standard archives.
        pngs.extend(sorted(glob.glob(osp.join(root, '**', '*.png'), recursive=True)))
    return pngs


def _summarize_label_results(results, split):
    split_uniques = set()
    unknown_by_file = {}
    for res in results:
        split_uniques.update(res.get('patch_uniques', []))
        if res.get('unknown_colors'):
            unknown_by_file[res['file']] = res['unknown_colors']
    print(f'[Check] {split} converted label ids: {sorted(split_uniques)}')
    if unknown_by_file:
        print(f'[Warn] {split} has unknown source RGB colors in {len(unknown_by_file)} file(s).')
        for idx, (fname, colors) in enumerate(sorted(unknown_by_file.items())[:10], start=1):
            preview = colors[:5]
            suffix = ' ...' if len(colors) > 5 else ''
            print(f'  [{idx}] {fname}: {preview}{suffix}')
        print('[Warn] Unknown colors were converted to 255(ignore_index).')


def _validate_converted_output(out_dir, split):
    ann_paths = sorted(glob.glob(osp.join(out_dir, 'ann_dir', split, '*.png')))
    if not ann_paths:
        raise FileNotFoundError(f'No converted annotations found under {out_dir}/ann_dir/{split}')
    all_uniques = set()
    for path in ann_paths:
        label = np.asarray(Image.open(path))
        uniques = validate_label_values(label, path)
        all_uniques.update(uniques)
    print(f'[Check] output ann_dir/{split} unique ids: {sorted(all_uniques)}')


def _check_pairing(out_dir, split):
    img_paths = sorted(glob.glob(osp.join(out_dir, 'img_dir', split, '*.png')))
    ann_paths = sorted(glob.glob(osp.join(out_dir, 'ann_dir', split, '*.png')))
    img_keys = {osp.splitext(osp.basename(p))[0] for p in img_paths}
    ann_keys = {osp.splitext(osp.basename(p))[0].replace('_instance_color_RGB', '') for p in ann_paths}
    missing_ann = sorted(img_keys - ann_keys)
    missing_img = sorted(ann_keys - img_keys)
    if missing_ann:
        raise RuntimeError(f'{split}: {len(missing_ann)} image patches do not have matching annotations. Example: {missing_ann[:5]}')
    if missing_img:
        raise RuntimeError(f'{split}: {len(missing_img)} annotation patches do not have matching images. Example: {missing_img[:5]}')
    print(f'[Check] {split} image/annotation patch pairing is OK: {len(img_keys)} pairs')


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    patch_h, patch_w = args.patch_height, args.patch_width
    overlap = args.overlap_area
    out_dir = args.out_dir or osp.join('data', 'iSAID')

    print(f'[Source] Adapted from: {ORIGINAL_SOURCE}')
    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')
    print('[Info] Archives will be extracted next to the zip files, not to /tmp.')
    print('[Info] Official iSAID setting: 16 classes, ignore_index=255, reduce_zero_label=False, seg suffix=_instance_color_RGB.png')

    ensure_dir(osp.join(out_dir, 'img_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'test'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'test'))

    for split in ['train', 'val', 'test']:
        if not osp.exists(osp.join(dataset_path, split)):
            raise FileNotFoundError(f'{split} is not in {dataset_path}')

    for split in ['train', 'val', 'test']:
        print(f'Processing split: {split}')
        img_archives = sorted(glob.glob(osp.join(dataset_path, split, 'images', '*.zip')))
        if not img_archives:
            raise FileNotFoundError(f'No image archives found under {dataset_path}/{split}/images')
        img_extract_roots = [_extract_archive_to_sibling(p) for p in img_archives]
        img_paths = _collect_pngs_from_extracted_dirs(img_extract_roots)
        if not img_paths:
            raise FileNotFoundError(f'No extracted images found for split={split}')

        if split != 'test' or args.crop_test:
            tasks = [(img_path, out_dir, split, patch_h, patch_w, overlap) for img_path in img_paths]
            _run_tasks(tasks, _worker_crop_image, f'{split} images')
        else:
            tasks = [(img_path, out_dir, split) for img_path in img_paths]
            _run_tasks(tasks, _worker_copy_image, f'{split} images')

        if split != 'test':
            label_archives = sorted(glob.glob(osp.join(dataset_path, split, 'Semantic_masks', '*.zip')))
            if not label_archives:
                raise FileNotFoundError(f'No label archives found under {dataset_path}/{split}/Semantic_masks')
            label_extract_roots = [_extract_archive_to_sibling(p) for p in label_archives]
            label_paths = _collect_pngs_from_extracted_dirs(label_extract_roots)
            if not label_paths:
                raise FileNotFoundError(f'No extracted labels found for split={split}')
            tasks = [(label_path, out_dir, split, patch_h, patch_w, overlap, args.strict) for label_path in label_paths]
            label_results = _run_tasks(tasks, _worker_crop_label, f'{split} labels')
            _summarize_label_results(label_results, split)
            _check_pairing(out_dir, split)

    if not args.skip_output_validation:
        for split in ['train', 'val']:
            _validate_converted_output(out_dir, split)

    print('Done!')


if __name__ == '__main__':
    main()
