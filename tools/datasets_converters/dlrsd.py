# -*- coding: utf-8 -*-
"""
DLRSD dataset converter for MMSegmentation-style projects.

What this script does:
1) Lets you place one or more DLRSD zip files under the dataset root
   (or dataset_root/raw/) and extracts them in place.
2) Collects RGB images and semantic masks from the extracted contents.
3) Converts DLRSD annotations to contiguous MMSeg labels in {0..16, 255}.
4) Creates a deterministic train/val split plus an `all` split.
5) Organizes the output as:
      out_dir/
        img_dir/{train,val,all}/*
        ann_dir/{train,val,all}/*.png
        splits/{train,val,all}.txt

Class order / palette are aligned with OVRS's DLRSD registration.
Reference:
https://github.com/caoql98/OVRS/blob/main/cat_seg/data/datasets/register_DLRSD.py
"""

import argparse
import glob
import hashlib
import multiprocessing as mp
import os
import os.path as osp
import re
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ORIGINAL_REFERENCE = (
    'https://github.com/caoql98/OVRS/blob/main/'
    'cat_seg/data/datasets/register_DLRSD.py'
)

# ids in the public OVRS registration are 1..17.
DLRSD_CATEGORIES = [
    {"color": (166, 202, 240), "id": 1, "name": "airplane"},
    {"color": (128, 128, 0), "id": 2, "name": "bare soil"},
    {"color": (0, 0, 128), "id": 3, "name": "buildings"},
    {"color": (255, 0, 0), "id": 4, "name": "cars"},
    {"color": (0, 128, 0), "id": 5, "name": "chaparral"},
    {"color": (128, 0, 0), "id": 6, "name": "court"},
    {"color": (255, 233, 233), "id": 7, "name": "dock"},
    {"color": (160, 160, 164), "id": 8, "name": "field"},
    {"color": (0, 128, 128), "id": 9, "name": "grass"},
    {"color": (90, 87, 255), "id": 10, "name": "mobile home"},
    {"color": (255, 255, 0), "id": 11, "name": "pavement"},
    {"color": (255, 192, 0), "id": 12, "name": "sand"},
    {"color": (0, 0, 255), "id": 13, "name": "sea"},
    {"color": (255, 0, 192), "id": 14, "name": "ship"},
    {"color": (128, 0, 128), "id": 15, "name": "tanks"},
    {"color": (0, 255, 0), "id": 16, "name": "trees"},
    {"color": (0, 255, 255), "id": 17, "name": "water"},
]

# contiguous MMSeg labels are 0..16.
DLRSD_PALETTE = {cat["id"] - 1: cat["color"] for cat in DLRSD_CATEGORIES}
VALID_LABELS = set(DLRSD_PALETTE.keys()) | {255}
IGNORE_INDEX = 255
PNG_COMPRESS_LEVEL = max(0, min(9, int(os.environ.get('OVRS_CONVERTER_PNG_COMPRESS', '1'))))
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
MASK_EXTS = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DLRSD dataset to mmsegmentation-style layout with validation')
    parser.add_argument('dataset_path', help='DLRSD root folder path')
    parser.add_argument('-o', '--out_dir', help='Output path')
    parser.add_argument('--seed', default=42, type=int,
                        help='Deterministic seed for train/val split')
    parser.add_argument('--val_ratio', default=0.2, type=float,
                        help='Validation ratio inside each source group')
    parser.add_argument('--strict', action='store_true',
                        help='Raise an error when an unknown mask color/value is found.')
    parser.add_argument('--skip_output_validation', action='store_true',
                        help='Skip scanning converted masks after conversion.')
    parser.add_argument('--force_extract', action='store_true',
                        help='Re-extract zip archives even if extracted folders already exist.')
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _pack_color(color):
    return (np.uint32(color[0]) << 16) | (np.uint32(color[1]) << 8) | np.uint32(color[2])


_DLRSD_PACKED_KEYS = np.array(sorted(_pack_color(c) for c in DLRSD_PALETTE.values()), dtype=np.uint32)
_DLRSD_PACKED_VALS = np.array(
    [
        next(idx for idx, rgb in DLRSD_PALETTE.items() if _pack_color(rgb) == key)
        for key in _DLRSD_PACKED_KEYS
    ],
    dtype=np.uint8,
)


def _extract_dir_for_archive(archive_path):
    parent = osp.dirname(archive_path)
    stem = osp.splitext(osp.basename(archive_path))[0]
    return osp.join(parent, f'{stem}_extracted')


def _extract_archive_to_sibling(archive_path, force_extract=False):
    extract_root = _extract_dir_for_archive(archive_path)
    if osp.isdir(extract_root) and any(True for _ in os.scandir(extract_root)) and not force_extract:
        return extract_root
    if osp.isdir(extract_root):
        shutil.rmtree(extract_root)
    ensure_dir(extract_root)
    print(f'[Info] Extracting {osp.basename(archive_path)} -> {extract_root}')
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)
    return extract_root


def _find_archives(dataset_path):
    candidates = []
    for root in [dataset_path, osp.join(dataset_path, 'raw')]:
        if osp.isdir(root):
            candidates.extend(sorted(glob.glob(osp.join(root, '*.zip'))))
            candidates.extend(sorted(glob.glob(osp.join(root, '**', '*.zip'), recursive=True)))
    seen = set()
    out = []
    for path in candidates:
        real = osp.realpath(path)
        if real not in seen:
            seen.add(real)
            out.append(path)
    return out


def _normalized_stem(path):
    stem = Path(path).stem
    stem = re.sub(r'(_mask|_label|_labels|_gt|_seg|_annotation|_annot|_ann)$', '', stem, flags=re.IGNORECASE)
    return stem.lower()


def _is_mask_like(path):
    lower = path.lower()
    parent_tokens = set(re.split(r'[^a-z0-9]+', lower))
    return any(tok in parent_tokens for tok in ['mask', 'masks', 'label', 'labels', 'gt', 'ann', 'annotation', 'annotations'])


def _is_image_like(path):
    lower = path.lower()
    parent_tokens = set(re.split(r'[^a-z0-9]+', lower))
    return any(tok in parent_tokens for tok in ['image', 'images', 'img', 'imgs', 'jpegimages', 'rgb'])


def _collect_files(extracted_dirs):
    all_files = []
    for root in extracted_dirs:
        for path in glob.glob(osp.join(root, '**', '*'), recursive=True):
            if osp.isfile(path):
                all_files.append(path)

    images = []
    masks = []
    for path in all_files:
        suffix = Path(path).suffix.lower()
        if suffix in IMAGE_EXTS:
            if _is_mask_like(path) and suffix not in {'.jpg', '.jpeg'}:
                masks.append(path)
            elif _is_image_like(path):
                images.append(path)
            else:
                if suffix in {'.jpg', '.jpeg'}:
                    images.append(path)
                else:
                    masks.append(path)
    return sorted(images), sorted(masks)


def _build_best_match_dict(paths):
    by_stem = defaultdict(list)
    by_name = defaultdict(list)
    for path in paths:
        by_stem[_normalized_stem(path)].append(path)
        by_name[Path(path).name.lower()].append(path)
    return by_stem, by_name


def _score_path(path):
    lower = path.lower()
    score = 0
    if _is_image_like(lower):
        score += 3
    if _is_mask_like(lower):
        score += 2
    score -= lower.count('multilabel') * 5
    score -= lower.count('thumbnail') * 5
    score -= lower.count('preview') * 5
    return score


def _pair_images_and_masks(image_paths, mask_paths):
    if not image_paths:
        raise FileNotFoundError('No candidate images found in extracted DLRSD archives.')
    if not mask_paths:
        raise FileNotFoundError('No candidate masks found in extracted DLRSD archives.')

    mask_by_stem, mask_by_name = _build_best_match_dict(mask_paths)
    pairs = {}
    missing = []
    for img_path in image_paths:
        stem = _normalized_stem(img_path)
        candidates = list(mask_by_stem.get(stem, []))
        if not candidates:
            base = Path(img_path).stem.lower()
            candidates = list(mask_by_name.get(base + '.png', [])) + list(mask_by_name.get(base + '.tif', []))
        if not candidates:
            missing.append(img_path)
            continue
        candidates = sorted(candidates, key=lambda p: (_score_path(p), p), reverse=True)
        pairs[img_path] = candidates[0]

    if not pairs:
        raise RuntimeError('Failed to pair any DLRSD images with masks from the extracted contents.')

    if missing:
        print(f'[Warn] {len(missing)} image(s) could not be paired with masks. Example: {missing[:5]}')
    print(f'[Check] Paired {len(pairs)} image/mask files.')
    return pairs


def _safe_group_name_from_parent(path):
    parent = Path(path).parent.name.strip().lower()
    if parent and parent not in {'images', 'image', 'imgs', 'img', 'jpegimages', 'labels', 'label', 'masks', 'mask'}:
        return re.sub(r'[^a-z0-9]+', '_', parent).strip('_') or 'default'
    stem = Path(path).stem.lower()
    m = re.match(r'([a-z_\-]+?)(?:\d+.*)?$', stem)
    if m:
        prefix = re.sub(r'[^a-z0-9]+', '_', m.group(1)).strip('_')
        if prefix:
            return prefix
    return 'default'


def _deterministic_shuffle(items, seed):
    def key_fn(x):
        digest = hashlib.sha1(f'{seed}:{x}'.encode('utf-8')).hexdigest()
        return digest
    return sorted(items, key=key_fn)


def build_split(pairs, val_ratio, seed):
    grouped = defaultdict(list)
    for img_path in pairs:
        grouped[_safe_group_name_from_parent(img_path)].append(img_path)

    train, val = [], []
    for group, paths in sorted(grouped.items()):
        ordered = _deterministic_shuffle(paths, seed)
        n_total = len(ordered)
        n_val = int(round(n_total * val_ratio))
        if n_total > 1:
            n_val = max(1, min(n_total - 1, n_val))
        else:
            n_val = 0
        val_part = ordered[:n_val]
        train_part = ordered[n_val:]
        val.extend(val_part)
        train.extend(train_part)
        print(f'[Split] group={group:<24} total={n_total:<4} train={len(train_part):<4} val={len(val_part):<4}')

    train = sorted(train)
    val = sorted(val)
    all_items = sorted(pairs)
    if not train:
        raise RuntimeError('Empty train split after DLRSD split generation.')
    if not val:
        print('[Warn] Validation split is empty. This can happen if the dataset is extremely small.')
    return {'train': train, 'val': val, 'all': all_items}


def _unique_nonzero_rows(arr):
    if arr.size == 0:
        return np.empty((0, arr.shape[1]), dtype=arr.dtype)
    arr = np.ascontiguousarray(arr)
    view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    uniq = np.unique(view)
    return uniq.view(arr.dtype).reshape(-1, arr.shape[1])


def color_mask_to_index(arr, treat_alpha_zero_as_ignore=True):
    if arr.ndim == 2:
        label = arr.astype(np.int32, copy=False)
        uniques = set(int(v) for v in np.unique(label).tolist())
        if uniques.issubset(set(range(17)) | {255}):
            return label.astype(np.uint8), []
        if uniques.issubset(set(range(1, 18)) | {255}):
            label = np.where(label == 255, 255, label - 1)
            return label.astype(np.uint8), []
        if uniques.issubset(set(range(18)) | {255}):
            label = label.copy()
            label[label == 0] = 255
            valid = (label >= 1) & (label <= 17)
            label[valid] = label[valid] - 1
            label[(label > 16) & (label != 255)] = 255
            return label.astype(np.uint8), []
        bad = sorted(v for v in uniques if v not in set(range(17)) | set(range(1, 18)) | {255})
        return np.where((label >= 0) & (label < 17), label, 255).astype(np.uint8), bad

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
    pos = np.searchsorted(_DLRSD_PACKED_KEYS, packed)

    out = np.full((packed.shape[0],), IGNORE_INDEX, dtype=np.uint8)
    in_range = pos < _DLRSD_PACKED_KEYS.size
    matched = in_range.copy()
    matched[in_range] = _DLRSD_PACKED_KEYS[pos[in_range]] == packed[in_range]
    out[matched] = _DLRSD_PACKED_VALS[pos[matched]]

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
            f'Valid ids must be 0..16 or 255(ignore_index).'
        )
    return [int(v) for v in uniques.tolist()]


def _save_mask(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, compress_level=PNG_COMPRESS_LEVEL)


def _worker_copy_and_convert(task):
    img_path, mask_path, out_dir, split, strict = task
    stem = Path(img_path).stem
    img_ext = Path(img_path).suffix.lower()
    if img_ext not in {'.jpg', '.jpeg'}:
        img_ext = '.jpg'

    img_dst = osp.join(out_dir, 'img_dir', split, stem + img_ext)
    ann_dst = osp.join(out_dir, 'ann_dir', split, stem + '.png')

    img = Image.open(img_path).convert('RGB')
    img.save(img_dst, quality=95)

    raw = np.asarray(Image.open(mask_path))
    label, unknown = color_mask_to_index(raw)
    if strict and unknown:
        raise ValueError(
            f'Unknown values/colors found in {mask_path}: {unknown[:10]}'
            + (' ...' if len(unknown) > 10 else '')
        )
    uniques = validate_label_values(label, mask_path)
    _save_mask(label, ann_dst)
    return {
        'file': Path(img_path).name,
        'split': split,
        'saved': 1,
        'unknown': unknown,
        'uniques': uniques,
    }


def _get_num_workers():
    env = os.environ.get('OVRS_CONVERTER_WORKERS')
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


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
            print(f"    [{i}/{len(tasks)}] {res['file']}")
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        future_map = {executor.submit(worker_fn, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            res = future.result()
            results.append(res)
            print(f"    [{idx}/{len(tasks)}] {res['file']}")
    return results


def _summarize_results(results, split):
    split_uniques = set()
    unknown_by_file = {}
    for res in results:
        split_uniques.update(res.get('uniques', []))
        if res.get('unknown'):
            unknown_by_file[res['file']] = res['unknown']
    print(f'[Check] {split} converted label ids: {sorted(split_uniques)}')
    if unknown_by_file:
        print(f'[Warn] {split} has unknown source values/colors in {len(unknown_by_file)} file(s).')
        for idx, (fname, colors) in enumerate(sorted(unknown_by_file.items())[:10], start=1):
            preview = colors[:5]
            suffix = ' ...' if len(colors) > 5 else ''
            print(f'  [{idx}] {fname}: {preview}{suffix}')
        print('[Warn] Unknown source values/colors were converted to 255(ignore_index).')


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
    img_paths = sorted(glob.glob(osp.join(out_dir, 'img_dir', split, '*')))
    ann_paths = sorted(glob.glob(osp.join(out_dir, 'ann_dir', split, '*.png')))
    img_keys = {Path(p).stem for p in img_paths}
    ann_keys = {Path(p).stem for p in ann_paths}
    missing_ann = sorted(img_keys - ann_keys)
    missing_img = sorted(ann_keys - img_keys)
    if missing_ann:
        raise RuntimeError(f'{split}: {len(missing_ann)} images do not have matching annotations. Example: {missing_ann[:5]}')
    if missing_img:
        raise RuntimeError(f'{split}: {len(missing_img)} annotations do not have matching images. Example: {missing_img[:5]}')
    print(f'[Check] {split} image/annotation pairing is OK: {len(img_keys)} pairs')


def _write_split_file(stems, path):
    ensure_dir(osp.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        for stem in stems:
            f.write(stem + '\n')


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir or osp.join('data', 'DLRSD')

    if not osp.isdir(dataset_path):
        raise FileNotFoundError(f'dataset_path does not exist: {dataset_path}')

    print(f'[Source] Reference metadata from: {ORIGINAL_REFERENCE}')
    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')
    print('[Info] DLRSD target setting: 17 classes -> contiguous ids 0..16, ignore_index=255')
    print('[Info] Place official DLRSD zip file(s) under dataset_path/ or dataset_path/raw/.')

    for split in ['train', 'val', 'all']:
        ensure_dir(osp.join(out_dir, 'img_dir', split))
        ensure_dir(osp.join(out_dir, 'ann_dir', split))
    ensure_dir(osp.join(out_dir, 'splits'))

    archives = _find_archives(dataset_path)
    if not archives:
        raise FileNotFoundError(
            f'No zip archives found under {dataset_path} or {osp.join(dataset_path, "raw")}. '
            'Please place the downloaded DLRSD zip file(s) there first.'
        )
    print(f'[Info] Found {len(archives)} archive(s).')
    extract_roots = [_extract_archive_to_sibling(p, force_extract=args.force_extract) for p in archives]

    image_paths, mask_paths = _collect_files(extract_roots)
    print(f'[Info] Candidate images: {len(image_paths)} | candidate masks: {len(mask_paths)}')
    pairs = _pair_images_and_masks(image_paths, mask_paths)

    split_map = build_split(pairs, val_ratio=args.val_ratio, seed=args.seed)

    for split, img_list in split_map.items():
        print(f'Processing split: {split}')
        tasks = [(img_path, pairs[img_path], out_dir, split, args.strict) for img_path in img_list]
        results = _run_tasks(tasks, _worker_copy_and_convert, f'{split} files')
        _summarize_results(results, split)
        _check_pairing(out_dir, split)
        stems = [Path(p).stem for p in img_list]
        _write_split_file(stems, osp.join(out_dir, 'splits', f'{split}.txt'))

    if not args.skip_output_validation:
        for split in ['train', 'val', 'all']:
            _validate_converted_output(out_dir, split)

    print('Done!')


if __name__ == '__main__':
    main()
