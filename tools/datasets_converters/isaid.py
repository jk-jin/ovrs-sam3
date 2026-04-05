# -*- coding: utf-8 -*-
"""
Source notice
-------------
This file is adapted for the user's OVRS-SAM3 project from the official
MMSegmentation dataset converter style.

Original project: OpenMMLab / MMSegmentation
License of original project: Apache License 2.0
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

ORIGINAL_SOURCE = (
    'https://github.com/open-mmlab/mmsegmentation/blob/main/'
    'tools/dataset_converters/isaid.py'
)

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

Image.MAX_IMAGE_PIXELS = None
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


def color_to_index(arr_3d):
    if arr_3d.ndim == 2:
        return arr_3d.astype(np.uint8)
    flat = arr_3d.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    pos = np.searchsorted(_ISAID_PACKED_KEYS, packed)
    out = np.full((packed.shape[0],), 255, dtype=np.uint8)
    valid = pos < _ISAID_PACKED_KEYS.size
    valid &= (_ISAID_PACKED_KEYS[pos.clip(max=_ISAID_PACKED_KEYS.size - 1)] == packed)
    out[valid] = _ISAID_PACKED_VALS[pos[valid]]
    return out.reshape(arr_3d.shape[0], arr_3d.shape[1])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmsegmentation-style layout')
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
    return osp.basename(src_path), saved


def _worker_copy_image(task):
    src_path, out_dir, mode = task
    dst = osp.join(out_dir, 'img_dir', mode, osp.basename(src_path))
    shutil.copy2(src_path, dst)
    return osp.basename(src_path), 1


def _worker_crop_label(task):
    src_path, out_dir, mode, patch_h, patch_w, overlap = task
    label_rgb = np.asarray(Image.open(src_path).convert('RGB'))
    label = color_to_index(label_rgb)
    img_h, img_w = label.shape
    if img_h < patch_h or img_w < patch_w:
        label = pad_image(label, max(img_h, patch_h), max(img_w, patch_w), 255)
    img_h, img_w = label.shape
    stem = osp.splitext(osp.basename(src_path))[0].split('_')[0]
    saved = 0
    for y_str, y_end, x_str, x_end in iter_patch_ranges(img_h, img_w, patch_h, patch_w, overlap):
        patch = label[y_str:y_end, x_str:x_end]
        name = f'{stem}_{y_str}_{y_end}_{x_str}_{x_end}_instance_color_RGB.png'
        _save_png(patch, osp.join(out_dir, 'ann_dir', mode, name))
        saved += 1
    return osp.basename(src_path), saved


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
    if not tasks:
        return
    workers = _get_num_workers()
    print(f'  {prefix}: {len(tasks)} files, workers={workers}')
    if workers <= 1 or len(tasks) == 1:
        for i, task in enumerate(tasks, start=1):
            name, saved = worker_fn(task)
            print(f'    [{i}/{len(tasks)}] {name} -> {saved}')
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        future_map = {executor.submit(worker_fn, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            name, saved = future.result()
            print(f'    [{idx}/{len(tasks)}] {name} -> {saved}')


def _collect_pngs_from_extracted_dirs(extracted_dirs):
    pngs = []
    for root in extracted_dirs:
        pngs.extend(sorted(glob.glob(osp.join(root, '**', '*.png'), recursive=True)))
    return pngs


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
        print(f'Extracting split: {split}')
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
            tasks = [(label_path, out_dir, split, patch_h, patch_w, overlap) for label_path in label_paths]
            _run_tasks(tasks, _worker_crop_label, f'{split} labels')
    print('Done!')


if __name__ == '__main__':
    main()
