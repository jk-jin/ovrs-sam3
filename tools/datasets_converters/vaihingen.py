# -*- coding: utf-8 -*-
"""
Source notice
-------------
This file is adapted for the OVRS-SAM3 project from the official
MMSegmentation dataset converter style.

Original project: OpenMMLab / MMSegmentation
Original license: Apache License 2.0

This rewritten version removes the runtime dependency on mmseg/mmcv/mmengine
and only uses Python standard library + NumPy + Pillow so it can be used
directly inside the current project.

Please keep this notice when redistributing the file.
"""


import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


ORIGINAL_SOURCE = (
    'https://github.com/open-mmlab/mmsegmentation/blob/main/'
    'tools/dataset_converters/vaihingen.py'
)

SPLITS = {
    'train': [
        'area1', 'area11', 'area13', 'area15', 'area17', 'area21',
        'area23', 'area26', 'area28', 'area3', 'area30', 'area32',
        'area34', 'area37', 'area5', 'area7'
    ],
    'val': [
        'area6', 'area24', 'area35', 'area16', 'area14', 'area22',
        'area10', 'area4', 'area2', 'area20', 'area8', 'area31',
        'area33', 'area27', 'area38', 'area12', 'area29'
    ],
}

COLOR_MAP = np.array([
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
], dtype=np.uint8)

Image.MAX_IMAGE_PIXELS = None
PNG_COMPRESS_LEVEL = max(0, min(9, int(os.environ.get('OVRS_CONVERTER_PNG_COMPRESS', '1'))))
_AREA_RE = re.compile(r'area(\d+)', re.IGNORECASE)


def _pack_rgb_array(rgb):
    flat = rgb.reshape(-1, 3).astype(np.uint32)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def _pack_rgb_color(color):
    return (np.uint32(color[0]) << 16) | (np.uint32(color[1]) << 8) | np.uint32(color[2])


_COLOR_KEYS = np.array(sorted(_pack_rgb_color(color) for color in COLOR_MAP.tolist()), dtype=np.uint32)
_COLOR_VALS = np.array(
    [next(i for i, rgb in enumerate(COLOR_MAP.tolist()) if _pack_rgb_color(rgb) == key) for key in _COLOR_KEYS],
    dtype=np.uint8,
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def save_array(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, compress_level=PNG_COMPRESS_LEVEL)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Vaihingen dataset to mmsegmentation-style layout')
    parser.add_argument('dataset_path', help='Vaihingen folder path')
    parser.add_argument('--tmp_dir', help='Temporary directory root')
    parser.add_argument('-o', '--out_dir', help='Output path')
    parser.add_argument('--clip_size', type=int, default=512,
                        help='Clipped size after preparation')
    parser.add_argument('--stride_size', type=int, default=256,
                        help='Stride argument kept for CLI compatibility with mmseg')
    return parser.parse_args()


def find_tif_files(root_dir):
    return sorted(glob.glob(osp.join(root_dir, '**', '*.tif'), recursive=True))


def rgb_label_to_index(rgb):
    packed = _pack_rgb_array(rgb)
    pos = np.searchsorted(_COLOR_KEYS, packed)
    out = np.full((packed.shape[0],), 0, dtype=np.uint8)
    valid = pos < _COLOR_KEYS.size
    valid &= (_COLOR_KEYS[pos.clip(max=_COLOR_KEYS.size - 1)] == packed)
    out[valid] = _COLOR_VALS[pos[valid]]
    return out.reshape(rgb.shape[0], rgb.shape[1])


def _positions(length, patch, stride):
    last = max(length - patch, 0)
    positions = list(range(0, last + 1, stride))
    if not positions:
        positions = [0]
    if positions[-1] != last:
        positions.append(last)
    return positions


def build_boxes(h, w, clip_size, stride_size):
    boxes = []
    for start_y in _positions(h, clip_size, stride_size):
        end_y = min(start_y + clip_size, h)
        for start_x in _positions(w, clip_size, stride_size):
            end_x = min(start_x + clip_size, w)
            boxes.append((start_x, start_y, end_x, end_y))
    return boxes


def _parse_area_id(path):
    match = _AREA_RE.search(osp.basename(path))
    if match is None:
        raise ValueError(f'Cannot parse Vaihingen area id from file name: {path}')
    return f'area{match.group(1)}'


def _is_label_path(path, archive_name):
    name = osp.basename(path).lower()
    archive_name = archive_name.lower()
    return ('noboundary' in name) or ('label' in name) or ('ground_truth' in archive_name)


def _pad_if_needed(arr, clip_size):
    h, w = arr.shape[:2]
    target_h = max(h, clip_size)
    target_w = max(w, clip_size)
    if target_h == h and target_w == w:
        return arr
    if arr.ndim == 3:
        out = np.full((target_h, target_w, arr.shape[2]), 0, dtype=arr.dtype)
        out[:h, :w, :] = arr
    else:
        out = np.full((target_h, target_w), 0, dtype=arr.dtype)
        out[:h, :w] = arr
    return out


def _worker_process(task):
    src_path, archive_name, out_dir, clip_size, stride_size = task
    area_id = _parse_area_id(src_path)
    if area_id not in SPLITS['train'] and area_id not in SPLITS['val']:
        return osp.basename(src_path), 'skip', 0, 'unknown'
    split = 'train' if area_id in SPLITS['train'] else 'val'
    is_label = _is_label_path(src_path, archive_name)
    image = load_rgb(src_path)
    if is_label:
        image = rgb_label_to_index(image)
    image = _pad_if_needed(image, clip_size)
    h, w = image.shape[:2]
    boxes = build_boxes(h, w, clip_size, stride_size)
    dst_dir = osp.join(out_dir, 'ann_dir' if is_label else 'img_dir', split)
    saved = 0
    for start_x, start_y, end_x, end_y in boxes:
        patch = image[start_y:end_y, start_x:end_x] if is_label else image[start_y:end_y, start_x:end_x, :]
        file_name = f'{area_id}_{start_x}_{start_y}_{end_x}_{end_y}.png'
        save_array(patch, osp.join(dst_dir, file_name))
        saved += 1
    return osp.basename(src_path), split, saved, 'label' if is_label else 'image'


def _get_num_workers():
    env = os.environ.get('OVRS_CONVERTER_WORKERS')
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


def _run_archive_tasks(tasks):
    workers = _get_num_workers()
    print(f'  files={len(tasks)}, workers={workers}')
    if workers <= 1 or len(tasks) == 1:
        for idx, task in enumerate(tasks, start=1):
            name, split, saved, kind = _worker_process(task)
            if split != 'skip':
                print(f'  [{idx}/{len(tasks)}] {name} -> {split} ({saved} {kind} patches)')
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as executor:
        futures = [executor.submit(_worker_process, task) for task in tasks]
        shown = 0
        for future in as_completed(futures):
            name, split, saved, kind = future.result()
            if split == 'skip':
                continue
            shown += 1
            print(f'  [{shown}/{len(tasks)}] {name} -> {split} ({saved} {kind} patches)')


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir or osp.join('data', 'vaihingen')

    print(f'[Source] Adapted from: {ORIGINAL_SOURCE}')
    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')

    ensure_dir(osp.join(out_dir, 'img_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'val'))

    zip_list = sorted(glob.glob(osp.join(dataset_path, '*.zip')))
    if not zip_list:
        raise FileNotFoundError(f'No zip files found in {dataset_path}')

    for zipp in zip_list:
        print(f'Processing archive: {osp.basename(zipp)}')
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            with zipfile.ZipFile(zipp) as zf:
                zf.extractall(tmp_dir)
            src_path_list = [p for p in find_tif_files(tmp_dir) if 'area9' not in osp.basename(p).lower()]
            if not src_path_list:
                raise FileNotFoundError(f'No .tif files found after extracting {zipp}')
            tasks = [(src_path, osp.basename(zipp), out_dir, args.clip_size, args.stride_size) for src_path in src_path_list]
            _run_archive_tasks(tasks)
    print('Done!')


if __name__ == '__main__':
    main()
