# -*- coding: utf-8 -*-
"""
Custom Vaihingen converter for OVRS-SAM3.

Important change in this version:
all archives are extracted to directories next to the archive files instead of
using the system temporary directory.

This version outputs 0-5 labels directly in the following order:
0 impervious_surface
1 building
2 low_vegetation
3 tree
4 car
5 clutter_background

Use reduce_zero_label=False with this version.
"""

import argparse
import glob
import os
import os.path as osp
import re
import shutil
import zipfile

import numpy as np
from PIL import Image

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
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
], dtype=np.uint8)

_AREA_RE = re.compile(r'area(\d+)', re.IGNORECASE)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Vaihingen dataset to mmsegmentation-style layout')
    parser.add_argument('dataset_path', help='Vaihingen folder path')
    parser.add_argument('--tmp_dir', help='Kept only for CLI compatibility; unused in this version')
    parser.add_argument('-o', '--out_dir', help='Output path')
    parser.add_argument('--clip_size', type=int, default=512,
                        help='Clipped size after preparation')
    parser.add_argument('--stride_size', type=int, default=256,
                        help='kept for CLI compatibility; not used for noBoundary window layout in this version')
    return parser.parse_args()


def _extract_dir_for_archive(archive_path):
    parent = osp.dirname(archive_path)
    stem = osp.splitext(osp.basename(archive_path))[0]
    return osp.join(parent, f'{stem}_extracted')


def _extract_archive_to_sibling(archive_path):
    extract_root = _extract_dir_for_archive(archive_path)
    if osp.isdir(extract_root) and any(True for _ in os.scandir(extract_root)):
        print(f'[Info] Reuse extracted directory: {extract_root}')
        return extract_root
    if osp.isdir(extract_root):
        shutil.rmtree(extract_root)
    ensure_dir(extract_root)
    print(f'[Info] Extracting {osp.basename(archive_path)} -> {extract_root}')
    with zipfile.ZipFile(archive_path, 'r') as zf:
        zf.extractall(extract_root)
    return extract_root


def find_tif_files(root_dir):
    files = glob.glob(osp.join(root_dir, '**', '*.tif'), recursive=True)
    files += glob.glob(osp.join(root_dir, '**', '*.tiff'), recursive=True)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f'No tif files found under: {root_dir}')
    return files


def rgb_label_to_index(rgb):
    h, w, _ = rgb.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    for idx, color in enumerate(COLOR_MAP):
        out[np.all(rgb == color.reshape(1, 1, 3), axis=2)] = idx
    return out


def iter_starts(length, clip_size):
    if length <= clip_size:
        return [0]
    starts = list(range(0, length - clip_size + 1, clip_size))
    last = length - clip_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_boxes(h, w, clip_size):
    for start_y in iter_starts(h, clip_size):
        end_y = start_y + clip_size
        for start_x in iter_starts(w, clip_size):
            end_x = start_x + clip_size
            yield start_x, start_y, end_x, end_y


def _parse_area_id(path):
    match = _AREA_RE.search(osp.basename(path))
    if match is None:
        raise ValueError(f'Cannot parse Vaihingen area id from file name: {path}')
    return f'area{match.group(1)}'


def _is_label_path(path, archive_name):
    name = osp.basename(path).lower()
    archive_name = archive_name.lower()
    return ('noboundary' in name) or ('label' in name) or ('ground_truth' in archive_name)


def save_array(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, compress_level=1)


def process_file(src_path, archive_name, out_dir, clip_size):
    area_id = _parse_area_id(src_path)
    if area_id not in SPLITS['train'] and area_id not in SPLITS['val']:
        return osp.basename(src_path), 'skip', 0, 'unknown'
    split = 'train' if area_id in SPLITS['train'] else 'val'
    is_label = _is_label_path(src_path, archive_name)
    image = np.asarray(Image.open(src_path).convert('RGB'))
    if is_label:
        image = rgb_label_to_index(image)
    h, w = image.shape[:2]
    dst_dir = osp.join(out_dir, 'ann_dir' if is_label else 'img_dir', split)
    saved = 0
    for start_x, start_y, end_x, end_y in build_boxes(h, w, clip_size):
        patch = image[start_y:end_y, start_x:end_x] if is_label else image[start_y:end_y, start_x:end_x, :]
        file_name = f'{area_id}_{start_x}_{start_y}_{end_x}_{end_y}.png'
        save_array(patch, osp.join(dst_dir, file_name))
        saved += 1
    return osp.basename(src_path), split, saved, 'label' if is_label else 'image'


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir or osp.join('data', 'vaihingen')

    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')
    print('[Info] Archives will be extracted next to the zip files, not to /tmp.')
    print('[Info] Use reduce_zero_label=False with this version.')

    ensure_dir(osp.join(out_dir, 'img_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'val'))

    zip_list = sorted(glob.glob(osp.join(dataset_path, '*.zip')))
    if not zip_list:
        raise FileNotFoundError(f'No zip files found in {dataset_path}')

    for zipp in zip_list:
        print(f'Processing archive: {osp.basename(zipp)}')
        extract_root = _extract_archive_to_sibling(zipp)
        tif_files = find_tif_files(extract_root)
        for idx, src_path in enumerate(tif_files, start=1):
            name, split, saved, kind = process_file(src_path, osp.basename(zipp), out_dir, args.clip_size)
            if split != 'skip':
                print(f'  [{idx}/{len(tif_files)}] {name} -> {split} ({saved} {kind} patches)')
    print('Done!')


if __name__ == '__main__':
    main()
