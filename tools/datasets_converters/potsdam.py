# -*- coding: utf-8 -*-
"""
Custom Potsdam converter for OVRS-SAM3.

This version is written for the noBoundary labels package:
    5_Labels_all_noBoundary.zip

Important change in this version:
all archives are extracted to directories next to the archive files instead of
using the system temporary directory.

The generated label ids are 0-5 directly:
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
import shutil
import zipfile

import numpy as np
from PIL import Image

TRAIN_SPLIT = {
    '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
    '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
    '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
}
VAL_SPLIT = {
    '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
    '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
}

COLOR_MAP = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format (noBoundary labels)')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='Kept only for CLI compatibility; unused in this version')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument('--clip_size', type=int, default=512,
                        help='clipped size of image after preparation')
    parser.add_argument('--stride_size', type=int, default=256,
                        help='kept for CLI compatibility; not used for noBoundary window layout in this version')
    return parser.parse_args()


def mkdir_or_exist(path):
    os.makedirs(path, exist_ok=True)


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
    mkdir_or_exist(extract_root)
    print(f'[Info] Extracting {osp.basename(archive_path)} -> {extract_root}')
    with zipfile.ZipFile(archive_path, 'r') as zf:
        zf.extractall(extract_root)
    return extract_root


def find_zip_files(dataset_path):
    all_zips = sorted(glob.glob(osp.join(dataset_path, '*.zip')))
    rgb_zips = [p for p in all_zips if '2_Ortho_RGB' in osp.basename(p)]
    label_zips = [p for p in all_zips if '5_Labels_all_noBoundary' in osp.basename(p)]
    if not rgb_zips:
        raise FileNotFoundError(f'Cannot find 2_Ortho_RGB.zip under: {dataset_path}')
    if not label_zips:
        raise FileNotFoundError(f'Cannot find 5_Labels_all_noBoundary.zip under: {dataset_path}')
    return rgb_zips + label_zips


def find_tifs(root_dir):
    tif_list = glob.glob(osp.join(root_dir, '**', '*.tif'), recursive=True)
    tif_list += glob.glob(osp.join(root_dir, '**', '*.tiff'), recursive=True)
    tif_list = sorted(set(tif_list))
    if not tif_list:
        raise FileNotFoundError(f'No tif files found under: {root_dir}')
    return tif_list


def parse_tile_id(src_path):
    stem = osp.splitext(osp.basename(src_path))[0]
    parts = stem.split('_')
    if len(parts) < 4:
        raise ValueError(f'Unexpected file name: {src_path}')
    return f'{parts[2]}_{parts[3]}'


def get_split(tile_id):
    if tile_id in TRAIN_SPLIT:
        return 'train'
    if tile_id in VAL_SPLIT:
        return 'val'
    raise KeyError(f'Unknown Potsdam tile id: {tile_id}')


def iter_starts(length, clip_size):
    if length <= clip_size:
        return [0]
    starts = list(range(0, length - clip_size + 1, clip_size))
    last = length - clip_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_boxes(h, w, clip_size):
    y_starts = iter_starts(h, clip_size)
    x_starts = iter_starts(w, clip_size)
    for start_y in y_starts:
        end_y = start_y + clip_size
        for start_x in x_starts:
            end_x = start_x + clip_size
            yield start_x, start_y, end_x, end_y


def rgb_label_to_index(image_rgb):
    h, w, _ = image_rgb.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    for idx, color in enumerate(COLOR_MAP):
        out[np.all(image_rgb == color.reshape(1, 1, 3), axis=2)] = idx
    return out


def save_image_patch(patch, save_path):
    Image.fromarray(patch.astype(np.uint8)).save(save_path, compress_level=1)


def save_label_patch(patch, save_path):
    Image.fromarray(patch.astype(np.uint8), mode='L').save(save_path, compress_level=1)


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    image = np.asarray(Image.open(image_path))
    clip_size = args.clip_size
    h, w = image.shape[:2]
    if to_label:
        image = rgb_label_to_index(image[:, :, :3])
    tile_id = parse_tile_id(image_path)
    idx_i, idx_j = tile_id.split('_')
    saved = 0
    for start_x, start_y, end_x, end_y in iter_boxes(h, w, clip_size):
        patch = image[start_y:end_y, start_x:end_x] if to_label else image[start_y:end_y, start_x:end_x, :3]
        save_name = f'{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.png'
        save_path = osp.join(clip_save_dir, save_name)
        if to_label:
            save_label_patch(patch, save_path)
        else:
            save_image_patch(patch, save_path)
        saved += 1
    return saved


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = osp.join('data', 'potsdam') if args.out_dir is None else args.out_dir

    print('[Info] Converting Potsdam noBoundary labels...')
    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')
    print('[Info] Archives will be extracted next to the zip files, not to /tmp.')
    print('[Info] Expected label ids after conversion: 0-5')
    print('[Info] For noBoundary labels, use reduce_zero_label=False in the dataset config.')

    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    zipp_list = find_zip_files(dataset_path)
    counts = {'train_img': 0, 'val_img': 0, 'train_ann': 0, 'val_ann': 0}

    for zipp in zipp_list:
        print(f'[Info] Processing archive: {osp.basename(zipp)}')
        extract_root = _extract_archive_to_sibling(zipp)
        src_path_list = find_tifs(extract_root)
        for i, src_path in enumerate(src_path_list, 1):
            tile_id = parse_tile_id(src_path)
            split = get_split(tile_id)
            is_label = 'label' in osp.basename(src_path).lower()
            dst_dir = osp.join(out_dir, 'ann_dir' if is_label else 'img_dir', split)
            num_patches = clip_big_image(src_path, dst_dir, args, to_label=is_label)
            key = f'{split}_ann' if is_label else f'{split}_img'
            counts[key] += num_patches
            print(f'  [{i}/{len(src_path_list)}] {osp.basename(src_path)} -> {split} ({num_patches} patches)')

    print('[Info] Patch summary:')
    print(f"  train images: {counts['train_img']}")
    print(f"  train labels: {counts['train_ann']}")
    print(f"  val images:   {counts['val_img']}")
    print(f"  val labels:   {counts['val_ann']}")
    print('[Info] With default args, the expected totals are 3456 train and 2016 val patches.')
    print('Done!')


if __name__ == '__main__':
    main()
