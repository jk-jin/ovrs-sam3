# -*- coding: utf-8 -*-
"""
Source notice
-------------
There is no official UAVid converter in MMSegmentation's
tools/dataset_converters directory that I could find.

This script follows the public mmseg-style UAVid conversion logic used in
SegEarth-style repositories, but is rewritten to:
1) work without installing mmseg/mmcv/mmengine,
2) support the real extracted directory structure often seen in the official
   UAVid release package, e.g.
      uavid_train/seq1/Images
      uavid_train/seq1/Labels
      uavid_val/seq1/Images
      uavid_test/seq1/Images
3) crop images/labels into mmseg-style folders:
      out_dir/
        img_dir/{train,val,test}
        ann_dir/{train,val}

Please keep this notice when redistributing the file.
"""

import argparse
import glob
import os
import os.path as osp
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VALID_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
IMAGE_DIR_NAMES = ('Images', 'images', 'Img', 'img')
LABEL_DIR_NAMES = ('Labels', 'labels', 'GT', 'gt', 'Label', 'label', 'Annotations', 'annotations')

# Official UAVid 8-class color mapping.
# 0 background_clutter
# 1 building
# 2 road
# 3 tree
# 4 low_vegetation
# 5 moving_car
# 6 static_car
# 7 human
UAVID_COLOR_TO_INDEX = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (128, 64, 128): 2,
    (0, 128, 0): 3,
    (128, 128, 0): 4,
    (64, 0, 128): 5,
    (192, 0, 192): 6,
    (64, 64, 0): 7,
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert UAVid dataset to mmsegmentation-style layout')
    parser.add_argument('dataset_path', help='UAVid folder path or a directory containing the official UAVid zip')
    parser.add_argument('-o', '--out_dir', help='Output path')

    # Public SegEarth-style arguments.
    parser.add_argument(
        '--patch_width', default=1280, type=int,
        help='Width of the cropped image patch')
    parser.add_argument(
        '--patch_height', default=1080, type=int,
        help='Height of the cropped image patch')
    parser.add_argument(
        '--overlap_area', default=0, type=int,
        help='Overlap area')

    # Legacy aliases kept for compatibility with earlier custom versions.
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Optional legacy alias: use square crop size for width and height')
    parser.add_argument('--stride', type=int, default=None,
                        help='Optional legacy alias: square stride, overlap becomes crop_size - stride')
    parser.add_argument('--copy_instead_of_move', action='store_true',
                        help='Copy files instead of moving files from the source dataset when not cropping')
    return parser.parse_args()


def normalize_name(name):
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


def is_image_file(path):
    return osp.isfile(path) and osp.splitext(path)[1].lower() in VALID_IMAGE_SUFFIXES


def _pick_direct_subdir(parent, names):
    for name in names:
        cand = osp.join(parent, name)
        if osp.isdir(cand):
            return cand
    return None


def _pack_rgb(arr):
    flat = arr.reshape(-1, 3).astype(np.uint32)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def color_label_to_index(label):
    if label.ndim == 2:
        return label.astype(np.uint8)
    if label.ndim != 3 or label.shape[2] != 3:
        raise ValueError(f'Unsupported label shape: {label.shape}')

    h, w, _ = label.shape
    packed = _pack_rgb(label)
    out = np.full((packed.shape[0],), 255, dtype=np.uint8)
    color_keys = {
        ((np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)): idx
        for (r, g, b), idx in UAVID_COLOR_TO_INDEX.items()
    }
    for packed_rgb, idx in color_keys.items():
        out[packed == packed_rgb] = idx
    return out.reshape(h, w)


def pad_if_needed(arr, target_h, target_w, pad_val):
    h, w = arr.shape[:2]
    if h >= target_h and w >= target_w:
        return arr
    target_h = max(h, target_h)
    target_w = max(w, target_w)
    if arr.ndim == 3:
        out = np.full((target_h, target_w, arr.shape[2]), pad_val, dtype=arr.dtype)
        out[:h, :w, :] = arr
    else:
        out = np.full((target_h, target_w), pad_val, dtype=arr.dtype)
        out[:h, :w] = arr
    return out


def iter_crop_boxes(img_h, img_w, patch_h, patch_w, overlap):
    step_w = patch_w - overlap
    step_h = patch_h - overlap
    if step_w <= 0 or step_h <= 0:
        raise ValueError('patch_width - overlap_area and patch_height - overlap_area must both be positive')

    for x in range(0, img_w, step_w):
        for y in range(0, img_h, step_h):
            x_str = x
            x_end = x + patch_w
            if x_end > img_w:
                diff_x = x_end - img_w
                x_str -= diff_x
                x_end = img_w
            y_str = y
            y_end = y + patch_h
            if y_end > img_h:
                diff_y = y_end - img_h
                y_str -= diff_y
                y_end = img_h
            yield y_str, y_end, x_str, x_end


def save_png(arr, path):
    ensure_dir(osp.dirname(path))
    Image.fromarray(arr.astype(np.uint8)).save(path, compress_level=1)


def make_crop_name(seq_name, src_path, y1, y2, x1, x2):
    stem = osp.splitext(osp.basename(src_path))[0]
    return f'{seq_name}_{stem}_{y1}_{y2}_{x1}_{x2}.png'


def slide_crop_image(src_path, out_dir, mode, seq_name, patch_h, patch_w, overlap):
    img = np.asarray(Image.open(src_path).convert('RGB'))
    img = pad_if_needed(img, patch_h, patch_w, 0)
    img_h, img_w, _ = img.shape

    saved = 0
    for y_str, y_end, x_str, x_end in iter_crop_boxes(img_h, img_w, patch_h, patch_w, overlap):
        img_patch = img[y_str:y_end, x_str:x_end, :]
        image_name = make_crop_name(seq_name, src_path, y_str, y_end, x_str, x_end)
        save_png(img_patch, osp.join(out_dir, 'img_dir', mode, image_name))
        saved += 1
    return saved


def slide_crop_label(src_path, out_dir, mode, seq_name, patch_h, patch_w, overlap):
    label_rgb = np.asarray(Image.open(src_path).convert('RGB'))
    label = color_label_to_index(label_rgb)
    label = pad_if_needed(label, patch_h, patch_w, 255)
    img_h, img_w = label.shape

    saved = 0
    for y_str, y_end, x_str, x_end in iter_crop_boxes(img_h, img_w, patch_h, patch_w, overlap):
        lab_patch = label[y_str:y_end, x_str:x_end]
        image_name = make_crop_name(seq_name, src_path, y_str, y_end, x_str, x_end)
        save_png(lab_patch, osp.join(out_dir, 'ann_dir', mode, image_name))
        saved += 1
    return saved


def maybe_transfer(src, dst, copy_only):
    ensure_dir(osp.dirname(dst))
    if copy_only:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def list_split_parts(zip_path):
    zip_dir = osp.dirname(zip_path)
    base = osp.splitext(osp.basename(zip_path))[0]
    return sorted(glob.glob(osp.join(zip_dir, base + '.z[0-9][0-9]')))


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0, result


def extract_zip_archive(zip_path, tmp_dir):
    split_parts = list_split_parts(zip_path)

    try:
        if not split_parts:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)
            return True, 'python-zipfile'
    except zipfile.BadZipFile:
        pass

    sevenz_bin = shutil.which('7z') or shutil.which('7zz')
    if sevenz_bin:
        ok, _ = run_cmd([sevenz_bin, 'x', '-y', zip_path, f'-o{tmp_dir}'])
        if ok:
            return True, osp.basename(sevenz_bin)

    unzip_bin = shutil.which('unzip')
    if unzip_bin:
        ok, _ = run_cmd([unzip_bin, '-oq', zip_path, '-d', tmp_dir])
        if ok:
            return True, 'unzip'

    bsdtar_bin = shutil.which('bsdtar')
    if bsdtar_bin:
        ok, _ = run_cmd([bsdtar_bin, '-xf', zip_path, '-C', tmp_dir])
        if ok:
            return True, 'bsdtar'

    zip_bin = shutil.which('zip')
    if zip_bin and split_parts:
        repaired_zip = osp.join(tmp_dir, '__uavid_repaired.zip')
        ok, _ = run_cmd([zip_bin, '-FF', zip_path, '--out', repaired_zip])
        if ok and osp.isfile(repaired_zip):
            if sevenz_bin:
                ok2, _ = run_cmd([sevenz_bin, 'x', '-y', repaired_zip, f'-o{tmp_dir}'])
                if ok2:
                    return True, f'{osp.basename(zip_bin)} -FF + {osp.basename(sevenz_bin)}'
            if unzip_bin:
                ok2, _ = run_cmd([unzip_bin, '-oq', repaired_zip, '-d', tmp_dir])
                if ok2:
                    return True, f'{osp.basename(zip_bin)} -FF + unzip'
            if bsdtar_bin:
                ok2, _ = run_cmd([bsdtar_bin, '-xf', repaired_zip, '-C', tmp_dir])
                if ok2:
                    return True, f'{osp.basename(zip_bin)} -FF + bsdtar'

    return False, None


def prepare_dataset_root(dataset_path):
    cleanup_dir = None

    def _extract(zip_path):
        tmp_dir = tempfile.mkdtemp(prefix='uavid_extract_')
        print(f'[Info] Found UAVid archive: {zip_path}', flush=True)
        split_parts = list_split_parts(zip_path)
        if split_parts:
            shown = ', '.join(osp.basename(p) for p in split_parts[:3])
            if len(split_parts) > 3:
                shown += ', ...'
            print(f'[Info] Detected split zip part(s): {shown}', flush=True)
        print('[Info] Extracting archive...', flush=True)
        extracted, extractor_name = extract_zip_archive(zip_path, tmp_dir)
        if not extracted:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                'Failed to extract UAVid archive. Python zipfile could not read it, '
                'and no compatible system extractor succeeded.')
        print(f'[Info] Extracted UAVid archive with: {extractor_name}', flush=True)
        return tmp_dir, tmp_dir

    if osp.isfile(dataset_path) and dataset_path.lower().endswith('.zip'):
        return _extract(dataset_path)

    if osp.isdir(dataset_path):
        preferred_names = [
            'uavid_v1.5_official_release_image_split.zip',
            'uavid_v1.5_official_release.zip',
        ]
        zip_candidates = []
        for name in preferred_names:
            cand = osp.join(dataset_path, name)
            if osp.isfile(cand):
                zip_candidates.append(cand)
        if not zip_candidates:
            zip_candidates.extend(sorted(glob.glob(osp.join(dataset_path, '*.zip'))))
        if zip_candidates:
            zip_candidates = sorted(
                zip_candidates,
                key=lambda p: (
                    'uavid' not in osp.basename(p).lower(),
                    'image_split' not in osp.basename(p).lower(),
                    len(osp.basename(p)),
                    osp.basename(p).lower(),
                ))
            return _extract(zip_candidates[0])

    return dataset_path, cleanup_dir


def find_split_roots(dataset_root):
    mapping = {'train': [], 'val': [], 'test': []}
    alias_map = {
        'train': {'uavidtrain', 'train', 'training', 'uavid_train'},
        'val': {'uavidval', 'uavidvalid', 'uavidvalidation', 'val', 'valid', 'validation', 'uavid_val', 'uavid_valid'},
        'test': {'uavidtest', 'test', 'testing', 'testgt', 'uavid_test'},
    }
    alias_norm = {k: {normalize_name(x) for x in v} for k, v in alias_map.items()}

    for cur, dirs, _files in os.walk(dataset_root):
        for d in dirs:
            dn = normalize_name(d)
            for split, aliases in alias_norm.items():
                if dn in aliases:
                    mapping[split].append(osp.join(cur, d))

    for split in mapping:
        mapping[split] = sorted(set(mapping[split]), key=lambda p: (p.count(os.sep), p))
    return mapping


def find_sequence_units(split_root):
    units = []
    seen = set()
    for cur, _dirs, _files in os.walk(split_root):
        image_dir = _pick_direct_subdir(cur, IMAGE_DIR_NAMES)
        label_dir = _pick_direct_subdir(cur, LABEL_DIR_NAMES)
        if image_dir is None and label_dir is None:
            continue
        seq_name = osp.basename(cur)
        real = osp.realpath(cur)
        if real in seen:
            continue
        seen.add(real)
        units.append({
            'seq_name': seq_name,
            'image_dir': image_dir,
            'label_dir': label_dir,
            'unit_dir': cur,
        })
    units.sort(key=lambda x: x['unit_dir'])
    return units


def collect_split_items(dataset_root, split):
    split_roots = find_split_roots(dataset_root)[split]
    image_items = []
    label_items = []

    for split_root in split_roots:
        for unit in find_sequence_units(split_root):
            seq_name = unit['seq_name']
            image_dir = unit['image_dir']
            label_dir = unit['label_dir']

            if image_dir and osp.isdir(image_dir):
                for name in sorted(os.listdir(image_dir)):
                    path = osp.join(image_dir, name)
                    if is_image_file(path):
                        image_items.append((path, seq_name))

            if split != 'test' and label_dir and osp.isdir(label_dir):
                for name in sorted(os.listdir(label_dir)):
                    path = osp.join(label_dir, name)
                    if is_image_file(path):
                        label_items.append((path, seq_name))

    return image_items, label_items


def process_images(image_items, out_dir, split, patch_h, patch_w, overlap):
    count = 0
    for src_path, seq_name in image_items:
        count += slide_crop_image(src_path, out_dir, split, seq_name, patch_h, patch_w, overlap)
    return count


def process_labels(label_items, out_dir, split, patch_h, patch_w, overlap):
    count = 0
    for src_path, seq_name in label_items:
        count += slide_crop_label(src_path, out_dir, split, seq_name, patch_h, patch_w, overlap)
    return count


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir or osp.join('data', 'uavid')

    patch_w = args.patch_width
    patch_h = args.patch_height
    overlap = args.overlap_area

    if args.crop_size is not None:
        patch_w = args.crop_size
        patch_h = args.crop_size
    if args.stride is not None:
        if args.crop_size is None:
            raise ValueError('When using legacy --stride, please also set --crop_size')
        overlap = args.crop_size - args.stride
    if overlap < 0:
        raise ValueError('overlap_area must be >= 0, and crop_size - stride must be >= 0')

    print(f'[Input]  {dataset_path}', flush=True)
    print(f'[Output] {out_dir}', flush=True)
    print('[Note] This UAVid converter is custom-written in mmseg style.', flush=True)
    print(f'[Info] Crop setting: patch_width={patch_w}, patch_height={patch_h}, overlap_area={overlap}', flush=True)

    dataset_root, cleanup_dir = prepare_dataset_root(dataset_path)
    split_roots = find_split_roots(dataset_root)
    print('[Info] Detected split roots:', flush=True)
    for split in ['train', 'val', 'test']:
        roots = split_roots[split]
        if roots:
            shown = ', '.join(roots[:3])
            if len(roots) > 3:
                shown += ', ...'
            print(f'  - {split}: {shown}', flush=True)
        else:
            print(f'  - {split}: <not found>', flush=True)

    ensure_dir(osp.join(out_dir, 'img_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'test'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'val'))

    try:
        for split in ['train', 'val', 'test']:
            image_items, label_items = collect_split_items(dataset_root, split)
            if not image_items:
                if split == 'val':
                    print('[Info] No validation split found. Skipping val.', flush=True)
                elif split == 'test':
                    print('Warning: no test images found for split=test', flush=True)
                else:
                    print('Warning: no files found for split=train', flush=True)
                continue

            print(f'[Info] {split}: found {len(image_items)} image(s)', flush=True)
            num_images = process_images(image_items, out_dir, split, patch_h, patch_w, overlap)
            print(f'{split}: processed {num_images} cropped image patch(es)', flush=True)

            if split != 'test':
                if label_items:
                    print(f'[Info] {split}: found {len(label_items)} label(s)', flush=True)
                    num_labels = process_labels(label_items, out_dir, split, patch_h, patch_w, overlap)
                    print(f'{split}: processed {num_labels} cropped label patch(es)', flush=True)
                else:
                    print(f'Warning: no labels found for split={split}', flush=True)
    finally:
        if cleanup_dir is not None and osp.isdir(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    print('Done!', flush=True)


if __name__ == '__main__':
    main()
