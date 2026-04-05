# -*- coding: utf-8 -*-
"""
UAVid converter for OVRS-SAM3.

There is no official UAVid converter in MMSegmentation's public
tools/dataset_converters directory.

This script follows the public mmseg-style UAVid conversion logic used in
SegEarth-style repositories, but is rewritten to:
1) work without installing mmseg/mmcv/mmengine,
2) support extracted directory structures such as:
      uavid_train/seq1/Images
      uavid_train/seq1/Labels
      uavid_val/seq1/Images
      uavid_test/seq1/Images
3) crop images/labels into mmseg-style folders
"""

import argparse
import glob
import os
import os.path as osp
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
VALID_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
IMAGE_DIR_NAMES = ('Images', 'images', 'Img', 'img')
LABEL_DIR_NAMES = ('Labels', 'labels', 'GT', 'gt', 'Label', 'label', 'Annotations', 'annotations')

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
    parser.add_argument('--patch_width', default=1280, type=int,
                        help='Width of the cropped image patch')
    parser.add_argument('--patch_height', default=1080, type=int,
                        help='Height of the cropped image patch')
    parser.add_argument('--overlap_area', default=0, type=int,
                        help='Overlap area')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Optional legacy alias: use square crop size for width and height')
    parser.add_argument('--stride', type=int, default=None,
                        help='Optional legacy alias: square stride, overlap becomes crop_size - stride')
    parser.add_argument('--copy_instead_of_move', action='store_true',
                        help='Copy files instead of moving files when not cropping')
    return parser.parse_args()


def normalize_name(name):
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


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
                x_str -= (x_end - img_w)
                x_end = img_w
            y_str = y
            y_end = y + patch_h
            if y_end > img_h:
                y_str -= (y_end - img_h)
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
        save_png(img_patch, osp.join(out_dir, 'img_dir', mode, make_crop_name(seq_name, src_path, y_str, y_end, x_str, x_end)))
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
        save_png(lab_patch, osp.join(out_dir, 'ann_dir', mode, make_crop_name(seq_name, src_path, y_str, y_end, x_str, x_end)))
        saved += 1
    return saved


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0, result


def list_split_parts(zip_path):
    zip_dir = osp.dirname(zip_path)
    base = osp.splitext(osp.basename(zip_path))[0]
    return sorted(glob.glob(osp.join(zip_dir, base + '.z[0-9][0-9]')))


def _extract_dir_for_archive(zip_path):
    parent = osp.dirname(zip_path)
    stem = osp.splitext(osp.basename(zip_path))[0]
    return osp.join(parent, f'{stem}_extracted')


def extract_zip_archive(zip_path, extract_root):
    split_parts = list_split_parts(zip_path)
    try:
        if not split_parts:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_root)
            return True, 'python-zipfile'
    except zipfile.BadZipFile:
        pass
    sevenz_bin = shutil.which('7z') or shutil.which('7zz')
    if sevenz_bin:
        ok, _ = run_cmd([sevenz_bin, 'x', '-y', zip_path, f'-o{extract_root}'])
        if ok:
            return True, osp.basename(sevenz_bin)
    unzip_bin = shutil.which('unzip')
    if unzip_bin:
        ok, _ = run_cmd([unzip_bin, '-oq', zip_path, '-d', extract_root])
        if ok:
            return True, 'unzip'
    bsdtar_bin = shutil.which('bsdtar')
    if bsdtar_bin:
        ok, _ = run_cmd([bsdtar_bin, '-xf', zip_path, '-C', extract_root])
        if ok:
            return True, 'bsdtar'
    zip_bin = shutil.which('zip')
    if zip_bin and split_parts:
        repaired_zip = osp.join(extract_root, '__uavid_repaired.zip')
        ok, _ = run_cmd([zip_bin, '-FF', zip_path, '--out', repaired_zip])
        if ok and osp.isfile(repaired_zip):
            if sevenz_bin:
                ok2, _ = run_cmd([sevenz_bin, 'x', '-y', repaired_zip, f'-o{extract_root}'])
                if ok2:
                    return True, f'{osp.basename(zip_bin)} -FF + {osp.basename(sevenz_bin)}'
            if unzip_bin:
                ok2, _ = run_cmd([unzip_bin, '-oq', repaired_zip, '-d', extract_root])
                if ok2:
                    return True, f'{osp.basename(zip_bin)} -FF + unzip'
    return False, None


def prepare_dataset_root(dataset_path):
    if osp.isfile(dataset_path) and dataset_path.lower().endswith('.zip'):
        zip_path = dataset_path
    elif osp.isdir(dataset_path):
        preferred_names = [
            'uavid_v1.5_official_release_image_split.zip',
            'uavid_v1.5_official_release.zip',
        ]
        zip_candidates = [osp.join(dataset_path, name) for name in preferred_names if osp.isfile(osp.join(dataset_path, name))]
        if not zip_candidates:
            zip_candidates = sorted(glob.glob(osp.join(dataset_path, '*.zip')))
        if zip_candidates:
            zip_candidates = sorted(
                zip_candidates,
                key=lambda p: ('uavid' not in osp.basename(p).lower(), 'image_split' not in osp.basename(p).lower(), len(osp.basename(p)), osp.basename(p).lower())
            )
            zip_path = zip_candidates[0]
        else:
            return dataset_path
    else:
        raise FileNotFoundError(dataset_path)

    extract_root = _extract_dir_for_archive(zip_path)
    if osp.isdir(extract_root) and any(True for _ in os.scandir(extract_root)):
        print(f'[Info] Reuse extracted directory: {extract_root}')
        return extract_root
    if osp.isdir(extract_root):
        shutil.rmtree(extract_root)
    ensure_dir(extract_root)
    print(f'[Info] Found UAVid archive: {zip_path}')
    split_parts = list_split_parts(zip_path)
    if split_parts:
        print(f'[Info] Detected split zip part(s): {", ".join(osp.basename(p) for p in split_parts)}')
    print(f'[Info] Extracting archive -> {extract_root}')
    extracted, extractor_name = extract_zip_archive(zip_path, extract_root)
    if not extracted:
        raise RuntimeError('Failed to extract UAVid archive with available extractors.')
    print(f'[Info] Extracted UAVid archive with: {extractor_name}')
    return extract_root


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
        key = (seq_name, image_dir, label_dir)
        if key in seen:
            continue
        seen.add(key)
        units.append((seq_name, image_dir, label_dir))
    return sorted(units)


def list_image_files(dir_path):
    if dir_path is None or not osp.isdir(dir_path):
        return []
    files = []
    for name in sorted(os.listdir(dir_path)):
        path = osp.join(dir_path, name)
        if osp.isfile(path) and osp.splitext(name)[1].lower() in VALID_IMAGE_SUFFIXES:
            files.append(path)
    return files


def maybe_transfer(src, dst, copy_only):
    ensure_dir(osp.dirname(dst))
    if copy_only:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def main():
    args = parse_args()
    patch_w = args.patch_width
    patch_h = args.patch_height
    overlap = args.overlap_area
    if args.crop_size is not None:
        patch_w = int(args.crop_size)
        patch_h = int(args.crop_size)
    if args.stride is not None:
        stride = int(args.stride)
        if stride <= 0:
            raise ValueError('--stride must be positive')
        if patch_w != patch_h:
            raise ValueError('--stride legacy mode expects square patches; pass --crop_size or equal patch sizes')
        overlap = patch_w - stride
        if overlap < 0:
            overlap = 0

    dataset_path = args.dataset_path
    out_dir = args.out_dir or osp.join('data', 'uavid')

    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')

    ensure_dir(osp.join(out_dir, 'img_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'test'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'val'))

    dataset_root = prepare_dataset_root(dataset_path)
    split_roots = find_split_roots(dataset_root)

    for split in ['train', 'val', 'test']:
        roots = split_roots.get(split, [])
        if not roots:
            if split == 'test':
                print('Warning: no test images found for split=test')
            else:
                print(f'Warning: no files found for split={split}')
            continue
        saved_images = 0
        saved_labels = 0
        for split_root in roots:
            units = find_sequence_units(split_root)
            for seq_name, image_dir, label_dir in units:
                img_files = list_image_files(image_dir)
                lab_files = list_image_files(label_dir)
                if not img_files and not lab_files:
                    continue
                print(f'[Info] split={split} seq={seq_name} images={len(img_files)} labels={len(lab_files)}')
                for img_path in img_files:
                    saved_images += slide_crop_image(img_path, out_dir, split, seq_name, patch_h, patch_w, overlap)
                if split != 'test':
                    for lab_path in lab_files:
                        saved_labels += slide_crop_label(lab_path, out_dir, split, seq_name, patch_h, patch_w, overlap)
        print(f'[Info] split={split} saved image patches: {saved_images}')
        if split != 'test':
            print(f'[Info] split={split} saved label patches: {saved_labels}')
    print('Done!')


if __name__ == '__main__':
    main()
