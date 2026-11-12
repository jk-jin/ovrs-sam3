#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare ORIGINAL DLRSD.zip for ReCA-SAM3 (no mmseg/mmcv dependency).

Run from the project root:
    python tools/datasets_converters/dlrsd.py data/datasets/DLRSD --overwrite

Input: DLRSD/Images/<scene>/*.tif and DLRSD/Labels/<scene>/*.png
Output: img_dir/{train,val,all}/*.jpg, ann_dir/{train,val,all}/*.png,
        splits/{train,val,all}.txt and processing_metadata.json.

CUSTOM protocol, NOT an RSKT-Seg split: within each of the 21 scene folders,
rank stable scene/stem IDs by SHA256("seed:scene/stem"); take the first
round-half-up(scene_size * val_ratio) for validation and the rest for training.
Default seed=42 and val_ratio=0.2 give 1680 train / 420 val / 2100 all.
There is no independent test set. "all" includes the training images.

Original indexed labels are ALWAYS 1..17 -> 0..16, with 0/255 -> ignore 255.
Palette PNGs are checked against the original class/color table. RGB/RGBA
annotations use exact RGB matching. Already-converted zero-based indexed
masks are NOT supported as input; use the original archive.

Dependencies: numpy, Pillow. Worker and PNG compression environment settings
match the other converters: RECA_CONVERTER_WORKERS, RECA_CONVERTER_PNG_COMPRESS.
Original data: https://sites.google.com/view/zhouwx/dataset
Class/color reference:
https://github.com/caoql98/OVRS/blob/main/cat_seg/data/datasets/register_DLRSD.py
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


CLASSES = (
    'airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court',
    'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea',
    'ship', 'tanks', 'trees', 'water',
)
PALETTE = (
    (166, 202, 240), (128, 128, 0), (0, 0, 128), (255, 0, 0),
    (0, 128, 0), (128, 0, 0), (255, 233, 233), (160, 160, 164),
    (0, 128, 128), (90, 87, 255), (255, 255, 0), (255, 192, 0),
    (0, 0, 255), (255, 0, 192), (128, 0, 128), (0, 255, 0),
    (0, 255, 255),
)
SCENES = (
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
    'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
    'storagetanks', 'tenniscourt',
)
MANAGED_OUTPUTS = ('img_dir', 'ann_dir', 'splits', 'processing_metadata.json')
IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
PROTOCOL = 'dlrsd_custom_scene_sha256_v1'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_path', help='Folder containing DLRSD.zip, or the ZIP itself')
    parser.add_argument('-o', '--out_dir', help='Default: the folder containing the archive')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace only generated outputs after the new conversion validates')
    parser.add_argument('--workers', type=int, default=int(
        os.environ.get('RECA_CONVERTER_WORKERS', min(os.cpu_count() or 1, 8))))
    args = parser.parse_args()
    if not math.isfinite(args.val_ratio) or not 0 < args.val_ratio < 1:
        parser.error('--val_ratio must be finite and strictly between 0 and 1')
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_archive(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == '.zip':
        return path
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset folder or ZIP not found: {path}')
    candidates = [p for folder in (path, path / 'raw') if folder.is_dir()
                  for p in folder.iterdir()
                  if p.is_file() and p.name.lower() == 'dlrsd.zip']
    if len(candidates) != 1:
        raise ValueError(f'Expected one DLRSD.zip in {path} or its raw/ folder; '
                         f'found {len(candidates)}. Pass the ZIP path explicitly.')
    return candidates[0]


def extract_archive(archive, destination):
    """Fresh extraction; never reuse an old extraction or scan old outputs."""
    with zipfile.ZipFile(archive) as source:
        seen = set()
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in item.filename
                    or (name.parts and ':' in name.parts[0])
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f'Unsafe archive member: {item.filename}')
            if item.is_dir():
                continue
            key = str(name).casefold()
            if key in seen:
                raise ValueError(f'Duplicate archive member: {item.filename}')
            seen.add(key)
            source.extract(item, destination)


def find_source_root(extracted):
    roots = []
    for folder in [extracted, *sorted(p for p in extracted.rglob('*') if p.is_dir())]:
        children = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
        if 'images' in children and 'labels' in children:
            roots.append((folder, children['images'], children['labels']))
    if len(roots) != 1:
        raise ValueError('Expected exactly one original DLRSD root with Images/ and Labels/; '
                         f'found {len(roots)}. This converter does not consume D2masks or old outputs.')
    return roots[0]


def collect_pairs(images_root, labels_root):
    def index_files(root, extensions):
        files = {}
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(root)
            if len(relative.parts) != 2:
                raise ValueError(f'Expected <scene>/<filename>: {path}')
            sample_id = f'{relative.parts[0].lower()}/{path.stem.lower()}'
            if sample_id in files:
                raise ValueError(f'Duplicate sample ID: {sample_id}')
            files[sample_id] = path
        return files

    images = index_files(images_root, IMAGE_EXTENSIONS)
    labels = index_files(labels_root, {'.png'})
    if images.keys() != labels.keys():
        raise ValueError(f'Image/mask mismatch. Missing masks: {sorted(images.keys()-labels.keys())[:5]}; '
                         f'missing images: {sorted(labels.keys()-images.keys())[:5]}')
    counts = Counter(sample_id.split('/')[0] for sample_id in images)
    if set(counts) != set(SCENES) or any(n != 100 for n in counts.values()):
        raise ValueError('Expected the complete ORIGINAL dataset: 21 scenes x 100 pairs. '
                         f'Found {len(images)} pairs with scene counts {dict(sorted(counts.items()))}. '
                         'Do not mix the original dataset with processed/expanded versions.')
    stems = [path.stem for path in images.values()]
    if len({stem.casefold() for stem in stems}) != len(stems):
        raise ValueError('Duplicate output filenames across scenes; refusing to overwrite samples.')
    return [{'id': key, 'stem': images[key].stem, 'image': str(images[key]),
             'mask': str(labels[key]), 'scene': key.split('/')[0]}
            for key in sorted(images)]


def assign_splits(records, seed, val_ratio):
    """The rank depends ONLY on seed and relative scene/stem, never paths."""
    for scene in SCENES:
        group = [row for row in records if row['scene'] == scene]
        ranked = sorted(group, key=lambda row: (
            hashlib.sha256(f"{seed}:{row['id']}".encode('utf-8')).hexdigest(), row['id']))
        n_val = int(math.floor(len(group) * val_ratio + 0.5))
        if not 0 < n_val < len(group):
            raise ValueError(f'val_ratio produces an empty train/val subset in {scene}')
        for i, row in enumerate(ranked):
            row['split'] = 'val' if i < n_val else 'train'
        print(f'[Split] {scene:<20} train={len(group)-n_val} val={n_val}')


def decode_mask(path):
    with Image.open(path) as image:
        arr = np.array(image)
        if image.mode == 'P' or arr.ndim == 2:
            if arr.ndim != 2 or arr.dtype.kind not in 'uib':
                raise ValueError(f'Unsupported indexed annotation: {path}')
            values = set(map(int, np.unique(arr)))
            if not values <= set(range(18)) | {255}:
                raise ValueError(f'Unknown original label IDs in {path}: {sorted(values)}')
            # The encoding is a DATASET contract, not guessed from this image's maximum.
            if image.mode == 'P':
                palette = image.getpalette()
                for value in sorted(values - {0, 255}):
                    color = tuple(palette[3*value:3*value+3]) if palette else ()
                    if color != PALETTE[value - 1]:
                        raise ValueError(f'{path}: palette index {value} has RGB {color}, '
                                         f'expected {PALETTE[value-1]}. Use ORIGINAL 1-based masks.')
            out = np.full(arr.shape, 255, dtype=np.uint8)
            valid = (arr >= 1) & (arr <= 17)
            out[valid] = arr[valid] - 1
            return out

        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'Unsupported annotation mode {image.mode}: {path}')
        rgb = np.asarray(image.convert('RGB'))
        out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        known = np.all(rgb == (0, 0, 0), axis=-1)  # Unlabeled black -> ignore.
        if image.mode == 'RGBA':
            known |= arr[..., 3] == 0
        for label_id, color in enumerate(PALETTE):
            matches = np.all(rgb == color, axis=-1)
            if image.mode == 'RGBA':
                matches &= arr[..., 3] != 0
            out[matches] = label_id
            known |= matches
        if not known.all():
            colors = np.unique(rgb[~known], axis=0)[:10].tolist()
            raise ValueError(f'Unknown RGB mask colors in {path}: {colors}')
        return out


def convert_one(task):
    row, stage, compress_level = task
    stage = Path(stage)
    stem = row['stem']
    label = decode_mask(row['mask'])
    image_dst = stage / 'img_dir' / 'all' / f'{stem}.jpg'
    mask_dst = stage / 'ann_dir' / 'all' / f'{stem}.png'
    with Image.open(row['image']) as image:
        if image.size != (256, 256) or label.shape != (256, 256):
            raise ValueError(f"{row['id']}: expected aligned 256x256 image/mask; "
                             f'got image={image.size}, mask={label.shape}')
        if image.format == 'JPEG' and image.mode == 'RGB':
            shutil.copy2(row['image'], image_dst)
        else:
            image.convert('RGB').save(image_dst, quality=95, subsampling=0)
    Image.fromarray(label).save(mask_dst, compress_level=compress_level)
    for src, kind in ((image_dst, 'img_dir'), (mask_dst, 'ann_dir')):
        shutil.copy2(src, stage / kind / row['split'] / src.name)
    return {
        'id': row['id'], 'stem': stem, 'scene': row['scene'], 'split': row['split'],
        'source_image_sha256': sha256_file(row['image']),
        'source_mask_sha256': sha256_file(row['mask']),
        'output_image_sha256': sha256_file(image_dst),
        'output_mask_sha256': sha256_file(mask_dst),
        'class_pixel_counts': np.bincount(label.ravel(), minlength=256).tolist(),
    }


def validate_outputs(stage, records):
    """Check the ACTUAL scanned directories, not just the manifest lists."""
    members = {}
    for split in ('train', 'val', 'all'):
        expected = {row['stem'] for row in records if split == 'all' or row['split'] == split}
        images = stage / 'img_dir' / split
        masks = stage / 'ann_dir' / split
        if {p.name for p in images.iterdir()} != {f'{s}.jpg' for s in expected}:
            raise ValueError(f'Unexpected image files in {split}')
        if {p.name for p in masks.iterdir()} != {f'{s}.png' for s in expected}:
            raise ValueError(f'Unexpected mask files in {split}')
        for stem in sorted(expected):
            with Image.open(images / f'{stem}.jpg') as image:
                if image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError(f'Invalid output image: {split}/{stem}')
                image.load()
            with Image.open(masks / f'{stem}.png') as mask:
                arr = np.array(mask)
                if mask.mode != 'L' or arr.shape != (256, 256):
                    raise ValueError(f'Invalid output mask: {split}/{stem}')
                if not np.all((arr < 17) | (arr == 255)):
                    raise ValueError(f'Invalid output IDs: {split}/{stem}')
        members[split] = expected
    if members['train'] & members['val'] or members['train'] | members['val'] != members['all']:
        raise ValueError('Train/val overlap or incomplete coverage')
    return {split: len(names) for split, names in members.items()}


def publish_outputs(stage, output, backup):
    """Replace only generated entries, and roll back if publication fails."""
    backup.mkdir()
    installed, moved = [], []
    try:
        for name in MANAGED_OUTPUTS:
            target = output / name
            if target.exists():
                target.rename(backup / name)
                moved.append(name)
            (stage / name).rename(target)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            (output / name).rename(stage / name)
        for name in reversed(moved):
            (backup / name).rename(output / name)
        raise


def main():
    args = parse_args()
    compress_level = int(os.environ.get('RECA_CONVERTER_PNG_COMPRESS', '1'))
    if not 0 <= compress_level <= 9:
        raise ValueError('RECA_CONVERTER_PNG_COMPRESS must be in 0..9')
    archive = find_archive(args.dataset_path)
    input_path = Path(args.dataset_path).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        input_path if input_path.is_dir() else archive.parent)
    output.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_OUTPUTS:
        target = output / name
        if target.is_symlink():
            raise ValueError(f'Refusing to replace an output symlink: {target}')
        if target.exists() and not args.overwrite:
            raise FileExistsError(f'{target} already exists. Use --overwrite to replace generated outputs.')
    # Do not publish over a source archive stored inside generated subdirectories.
    for name in MANAGED_OUTPUTS:
        if output / name in archive.parents:
            raise ValueError('The source ZIP must not be stored inside a managed output directory.')

    print(f'[Input] {archive}')
    print(f'[Output] {output}')
    print(f'[Protocol] CUSTOM split: {PROTOCOL}; seed={args.seed}, val_ratio={args.val_ratio}')
    print('[Info] No independent test split. Fresh extraction; existing outputs are not used as input.')
    archive_sha256 = sha256_file(archive)
    with tempfile.TemporaryDirectory(prefix='.dlrsd-build-', dir=output) as tmp:
        work = Path(tmp)
        extracted, stage = work / 'raw', work / 'prepared'
        print('[Info] Extracting the original archive...')
        extract_archive(archive, extracted)
        _, images_root, labels_root = find_source_root(extracted)
        records = collect_pairs(images_root, labels_root)
        assign_splits(records, args.seed, args.val_ratio)
        for kind in ('img_dir', 'ann_dir'):
            for split in ('train', 'val', 'all'):
                (stage / kind / split).mkdir(parents=True)
        (stage / 'splits').mkdir()
        tasks = [(row, str(stage), compress_level) for row in records]
        results = []
        if args.workers == 1:
            iterator = map(convert_one, tasks)
            for i, result in enumerate(iterator, 1):
                results.append(result)
                if i % 100 == 0 or i == len(tasks):
                    print(f'[Convert] {i}/{len(tasks)}')
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=mp.get_context('spawn')) as pool:
                for i, result in enumerate(pool.map(convert_one, tasks, chunksize=8), 1):
                    results.append(result)
                    if i % 100 == 0 or i == len(tasks):
                        print(f'[Convert] {i}/{len(tasks)}')
        results.sort(key=lambda row: row['id'])
        counts = validate_outputs(stage, records)
        split_hashes = {}
        for split in ('train', 'val', 'all'):
            stems = sorted(row['stem'] for row in records if split == 'all' or row['split'] == split)
            text = ''.join(stem + '\n' for stem in stems)
            path = stage / 'splits' / f'{split}.txt'
            path.write_text(text, encoding='utf-8')
            split_hashes[split] = sha256_file(path)
        histogram = np.sum([row.pop('class_pixel_counts') for row in results], axis=0)
        metadata = {
            'schema_version': 1, 'protocol_id': PROTOCOL, 'official_split': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_archive': archive.name, 'source_archive_sha256': archive_sha256,
            'seed': args.seed, 'val_ratio': args.val_ratio,
            'split_rule': 'per scene; ascending SHA256(UTF8(f"{seed}:{lowercase_scene}/{lowercase_stem}")); '
                          'tie-break by sample ID; first floor(N*val_ratio+0.5) are validation',
            'counts': counts, 'independent_test_set': False,
            'scene_counts': dict(sorted(Counter(row['scene'] for row in records).items())),
            'classes': list(CLASSES), 'palette': list(PALETTE),
            'indexed_source_encoding': '1..17; 0/255 are ignored',
            'output_encoding': '0..16; ignore=255; reduce_zero_label=False; background disabled',
            'image_size': [256, 256], 'conversion_geometry': 'none',
            'image_encoding': 'RGB JPEG; preserve input JPEG bytes, otherwise quality=95/subsampling=0',
            'split_file_sha256': split_hashes,
            'class_pixel_counts': {name: int(histogram[i]) for i, name in enumerate(CLASSES)},
            'ignored_pixel_count': int(histogram[255]),
            'source_url': 'https://sites.google.com/view/zhouwx/dataset',
            'samples': results,
        }
        (stage / 'processing_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        publish_outputs(stage, output, work / 'previous_outputs')
    print(f'[Check] Passed: {counts}; train/val disjoint, complete coverage, valid label IDs.')
    print('[Done] Archive preserved. Split lists and processing_metadata.json saved.')


if __name__ == '__main__':
    main()
