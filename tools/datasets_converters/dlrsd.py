#!/usr/bin/env python3
"""Prepare the RSKT-Seg/OVRSISBench DLRSD.zip for ReCA-SAM3.

Run from the project root:
    python tools/datasets_converters/dlrsd.py data/datasets/DLRSD --overwrite

The archive must contain sibling DLRSD/ and DLRSD_split/ directories:
    DLRSD/{imgs,D2masks}/                 7002 pairs
    DLRSD_split/train/{imgs,D2masks}/     5601 pairs
    DLRSD_split/val/{imgs,D2masks}/       1401 pairs

Output: img_dir/{train,val,all}/*.jpg, ann_dir/{train,val,all}/*.png,
        splits/{train,val,all}.txt and processing_metadata.json.

Use the supplied train/val membership without random splitting. D2masks
already contain IDs 0..16 (255 is ignored): do NOT subtract one. The separate
masks/ directories contain another encoding and are deliberately not used.
JPEG image bytes are preserved. No resizing, cropping or augmentation occurs.
The all/ output includes training images; there is no independent test split.

Dependencies: numpy, Pillow.
Source: https://huggingface.co/datasets/kkk2026/OVRSISBenchtest
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


CLASSES = (
    'airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court',
    'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea',
    'ship', 'tanks', 'trees', 'water',
)
EXPECTED_COUNTS = {'train': 5601, 'val': 1401, 'all': 7002}
MANAGED_OUTPUTS = ('img_dir', 'ann_dir', 'splits', 'processing_metadata.json')
PROTOCOL = 'dlrsd_ovrsisbench_official_v1'
SOURCE_URL = 'https://huggingface.co/datasets/kkk2026/OVRSISBenchtest'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_path', help='Folder containing DLRSD.zip, or the ZIP itself')
    parser.add_argument('-o', '--out_dir', help='Default: the folder containing the archive')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace generated outputs only after conversion validates')
    parser.add_argument('--workers', type=int, default=int(os.environ.get(
        'RECA_CONVERTER_WORKERS', min(os.cpu_count() or 1, 8))))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_archive(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == '.zip':
        return path
    candidates = [p for folder in (path, path / 'raw') if folder.is_dir()
                  for p in folder.iterdir()
                  if p.is_file() and p.name.lower() == 'dlrsd.zip']
    if len(candidates) != 1:
        raise ValueError(f'Expected one DLRSD.zip in {path} or its raw/ folder; '
                         'pass the ZIP path explicitly.')
    return candidates[0]


def collect_samples(source):
    """Validate the official layout and use its split membership verbatim."""
    members = {}
    for info in source.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path in members:
            raise ValueError(f'Duplicate archive member: {info.filename}')
        if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
            raise ValueError(f'Unsafe archive member: {info.filename}')
        members[path] = info

    # An optional enclosing directory is allowed, but the dataset layout is fixed.
    roots = {path.parents[3] for path in members
             if len(path.parts) >= 4 and path.parts[-4:-1]
             == ('DLRSD_split', 'train', 'imgs')}
    if len(roots) != 1:
        raise ValueError('Expected one official DLRSD_split/train/imgs directory. '
                         'Use the OVRSISBench DLRSD.zip containing the supplied split.')
    root = roots.pop()

    def pairs(directory):
        def index(folder, suffix):
            found = {p.stem: info for p, info in members.items()
                     if p.parent == directory / folder and p.suffix.lower() == suffix}
            if len({stem.casefold() for stem in found}) != len(found):
                raise ValueError(f'Duplicate sample filenames in {directory / folder}')
            return found

        images, masks = index('imgs', '.jpg'), index('D2masks', '.png')
        if not images or images.keys() != masks.keys():
            raise ValueError(f'Image/D2mask mismatch in {directory}; '
                             f'missing masks={sorted(images.keys()-masks.keys())[:5]}, '
                             f'missing images={sorted(masks.keys()-images.keys())[:5]}')
        return images, masks

    datasets = {split: pairs(root / 'DLRSD_split' / split)
                for split in ('train', 'val')}
    datasets['all'] = pairs(root / 'DLRSD')
    stems = {split: set(images) for split, (images, _) in datasets.items()}
    if stems['train'] & stems['val']:
        raise ValueError('Official train/val membership overlaps.')
    if stems['all'] != stems['train'] | stems['val']:
        raise ValueError('DLRSD all/ must equal the official train/val union.')
    counts = {split: len(ids) for split, ids in stems.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Expected complete official split {EXPECTED_COUNTS}, got {counts}.')

    records = []
    for split in ('train', 'val'):
        for stem in sorted(stems[split]):
            image, mask = (files[stem] for files in datasets[split])
            # The archive duplicates all/ samples in train/val: check those copies.
            for item, files in zip((image, mask), datasets['all']):
                original = files[stem]
                if (item.CRC, item.file_size) != (original.CRC, original.file_size):
                    raise ValueError(f'Split/all content mismatch: {item.filename}')
            records.append({'stem': stem, 'split': split,
                            'image': image.filename, 'mask': mask.filename})
    return sorted(records, key=lambda row: row['stem'])


def convert_one(row, source, stage, compress_level):
    image_bytes = source.read(row['image'])
    mask_bytes = source.read(row['mask'])
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (256, 256):
            raise ValueError(f"Expected RGB 256x256 JPEG: {row['image']}")
        image.load()
    with Image.open(io.BytesIO(mask_bytes)) as mask:
        label = np.array(mask)
    if (label.shape != (256, 256) or label.dtype.kind not in 'ui'
            or not np.all(((label >= 0) & (label < len(CLASSES))) | (label == 255))):
        raise ValueError(f"Expected 256x256 D2mask IDs 0..16/255: {row['mask']}")
    label = label.astype(np.uint8)
    image_dst = stage / 'img_dir' / 'all' / f"{row['stem']}.jpg"
    mask_dst = stage / 'ann_dir' / 'all' / f"{row['stem']}.png"
    image_dst.write_bytes(image_bytes)
    # Re-encode index values as grayscale; palette colors never become class IDs.
    Image.fromarray(label).save(mask_dst, compress_level=compress_level)
    for path, kind in ((image_dst, 'img_dir'), (mask_dst, 'ann_dir')):
        shutil.copyfile(path, stage / kind / row['split'] / path.name)
    return {**row, 'source_image_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'source_mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
            'output_mask_sha256': sha256_file(mask_dst),
            'class_pixel_counts': np.bincount(label.ravel(), minlength=256).tolist()}


def validate_outputs(stage, records):
    counts = {}
    for split in ('train', 'val', 'all'):
        stems = {row['stem'] for row in records if split == 'all' or row['split'] == split}
        for kind, suffix in (('img_dir', '.jpg'), ('ann_dir', '.png')):
            actual = {p.name for p in (stage / kind / split).iterdir()}
            if actual != {stem + suffix for stem in stems}:
                raise ValueError(f'Unexpected output files in {kind}/{split}')
        counts[split] = len(stems)
    return counts


def publish_outputs(stage, output, backup):
    """Replace only generated entries, rolling back a failed replacement."""
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


def main(argv=None):
    args = parse_args(argv)
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
        if target.is_symlink() or target in archive.parents:
            raise ValueError(f'Output must not be a symlink or contain the source ZIP: {target}')
        if target.exists() and not args.overwrite:
            raise FileExistsError(f'{target} exists; use --overwrite to replace generated outputs.')

    print(f'[Input] {archive}\n[Output] {output}\n[Protocol] {PROTOCOL}', flush=True)
    with zipfile.ZipFile(archive) as source:
        records = collect_samples(source)
        with tempfile.TemporaryDirectory(prefix='.dlrsd-build-', dir=output) as tmp:
            work = Path(tmp)
            stage = work / 'prepared'
            for kind in ('img_dir', 'ann_dir'):
                for split in EXPECTED_COUNTS:
                    (stage / kind / split).mkdir(parents=True)
            (stage / 'splits').mkdir()
            convert = partial(convert_one, source=source, stage=stage,
                              compress_level=compress_level)
            results = []
            # ZipFile serializes file seeks internally; decoding/writing overlaps.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, result in enumerate(pool.map(convert, records), 1):
                    results.append(result)
                    if i % 500 == 0 or i == len(records):
                        print(f'[Convert] {i}/{len(records)}', flush=True)
            counts = validate_outputs(stage, records)
            split_hashes = {}
            for split in EXPECTED_COUNTS:
                path = stage / 'splits' / f'{split}.txt'
                path.write_text(''.join(row['stem'] + '\n' for row in records
                                         if split == 'all' or row['split'] == split),
                                encoding='utf-8')
                split_hashes[split] = sha256_file(path)
            histogram = np.sum([row.pop('class_pixel_counts') for row in results], axis=0)
            metadata = {
                'schema_version': 2, 'protocol_id': PROTOCOL, 'official_split': True,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_url': SOURCE_URL, 'source_archive': archive.name,
                'source_archive_sha256': sha256_file(archive),
                'split_rule': 'preserve DLRSD_split/train and DLRSD_split/val membership',
                'counts': counts, 'independent_test_set': False,
                'all_includes_training': True, 'classes': list(CLASSES),
                'source_encoding': 'D2masks: 0..16, ignore=255; no label shift',
                'output_encoding': '0..16, ignore=255; reduce_zero_label=False',
                'image_size': [256, 256], 'conversion_geometry': 'none',
                'image_encoding': 'source RGB JPEG bytes preserved',
                'split_file_sha256': split_hashes,
                'class_pixel_counts': {name: int(histogram[i]) for i, name in enumerate(CLASSES)},
                'ignored_pixel_count': int(histogram[255]), 'samples': results,
            }
            (stage / 'processing_metadata.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            publish_outputs(stage, output, work / 'previous_outputs')
    print(f'[Done] {counts}; official membership preserved, archive unchanged.')
    print('[Info] all/ includes train and val; no independent test split is created.')


if __name__ == '__main__':
    main()
