# -*- coding: utf-8 -*-
"""
Source notice
-------------
This file is adapted for the user's OVRS-SAM3 project from the official
MMSegmentation dataset converter style.

Original project: OpenMMLab / MMSegmentation
License of original project: Apache License 2.0

This rewritten version removes the runtime dependency on mmseg/mmcv/mmengine
and only uses Python standard library so it can be used directly inside the
current project.
"""

import argparse
import os
import os.path as osp
import shutil
import zipfile
from pathlib import Path

ORIGINAL_SOURCE = (
    'https://github.com/open-mmlab/mmsegmentation/blob/main/'
    'tools/dataset_converters/loveda.py'
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LoveDA dataset to mmsegmentation-style layout')
    parser.add_argument('dataset_path', help='LoveDA folder path')
    parser.add_argument('--tmp_dir', help='Kept only for CLI compatibility; unused in this version')
    parser.add_argument('-o', '--out_dir', help='Output path')
    return parser.parse_args()


def _find_archive(dataset_path, name):
    path = osp.join(dataset_path, name)
    if osp.isfile(path):
        return path
    lower_map = {f.lower(): f for f in os.listdir(dataset_path)}
    mapped = lower_map.get(name.lower())
    if mapped is not None:
        return osp.join(dataset_path, mapped)
    raise FileNotFoundError(f'Missing required archive: {name}')


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
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)
    return extract_root


def _iter_files(root_dir):
    for cur, _, files in os.walk(root_dir):
        for name in files:
            yield osp.join(cur, name)


def _copy_all(src_dir, dst_dir):
    ensure_dir(dst_dir)
    count = 0
    for src in _iter_files(src_dir):
        dst = osp.join(dst_dir, osp.basename(src))
        shutil.copy2(src, dst)
        count += 1
    return count


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir or osp.join('data', 'loveDA')

    print(f'[Source] Adapted from: {ORIGINAL_SOURCE}')
    print(f'[Input]  {dataset_path}')
    print(f'[Output] {out_dir}')

    ensure_dir(osp.join(out_dir, 'img_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'val'))
    ensure_dir(osp.join(out_dir, 'img_dir', 'test'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'train'))
    ensure_dir(osp.join(out_dir, 'ann_dir', 'val'))

    required = ['Train.zip', 'Val.zip', 'Test.zip']
    archives = {name: _find_archive(dataset_path, name) for name in required}

    for dataset in ['Train', 'Val', 'Test']:
        archive_path = archives[f'{dataset}.zip']
        extract_root = _extract_archive_to_sibling(archive_path)

        candidates = [extract_root, osp.join(extract_root, dataset)]
        source_root = None
        for cand in candidates:
            if osp.isdir(cand):
                rural = osp.join(cand, 'Rural', 'images_png')
                urban = osp.join(cand, 'Urban', 'images_png')
                if osp.isdir(rural) or osp.isdir(urban):
                    source_root = cand
                    break
        if source_root is None:
            raise FileNotFoundError(
                f'Cannot locate extracted LoveDA root for {dataset} under {extract_root}')

        split = dataset.lower()
        for location in ['Rural', 'Urban']:
            image_src = osp.join(source_root, location, 'images_png')
            mask_src = osp.join(source_root, location, 'masks_png')
            if not osp.isdir(image_src):
                raise FileNotFoundError(f'Cannot find: {image_src}')
            copied_images = _copy_all(image_src, osp.join(out_dir, 'img_dir', split))
            print(f'  {dataset}/{location}/images_png -> {copied_images} files')
            if dataset != 'Test':
                if not osp.isdir(mask_src):
                    raise FileNotFoundError(f'Cannot find: {mask_src}')
                copied_masks = _copy_all(mask_src, osp.join(out_dir, 'ann_dir', split))
                print(f'  {dataset}/{location}/masks_png  -> {copied_masks} files')

    print('Done!')


if __name__ == '__main__':
    main()
