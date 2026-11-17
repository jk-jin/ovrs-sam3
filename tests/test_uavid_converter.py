from __future__ import annotations

import contextlib
import io
import json
import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from tests._bootstrap import REPO_ROOT, load_module


converter = load_module('tools.datasets_converters.uavid')
prompts = load_module('data.class_prompts')
COLORS = [(0, 0, 0), (128, 0, 0), (128, 64, 128), (0, 128, 0),
          (128, 128, 0), (64, 0, 128), (192, 0, 192), (64, 64, 0)]


def png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


class TestUAVidConverter(unittest.TestCase):
    def run_converter(self, source, output, *options):
        with contextlib.redirect_stdout(io.StringIO()):
            converter.main([str(source), '-o', str(output),
                            '--patch_width', '10', '--patch_height', '2', *options])

    def test_rgb_colors_merge_both_cars_and_shift_human(self):
        mask = np.array([COLORS + [(1, 2, 3)]], dtype=np.uint8)
        actual = converter.color_label_to_index(mask)
        np.testing.assert_array_equal(actual, [[0, 1, 2, 3, 4, 5, 5, 6, 255]])
        self.assertEqual(actual.dtype, np.uint8)

    def test_original_index_mask_keeps_ignore(self):
        mask = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 255, 99]], dtype=np.uint8)
        np.testing.assert_array_equal(converter.color_label_to_index(mask),
                                      [[0, 1, 2, 3, 4, 5, 5, 6, 255, 255]])

    def test_grayscale_and_palette_masks_are_loaded_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray = root / 'gray.png'
            Image.fromarray(np.array([[5, 6, 7, 255]], dtype=np.uint8)).save(gray)
            palette = Image.fromarray(np.array([[23, 199, 255, 1]], dtype=np.uint8)).convert('P')
            entries = [0] * 768
            for index, color in [(23, COLORS[5]), (199, COLORS[6]), (255, COLORS[7]), (1, (1, 2, 3))]:
                entries[index * 3:index * 3 + 3] = color
            palette.putpalette(entries)
            palette.save(root / 'palette.png')
            for name in ('gray', 'palette'):
                converter.slide_crop_label(str(root / f'{name}.png'), str(root / 'out'),
                                           'val', 'seq', 2, 5, 0)
                output = next((root / 'out/ann_dir/val').glob(f'seq_{name}_*.png'))
                with Image.open(output) as image:
                    self.assertEqual(image.mode, 'L')
                    expected = np.full((2, 5), 255, dtype=np.uint8)
                    expected[0, :4] = [5, 5, 6, 255]
                    np.testing.assert_array_equal(np.asarray(image), expected)

    def test_zip_rebuild_clears_old_outputs_and_preserves_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'uavid_v1.5_official_release_image_split.zip'
            rgb = np.array([COLORS + [(1, 2, 3)]], dtype=np.uint8)
            with zipfile.ZipFile(archive, 'w') as source:
                for split in ('train', 'val', 'test'):
                    source.writestr(f'uavid_{split}/seq1/Images/000.png',
                                    png(Image.fromarray(np.zeros_like(rgb))))
                    if split != 'test':
                        source.writestr(f'uavid_{split}/seq1/Labels/000.png', png(Image.fromarray(rgb)))
            archive_bytes = archive.read_bytes()
            self.run_converter(root, root)
            for split in ('train', 'val'):
                images = list((root / f'img_dir/{split}').glob('*.png'))
                masks = list((root / f'ann_dir/{split}').glob('*.png'))
                self.assertEqual([p.name for p in images], [p.name for p in masks])
                with Image.open(masks[0]) as image:
                    expected = np.full((2, 10), 255, dtype=np.uint8)
                    expected[0, :9] = [0, 1, 2, 3, 4, 5, 5, 6, 255]
                    np.testing.assert_array_equal(np.asarray(image), expected)
            self.assertEqual(len(list((root / 'img_dir/test').glob('*.png'))), 1)
            self.assertFalse((root / 'ann_dir/test').exists())
            stale = root / 'ann_dir/val/old_eight_class.png'
            stale.write_bytes(b'old')
            (root / 'notes.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                self.run_converter(archive, root)
            self.assertTrue(stale.exists())
            self.run_converter(archive, root, '--overwrite')
            self.assertFalse(stale.exists())
            self.assertEqual(archive.read_bytes(), archive_bytes)
            self.assertEqual((root / 'notes.txt').read_text(), 'keep')
            metadata = json.loads((root / 'processing_metadata.json').read_text())
            self.assertEqual(metadata['label_space'], 'uavid-7class-merged-car-v2')
            self.assertEqual(metadata['classes'], list(converter.UAVID_CLASSES))
            self.assertEqual(metadata['num_classes'], 7)
            self.assertEqual(metadata['raw_id_to_class_id']['6'], 5)
            self.assertEqual(metadata['raw_id_to_class_id']['7'], 6)
            self.assertEqual(metadata['counts']['val'], dict(image_patches=1, label_patches=1))

    def test_missing_raw_input_does_not_clear_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / 'raw', root / 'out'
            raw.mkdir()
            (output / 'img_dir').mkdir(parents=True)
            sentinel = output / 'img_dir/old.png'
            sentinel.write_bytes(b'keep')
            with self.assertRaises(FileNotFoundError):
                self.run_converter(raw, output, '--overwrite')
            self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_raw_input_inside_generated_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'img_dir/train/seq/Images'
            image_dir.mkdir(parents=True)
            image = image_dir / '000.png'
            Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(image)
            before = image.read_bytes()
            with self.assertRaisesRegex(ValueError, 'outside generated'):
                self.run_converter(root, root, '--overwrite')
            self.assertEqual(image.read_bytes(), before)

    def test_config_prompts_match_merged_labels(self):
        cfg = runpy.run_path(str(REPO_ROOT / 'configs/datasets/eval/uavid.py'))
        dataset = cfg['val_dataloader']['dataset']
        self.assertEqual(dataset['classes'], list(converter.UAVID_CLASSES))
        self.assertTrue(dataset['background_cfg']['exclude_from_forward'])
        names, mapping = prompts.expand_class_prompts(dataset['classes'][1:])
        self.assertEqual(names, ['building', 'road', 'tree', 'low vegetation', 'car', 'human'])
        self.assertEqual(mapping, list(range(6)))
        entry = runpy.run_path(str(REPO_ROOT / 'configs/test/uavid.py'))
        self.assertEqual(entry['evaluation']['protocol_id'], 'uavid_7class_merged_car_val_resize1008_v2')


if __name__ == '__main__':
    unittest.main()
