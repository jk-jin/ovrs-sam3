from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests._bootstrap import load_module


converter = load_module("tools.datasets_converters.dlrsd")
COUNTS = {"train": 2, "val": 1, "all": 3}


def encode_image(array, format, mode=None):
    stream = io.BytesIO()
    image = Image.fromarray(array)
    if mode == "P":
        image = image.convert("P")
        image.putpalette([value for i in range(256) for value in (i, 0, 255 - i)])
    image.save(stream, format=format)
    return stream.getvalue()


def make_archive(path, *, overlap=False, missing_mask=False, label=16,
                 mismatch=False, image_size=256, wrapper=""):
    image = encode_image(np.zeros((image_size, image_size, 3), dtype=np.uint8), "JPEG")
    # Includes class zero, class 16 and ignore: no value may be shifted.
    array = np.zeros((256, 256), dtype=np.uint8)
    array[0, 0], array[0, 1] = label, 255
    mask = encode_image(array, "PNG")
    with zipfile.ZipFile(path, "w") as source:
        for stem, split in [("airplane", "train"), ("water", "train"), ("tank", "val")]:
            for root in ["DLRSD", f"DLRSD_split/{split}"]:
                prefix = wrapper + root
                source.writestr(f"{prefix}/imgs/{stem}.jpg", image)
                if not (missing_mask and root.endswith("val")):
                    content = mask
                    if mismatch and root.endswith("val"):
                        content = encode_image(np.ones((256, 256), dtype=np.uint8), "PNG")
                    source.writestr(f"{prefix}/D2masks/{stem}.png", content)
                # Raw masks use another encoding and must not be consumed.
                source.writestr(f"{prefix}/masks/{stem}.png", b"not a D2mask")
        if overlap:
            source.writestr(f"{wrapper}DLRSD_split/val/imgs/airplane.jpg", image)
            source.writestr(f"{wrapper}DLRSD_split/val/D2masks/airplane.png", mask)
    return image, array


class TestDLRSDConverter(unittest.TestCase):
    def run_converter(self, archive, output, *options):
        with patch.object(converter, "EXPECTED_COUNTS", COUNTS), \
                contextlib.redirect_stdout(io.StringIO()):
            converter.main([str(archive), "-o", str(output), "--workers", "2", *options])

    def test_official_membership_and_zero_based_labels_survive_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, output = root / "DLRSD.zip", root / "prepared"
            image_bytes, labels = make_archive(archive, wrapper="download/")
            archive_hash = converter.sha256_file(archive)
            self.run_converter(archive, output)
            self.assertEqual((output / "splits/train.txt").read_text(), "airplane\nwater\n")
            self.assertEqual((output / "splits/val.txt").read_text(), "tank\n")
            metadata = json.loads((output / "processing_metadata.json").read_text())
            self.assertTrue(metadata["official_split"])
            self.assertEqual(metadata["counts"], COUNTS)
            self.assertEqual(metadata["source_archive_sha256"], archive_hash)
            self.assertEqual(converter.sha256_file(archive), archive_hash)
            for split, stems in {"train": ["airplane", "water"], "val": ["tank"],
                                 "all": ["airplane", "tank", "water"]}.items():
                for stem in stems:
                    self.assertEqual((output / f"img_dir/{split}/{stem}.jpg").read_bytes(), image_bytes)
                    with Image.open(output / f"ann_dir/{split}/{stem}.png") as mask:
                        self.assertEqual(mask.mode, "L")
                        np.testing.assert_array_equal(np.asarray(mask), labels)

    def test_invalid_archive_fails_without_replacing_existing_outputs(self):
        cases = [dict(overlap=True), dict(missing_mask=True), dict(mismatch=True),
                 dict(label=17), dict(image_size=128)]
        for settings in cases:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive, output = root / "DLRSD.zip", root / "prepared"
                make_archive(archive, **settings)
                output.mkdir()
                (output / "img_dir").mkdir()
                sentinel = output / "img_dir/previous.txt"
                sentinel.write_text("previous dataset")
                with self.assertRaises(ValueError):
                    self.run_converter(archive, output, "--overwrite")
                self.assertEqual(sentinel.read_text(), "previous dataset")
                self.assertFalse((output / "processing_metadata.json").exists())

    def test_overwrite_removes_stale_generated_files_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "DLRSD.zip"
            make_archive(archive)
            self.run_converter(archive, root)
            stale = root / "ann_dir/train/old_custom_split.png"
            stale.write_bytes(b"old mask")
            (root / "notes.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                self.run_converter(archive, root)
            self.run_converter(archive, root, "--overwrite")
            self.assertFalse(stale.exists())
            self.assertEqual((root / "notes.txt").read_text(), "keep")
            self.assertTrue(archive.exists())

    def test_incomplete_official_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "DLRSD.zip"
            make_archive(archive)
            with zipfile.ZipFile(archive) as source, self.assertRaisesRegex(ValueError, "complete official split"):
                converter.collect_samples(source)

    def test_original_images_labels_archive_is_not_accepted_as_official(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "DLRSD.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("DLRSD/Images/airplane/airplane00.tif", b"original")
                source.writestr("DLRSD/Labels/airplane/airplane00.png", b"original")
            with zipfile.ZipFile(archive) as source, self.assertRaisesRegex(ValueError, "DLRSD_split"):
                converter.collect_samples(source)

    def test_indexed_palette_is_not_converted_to_color_intensities(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            for kind in ("img_dir", "ann_dir"):
                for split in ("train", "all"):
                    (stage / kind / split).mkdir(parents=True)
            path = stage / "sample.zip"
            labels = np.tile(np.arange(17, dtype=np.uint8), (256, 16))[:, :256]
            with zipfile.ZipFile(path, "w") as source:
                source.writestr("image.jpg", encode_image(np.zeros((256, 256, 3), dtype=np.uint8), "JPEG"))
                source.writestr("mask.png", encode_image(labels, "PNG", "P"))
            with zipfile.ZipFile(path) as source:
                converter.convert_one(dict(stem="sample", split="train", image="image.jpg",
                                           mask="mask.png"), source, stage, 1)
            with Image.open(stage / "ann_dir/train/sample.png") as mask:
                np.testing.assert_array_equal(np.asarray(mask), labels)


if __name__ == "__main__":
    unittest.main()
