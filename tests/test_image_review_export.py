"""Tests for original-image and label-mask TIFF pair export."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from moma_lineage_analysis.image_review_export import export_review_pair


def _image():
    return (np.arange(6 * 9, dtype=np.uint16).reshape(6, 9) * 100) + 7


def _mask():
    mask = np.zeros((6, 9), dtype=np.uint8)
    mask[1:4, 2:5] = 2
    mask[4:6, 6:9] = 7
    return mask


def _export(tmp_path: Path, tag="a1b2c3d4"):
    with patch("moma_lineage_analysis.image_review_export._random_tag", return_value=tag):
        return export_review_pair(
            _image(), _mask(), output_dir=tmp_path,
            fov_name="FOV 000", time_point="folder/t00001.tiff",
            optical_channel="BF 1", side_channel=3,
        )


def test_export_preserves_original_pixels_dtype_and_shared_tag(tmp_path: Path):
    image_path, mask_path = _export(tmp_path)
    assert image_path.name == "FOV_000_t00001_BF_1_side3_a1b2c3d4.tif"
    assert mask_path.name == "FOV_000_t00001_BF_1_side3_a1b2c3d4_masks.tif"
    assert len(image_path.stem.rsplit("_", 1)[1]) == 8
    assert mask_path.stem == image_path.stem + "_masks"
    saved_image = tifffile.imread(image_path)
    assert saved_image.dtype == np.uint16
    np.testing.assert_array_equal(saved_image, _image())
    saved_mask = tifffile.imread(mask_path)
    assert saved_mask.dtype == np.uint8
    np.testing.assert_array_equal(saved_mask, _mask())


def test_repeat_export_uses_new_tags_and_never_overwrites(tmp_path: Path):
    first_image, first_mask = _export(tmp_path, "deadbeef")
    initial = first_image.read_bytes()
    second_image, second_mask = _export(tmp_path, "cafebabe")
    assert first_image.name != second_image.name
    assert first_mask.name != second_mask.name
    assert first_image.read_bytes() == initial
    assert second_image.name.endswith("_cafebabe.tif")
    assert second_mask.name.endswith("_cafebabe_masks.tif")


def test_export_rejects_non_source_image_bad_mask_and_missing_folder(tmp_path: Path):
    with pytest.raises(ValueError, match="2D numeric source image"):
        export_review_pair(
            np.zeros((3, 4, 3), dtype=np.uint8), _mask(), output_dir=tmp_path,
            fov_name="fov", time_point="t0", optical_channel="phase", side_channel=1,
        )
    with pytest.raises(ValueError, match="2D integer"):
        export_review_pair(
            _image(), None, output_dir=tmp_path, fov_name="fov", time_point="t0",
            optical_channel="phase", side_channel=1,
        )
    with pytest.raises(ValueError, match="matching"):
        export_review_pair(
            _image(), np.zeros((2, 2), dtype=np.uint8), output_dir=tmp_path,
            fov_name="fov", time_point="t0", optical_channel="phase", side_channel=1,
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        export_review_pair(
            _image(), _mask(), output_dir=tmp_path / "missing", fov_name="fov",
            time_point="t0", optical_channel="phase", side_channel=1,
        )
