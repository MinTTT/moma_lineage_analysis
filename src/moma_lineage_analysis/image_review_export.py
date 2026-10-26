"""Write an original source image and its label mask as a tagged TIFF pair."""

from __future__ import annotations

from pathlib import Path
import re
import secrets

import numpy as np
import tifffile

# Lets Streamlit sessions detect and refresh the pre-original-image exporter.
EXPORTS_ORIGINAL_IMAGE = True


def _filename_part(value, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip(" ._")
    return text or fallback


def _random_tag() -> str:
    """Return an eight-character hexadecimal tag for one image/mask pair."""

    return secrets.token_hex(4)


def _validated_pair(image: np.ndarray, mask: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 2
        or image.size == 0
        or not np.issubdtype(image.dtype, np.number)
        or np.issubdtype(image.dtype, np.complexfloating)
    ):
        raise ValueError("TIFF export requires a non-empty 2D numeric source image.")
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.shape != image.shape
        or not np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise ValueError("Mask export requires a 2D integer label image matching the source image.")
    if (mask < 0).any():
        raise ValueError("Mask labels must be nonnegative.")


def export_review_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    output_dir: str | Path,
    fov_name: str,
    time_point: str,
    optical_channel: str,
    side_channel,
) -> tuple[Path, Path]:
    """Save the original source image and label mask without any overlays.

    Pixel values and dtypes are preserved. Both names share one random
    eight-character tag, and the mask name adds _masks. If writing either file
    fails, files created by this attempt are removed.
    """

    _validated_pair(image, mask)
    directory = Path(output_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image export folder does not exist: {directory}")

    basename = "_".join((
        _filename_part(fov_name, "fov"),
        _filename_part(Path(time_point).stem, "frame"),
        _filename_part(optical_channel, "channel"),
        "side" + _filename_part(side_channel, "unknown"),
    ))
    for _ in range(100):
        tag = _random_tag()
        image_path = directory / f"{basename}_{tag}.tif"
        mask_path = directory / f"{basename}_{tag}_masks.tif"
        image_created = False
        mask_created = False
        try:
            with image_path.open("xb") as handle:
                image_created = True
                tifffile.imwrite(handle, image, photometric="minisblack")
            with mask_path.open("xb") as handle:
                mask_created = True
                tifffile.imwrite(handle, mask, photometric="minisblack")
            return image_path, mask_path
        except FileExistsError:
            # A rare tag collision must not touch either pre-existing file.
            if image_created:
                image_path.unlink(missing_ok=True)
            if mask_created:
                mask_path.unlink(missing_ok=True)
        except Exception:
            if image_created:
                image_path.unlink(missing_ok=True)
            if mask_created:
                mask_path.unlink(missing_ok=True)
            raise
    raise FileExistsError(f"Could not allocate a unique TIFF pair for {basename} in {directory}.")
