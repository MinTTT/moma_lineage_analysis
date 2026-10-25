"""Backend helpers for reviewing images behind trajectory selections.

The upstream MoMa export layout is
``cells_sideChannel_image_dict["fov_channels_image_dict"][side][time][optical]``
and ``...["fov_cells_mask_dict"][side][time]``. These helpers deliberately
return the saved side-channel arrays unchanged (including any upstream
orientation choice) and never derive image paths from trajectory names.

Chosen overlay colors:
- selected cells: vivid cyan ``(0, 220, 255)``;
- missing selected labels are reported, not substituted;
- all other cells (when requested): warm orange ``(255, 165, 0)``.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ImageReviewError",
    "ReviewObservation",
    "ReviewFrame",
    "ReviewSnapshot",
    "make_review_snapshot",
    "load_review_pickle",
    "optical_channels_for_frames",
    "get_review_frame",
    "get_review_image_shape",
    "normalize_review_image",
    "render_review_overlay",
]

CSV_SUFFIX = "_cells_df.csv"
PICKLE_SUFFIX = "_exported_data.pkl"
SELECTED_COLOR = (0, 220, 255)  # vivid cyan
OTHER_CELL_COLOR = (255, 165, 0)  # distinguishable warm orange
_NATURAL_TIME_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


class ImageReviewError(ValueError):
    """Raised when review metadata or an export payload cannot be trusted."""


@dataclass(frozen=True)
class ReviewObservation:
    """One selected observation resolved against its source CSV row."""

    fov_cell_index: int
    cell_index: int
    time_h: float | None
    time_s: float | None
    excluded: bool
    division: bool


@dataclass(frozen=True)
class ReviewFrame:
    """All selected observations that share one side-channel time frame."""

    side_channel: int | str
    time_point: str
    time_h: float | None
    time_s: float | None
    observations: tuple[ReviewObservation, ...]


@dataclass(frozen=True)
class ReviewSnapshot:
    """Detached, time-ordered review request for one trajectory."""

    trajectory_key: str
    source_csv: str
    pickle_path: str
    frames: tuple[ReviewFrame, ...]


def _require_mapping(value, description: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ImageReviewError(f"{description} must be a mapping; got {type(value).__name__}.")
    return value


def _is_bool(value) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not _is_bool(value)


def _normalize_fov_cell_index(value) -> int:
    if not _is_integer(value):
        raise ImageReviewError(
            "Selected fov_cell_index values must be finite integers; "
            f"got {value!r}."
        )
    return int(value)


def _normalize_cell_label(value, description: str = "cell_index") -> int:
    if not _is_integer(value):
        raise ImageReviewError(f"{description} must be a positive integer label; got {value!r}.")
    label = int(value)
    if label <= 0:
        raise ImageReviewError(f"{description} must be a positive integer label; got {label}.")
    return label


def _optional_time(value, field: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImageReviewError(f"{field} must be numeric or missing; got {value!r}.") from exc
    if not np.isfinite(number):
        return None
    return float(number)


def _metadata_text(value, field: str) -> str:
    if value is None or isinstance(value, bytes):
        raise ImageReviewError(f"CSV {field} metadata must be text, not {type(value).__name__}.")
    if isinstance(value, float) and not np.isfinite(value):
        raise ImageReviewError(f"CSV {field} metadata must not be NaN or infinite.")
    text = str(value)
    if not text:
        raise ImageReviewError(f"CSV {field} metadata must not be empty.")
    return text


def _derive_pickle_path(source_csv_path: str | Path) -> Path:
    source = Path(source_csv_path)
    if source.suffix != ".csv" or not source.name.endswith(CSV_SUFFIX):
        raise ImageReviewError(
            "Image review requires the original *_cells_df.csv source path. "
            f"Got {source!s}; rerun Process Data in the exported-data directory."
        )
    if not source.is_file():
        raise ImageReviewError(
            "Cannot locate the original source CSV for image review. "
            f"Expected {source!s}; reload the data with Process Data."
        )
    return source.with_name(source.name[: -len(CSV_SUFFIX)] + PICKLE_SUFFIX)


def _natural_frame_number(time_point: str) -> int | None:
    # ND2 exports commonly use ``t2.tiff``/``t00001.tif``; compare their stem
    # rather than letting the extension defeat natural numeric ordering.
    match = _NATURAL_TIME_RE.match(Path(time_point).stem)
    if match is None:
        return None
    return int(match.group("number"))


def _comparison_scales_match(frame_a: ReviewFrame, frame_b: ReviewFrame) -> bool:
    if frame_a.time_h is None or frame_b.time_h is None:
        return False
    if frame_a.time_s is None or frame_b.time_s is None:
        return False
    delta_h = frame_b.time_h - frame_a.time_h
    delta_s = frame_b.time_s - frame_a.time_s
    scale = abs(frame_a.time_s) + abs(frame_b.time_s) + abs(delta_s)
    tolerance = max(1e-9, 1e-9 * scale)
    return abs(delta_s - delta_h * 3600.0) <= tolerance


def _seconds_consistent_with_hours(frames: Sequence[ReviewFrame]) -> bool:
    with_hours = [frame for frame in frames if frame.time_h is not None]
    if not with_hours:
        return True
    if any(frame.time_s is None for frame in with_hours):
        return False
    first = with_hours[0]
    return all(_comparison_scales_match(first, frame) for frame in with_hours[1:])


def _frame_sort_key(order: int, frame: ReviewFrame, *, use_hours: bool, use_seconds: bool):
    natural = _natural_frame_number(frame.time_point)
    natural_key = (1, natural) if natural is not None else (2, frame.time_point)
    if use_hours:
        return (0, frame.time_h, order)
    if use_seconds:
        return (0, frame.time_s, order)
    return (1, natural_key, order)


def _sort_frames_chronologically(frames: Sequence[ReviewFrame]) -> tuple[ReviewFrame, ...]:
    """Sort on one coherent clock; never lexicographic t10-before-t2 order."""

    use_hours = all(frame.time_h is not None for frame in frames)
    use_seconds = False
    if not use_hours:
        use_seconds = all(frame.time_s is not None for frame in frames) and (
            _seconds_consistent_with_hours(frames)
        )
    ordered = [
        frame
        for _, frame in sorted(
            enumerate(frames),
            key=lambda item: _frame_sort_key(
                item[0], item[1], use_hours=use_hours, use_seconds=use_seconds
            ),
        )
    ]
    return tuple(ordered)


def _trajectory_frame(trajectory):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a package dependency
        raise ImageReviewError("pandas is required to resolve trajectory observations.") from exc
    if not isinstance(trajectory, pd.DataFrame):
        raise ImageReviewError(
            "trajectory_data['trajectory'] must be the full trajectory pandas DataFrame, "
            f"not {type(trajectory).__name__}."
        )
    return trajectory


def _annotation_ids(values, description: str) -> set[int]:
    if values is None:
        return set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ImageReviewError(f"trajectory_data[{description!r}] must be an iterable of ids.") from exc
    return {_normalize_fov_cell_index(value) for value in iterator}


def make_review_snapshot(
    trajectory_key: str,
    trajectory_data: dict,
    selected_indices: Iterable[int],
    source_csv_path: str | Path | None,
) -> ReviewSnapshot:
    """Resolve selected full-trajectory rows into a detached image snapshot.

    The full ``trajectory_data['trajectory']`` frame is required; a plotting
    subset cannot safely resolve selections. ``selected_indices`` contains
    ``fov_cell_index`` values and may repeat them. Source and pickle paths use
    exact ``*_cells_df.csv`` -> ``*_exported_data.pkl`` suffix replacement,
    preserving padded FOV stems. Image data is not read here.
    """

    if not isinstance(trajectory_key, str) or not trajectory_key:
        raise ImageReviewError("trajectory_key must be a non-empty string.")
    if source_csv_path is None:
        raise ImageReviewError(
            "The original *_cells_df.csv path is unavailable for this trajectory. "
            "Image review cannot infer it from the trajectory key; rerun Process Data "
            "to reload the exported-data directory."
        )
    if not isinstance(trajectory_data, Mapping):
        raise ImageReviewError("trajectory_data must be a mapping.")
    if "trajectory" not in trajectory_data:
        raise ImageReviewError("trajectory_data is missing the full 'trajectory' DataFrame.")
    trajectory = _trajectory_frame(trajectory_data["trajectory"])
    required = ("fov_cell_index", "channels", "time_point", "cell_index")
    missing = [column for column in required if column not in trajectory.columns]
    if missing:
        raise ImageReviewError(
            "trajectory_data['trajectory'] is missing required metadata columns: "
            + ", ".join(missing)
        )

    try:
        raw_selected = list(selected_indices)
    except TypeError as exc:
        raise ImageReviewError("selected_indices must be an iterable of fov_cell_index values.") from exc
    selected_ids: list[int] = []
    seen: set[int] = set()
    for raw_id in raw_selected:
        cell_id = _normalize_fov_cell_index(raw_id)
        if cell_id not in seen:
            seen.add(cell_id)
            selected_ids.append(cell_id)

    pickle_path = _derive_pickle_path(source_csv_path)
    excluded_ids = _annotation_ids(trajectory_data.get("excluded_indices", ()), "excluded_indices")
    division_ids = _annotation_ids(trajectory_data.get("div_indices", ()), "div_indices")

    groups: dict[tuple[int | str, str], list[ReviewObservation]] = {}
    group_order: list[tuple[int | str, str]] = []
    group_times: dict[tuple[int | str, str], tuple[float | None, float | None]] = {}

    for cell_id in selected_ids:
        matches = trajectory.index[trajectory["fov_cell_index"] == cell_id].tolist()
        if not matches:
            raise ImageReviewError(
                f"Selected fov_cell_index {cell_id} is absent from the full trajectory. "
                "Reload the data with Process Data before reviewing images."
            )
        if len(matches) > 1:
            raise ImageReviewError(
                f"Selected fov_cell_index {cell_id} matches {len(matches)} trajectory rows; "
                "ambiguous duplicate row IDs cannot be reviewed safely."
            )
        # Positional lookup is immune to duplicate arbitrary DataFrame index
        # labels, for which ``.loc[matches[0]]`` may return a DataFrame.
        row = trajectory.loc[trajectory["fov_cell_index"] == cell_id].iloc[0]
        side_raw = row["channels"]
        if _is_integer(side_raw):
            side: int | str = int(side_raw)
        else:
            side_text = _metadata_text(side_raw, "channels")
            side_alias = _numeric_alias(side_text)
            side = side_alias if side_alias is not None else side_text
        time_point = _metadata_text(row["time_point"], "time_point")
        label = _normalize_cell_label(row["cell_index"])
        time_h = _optional_time(row.get("time_h"), "time_h")
        time_s = _optional_time(row.get("time_s"), "time_s")
        observation = ReviewObservation(
            fov_cell_index=cell_id,
            cell_index=label,
            time_h=time_h,
            time_s=time_s,
            excluded=cell_id in excluded_ids,
            division=cell_id in division_ids,
        )
        key = (side, time_point)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
            group_times[key] = (time_h, time_s)
        else:
            expected_h, expected_s = group_times[key]
            if time_h != expected_h or time_s != expected_s:
                raise ImageReviewError(
                    "Rows grouped into the same side-channel/time frame have inconsistent "
                    f"time metadata at side={side!r}, time_point={time_point!r}."
                )
        groups[key].append(observation)

    frames: list[ReviewFrame] = []
    for key in group_order:
        observations = tuple(groups[key])
        time_h, time_s = group_times[key]
        frames.append(
            ReviewFrame(
                side_channel=key[0],
                time_point=key[1],
                time_h=time_h,
                time_s=time_s,
                observations=observations,
            )
        )

    return ReviewSnapshot(
        trajectory_key=trajectory_key,
        source_csv=str(Path(source_csv_path)),
        pickle_path=str(pickle_path),
        frames=_sort_frames_chronologically(frames),
    )


def load_review_pickle(path: str | Path) -> dict:
    """Read an explicitly requested upstream export pickle and validate shape.

    Standard-library pickle loading is used only because these are user-owned
    pipeline artifacts and the user has authorized this path. No upstream
    module, Cellpose model, or GPU resource is imported.
    """

    pickle_path = Path(path)
    if not pickle_path.is_file():
        raise ImageReviewError(f"Review pickle does not exist: {pickle_path!s}.")
    try:
        with pickle_path.open("rb") as handle:
            export = pickle.load(handle)
    except Exception as exc:
        raise ImageReviewError(
            f"Could not read review pickle {pickle_path!s}: {exc.__class__.__name__}: {exc}"
        ) from exc
    if not isinstance(export, dict):
        raise ImageReviewError(
            f"Review pickle {pickle_path!s} must contain a built-in dict; "
            f"got {type(export).__name__}."
        )
    info = export.get("info_dict")
    side = export.get("cells_sideChannel_image_dict")
    _require_mapping(info, "Review pickle 'info_dict'")
    side = _require_mapping(side, "Review pickle 'cells_sideChannel_image_dict'")
    _require_mapping(side.get("fov_channels_image_dict"), "Review pickle images mapping")
    _require_mapping(side.get("fov_cells_mask_dict"), "Review pickle masks mapping")
    return export


def _numeric_alias(value) -> int | None:
    if _is_integer(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d+", text):
            return int(text)
    return None


def _side_keys(mapping: Mapping, description: str) -> list[int | str]:
    """Validate one mapping's side keys and reject ambiguous numeric aliases."""

    keys = list(mapping.keys())
    numeric_aliases: dict[int, list[int | str]] = {}
    for key in keys:
        alias = _numeric_alias(key)
        if alias is not None:
            numeric_aliases.setdefault(alias, []).append(key)
    duplicates = {
        alias: aliases for alias, aliases in numeric_aliases.items() if len(aliases) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"{alias}: {[repr(key) for key in aliases]}" for alias, aliases in duplicates.items()
        )
        raise ImageReviewError(
            f"Ambiguous duplicate numeric aliases in {description}: {detail}."
        )
    return keys


def _resolve_side_channel(
    mapping: Mapping, requested: int | str, description: str
) -> int | str:
    """Resolve a side by exact key or one unambiguous numeric alias."""

    keys = _side_keys(mapping, description)
    if requested in keys:
        return requested
    requested_alias = _numeric_alias(requested)
    if requested_alias is not None:
        matches = [key for key in keys if _numeric_alias(key) == requested_alias]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ImageReviewError(
                f"Side-channel {requested!r} has duplicate numeric aliases in "
                f"{description}: {matches!r}."
            )
    raise ImageReviewError(
        f"Side-channel {requested!r} is unavailable in {description}; "
        f"available keys: {keys!r}."
    )


def _payload(export: Mapping):
    side = _require_mapping(
        export.get("cells_sideChannel_image_dict"), "Export 'cells_sideChannel_image_dict'"
    )
    images = _require_mapping(side.get("fov_channels_image_dict"), "Export images mapping")
    masks = _require_mapping(side.get("fov_cells_mask_dict"), "Export masks mapping")
    info = _require_mapping(export.get("info_dict"), "Export 'info_dict'")
    return images, masks, info


def _time_mapping(mapping: Mapping, side: int | str, description: str) -> Mapping:
    value = mapping.get(side)
    if value is None:
        raise ImageReviewError(f"{description} has no entry for side-channel {side!r}.")
    return _require_mapping(value, f"{description}[{side!r}]")


def _ordered_names(names: Iterable[str], detection_channel) -> tuple[str, ...]:
    ordered: list[str] = []
    for name in names:
        if isinstance(name, str) and name and name not in ordered:
            ordered.append(name)
    if isinstance(detection_channel, str) and detection_channel in ordered:
        ordered.remove(detection_channel)
        ordered.insert(0, detection_channel)
    return tuple(ordered)


def optical_channels_for_frames(
    export: dict, frames: Sequence[ReviewFrame]
) -> tuple[str, ...]:
    """Return the union of optical channels available in selected frames.

    The detection channel from ``info_dict`` is listed first when supported.
    A frame missing from the export is skipped; if no selected frame supports
    any optical channel, an ``ImageReviewError`` is raised.
    """

    _require_mapping(export, "Export payload")
    images, _, info = _payload(export)
    _side_keys(images, "Export images mapping")  # aliases must be unambiguous
    detection = info.get("cell_detection_opticalChannel_name")
    supported: list[str] = []
    for frame in frames:
        try:
            side = _resolve_side_channel(images, frame.side_channel, "Export images mapping")
            by_time = _time_mapping(images, side, "Export images mapping")
        except ImageReviewError:
            continue
        channels = by_time.get(frame.time_point)
        if channels is None:
            continue
        channels = _require_mapping(
            channels, f"Export images mapping[{side!r}][{frame.time_point!r}]"
        )
        for name in channels.keys():
            if isinstance(name, str) and name and name not in supported:
                supported.append(name)
    if not supported:
        raise ImageReviewError(
            "None of the selected frames has an optical-channel image in the export."
        )
    return _ordered_names(supported, detection if detection in supported else None)


def _validated_image(value, frame: ReviewFrame, optical_channel: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ImageReviewError(
            f"Image for side={frame.side_channel!r}, time_point={frame.time_point!r}, "
            f"optical={optical_channel!r} must be a NumPy array; got {type(value).__name__}."
        )
    if value.ndim != 2:
        raise ImageReviewError(
            f"Image for side={frame.side_channel!r}, time_point={frame.time_point!r}, "
            f"optical={optical_channel!r} must be 2D; got shape {value.shape}."
        )
    if value.size == 0:
        raise ImageReviewError(
            f"Image for side={frame.side_channel!r}, time_point={frame.time_point!r}, "
            f"optical={optical_channel!r} must not be empty; got shape {value.shape}."
        )
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise ImageReviewError(
            f"Image for side={frame.side_channel!r}, time_point={frame.time_point!r}, "
            f"optical={optical_channel!r} must be real numeric; got dtype {value.dtype}."
        )
    if not np.isfinite(value).any():
        raise ImageReviewError(
            f"Image for side={frame.side_channel!r}, time_point={frame.time_point!r}, "
            f"optical={optical_channel!r} has no finite numeric pixels."
        )
    return value


def _mask_error(
    image: np.ndarray, frame: ReviewFrame, reason: str, value=None
) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if reason == "malformed":
        return (
            f"Mask malformed for side={frame.side_channel!r}, "
            f"time_point={frame.time_point!r} (shape {shape}, dtype {dtype}, expected "
            f"nonnegative integer shape {image.shape}); showing the image without outlines."
        )
    return (
        f"Mask {reason} for side={frame.side_channel!r}, "
        f"time_point={frame.time_point!r} (shape {shape}, dtype {dtype}, expected "
        f"nonnegative integer shape {image.shape}); showing the image without outlines."
    )


def _validated_mask(
    value, image: np.ndarray, frame: ReviewFrame
) -> tuple[np.ndarray | None, str | None]:
    """Return ``(mask, diagnostic)`` without global warning side effects."""

    if value is None:
        return None, (
            f"Mask missing for side={frame.side_channel!r}, time_point={frame.time_point!r}; "
            "the image will be shown without cell outlines."
        )
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.shape != image.shape
        or value.size == 0
    ):
        return None, _mask_error(image, frame, "malformed", value)
    if (
        not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.bool_)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        return None, _mask_error(image, frame, "malformed", value)
    if not np.isfinite(value).all() or (value < 0).any() or (value % 1 != 0).any():
        return None, _mask_error(image, frame, "malformed", value)
    return value, None


def _review_image(export: dict, frame: ReviewFrame, optical_channel: str) -> np.ndarray:
    """Resolve and validate one image without reading or validating its mask."""

    _require_mapping(export, "Export payload")
    if not isinstance(optical_channel, str) or not optical_channel:
        raise ImageReviewError("optical_channel must be a non-empty string.")
    images, _, _ = _payload(export)
    side = _resolve_side_channel(images, frame.side_channel, "Export images mapping")
    image_times = _time_mapping(images, side, "Export images mapping")
    channel_map = image_times.get(frame.time_point)
    if channel_map is None:
        raise ImageReviewError(
            f"Image frame side={side!r}, time_point={frame.time_point!r} is missing; "
            "time_point matching is exact and no frame string will be guessed."
        )
    channel_map = _require_mapping(
        channel_map, f"Export images mapping[{side!r}][{frame.time_point!r}]"
    )
    if optical_channel not in channel_map:
        raise ImageReviewError(
            f"Optical channel {optical_channel!r} is missing for side={side!r}, "
            f"time_point={frame.time_point!r}; available: {list(channel_map)!r}."
        )
    return _validated_image(channel_map[optical_channel], frame, optical_channel)


def get_review_image_shape(
    export: dict, frame: ReviewFrame, optical_channel: str
) -> tuple[int, int]:
    """Return image dimensions for responsive layout without mask processing."""

    return _review_image(export, frame, optical_channel).shape


def get_review_frame(
    export: dict, frame: ReviewFrame, optical_channel: str
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...]]:
    """Return ``(image, mask, diagnostics)`` for one exact side/time frame.

    The image array is the saved side-channel crop, never flipped or cropped
    again. A missing image raises ``ImageReviewError`` so the UI can render a
    per-frame warning. Mask and selected-label problems are returned as
    diagnostics while preserving the displayable image.
    """

    image = _review_image(export, frame, optical_channel)
    _, masks, _ = _payload(export)

    messages: list[str] = []
    mask_value = None
    mask_side = None
    try:
        mask_side = _resolve_side_channel(masks, frame.side_channel, "Export masks mapping")
    except ImageReviewError as exc:
        if "duplicate numeric aliases" in str(exc):
            raise
        messages.append(
            f"Mask missing for side={frame.side_channel!r}: "
            f"{exc}; showing the image without outlines."
        )
    else:
        mask_times = masks.get(mask_side)
        if isinstance(mask_times, Mapping):
            mask_value = mask_times.get(frame.time_point)
        else:
            messages.append(
                f"Mask mapping malformed for side={mask_side!r}; showing the image "
                "without outlines."
            )
    mask, diagnostic = _validated_mask(mask_value, image, frame)
    if diagnostic is not None:
        messages.append(diagnostic)

    if mask is not None:
        present = np.unique(mask)
        for observation in frame.observations:
            if observation.cell_index not in present:
                messages.append(
                    f"Selected cell label {observation.cell_index} is absent from mask "
                    f"side={mask_side!r}, time_point={frame.time_point!r}; no other cell "
                    "will be substituted."
                )
    return image, mask, tuple(messages)


def _validate_2d_numeric(image: np.ndarray, description: str = "image") -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ImageReviewError(f"{description} must be a NumPy array; got {type(image).__name__}.")
    if image.ndim != 2:
        raise ImageReviewError(f"{description} must be 2D; got shape {image.shape}.")
    if image.size == 0:
        raise ImageReviewError(f"{description} must not be empty; got shape {image.shape}.")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise ImageReviewError(f"{description} must be real numeric; got dtype {image.dtype}.")
    return image


def normalize_review_image(
    image: np.ndarray, contrast_limits: tuple[float, float] | None = None
) -> np.ndarray:
    """Normalize a finite/non-finite numeric 2D image to uint8 grayscale.

    Default limits are the 1st and 99th finite percentiles. Constant images
    map to their finite intensity when it already lies in the uint8 range
    (otherwise neutral mid-gray). Explicit finite limits may be shared across
    a displayed page. The input array is never modified.
    """

    source = _validate_2d_numeric(image)
    values = source.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(source.shape, 127, dtype=np.uint8)
    if contrast_limits is not None:
        if len(contrast_limits) != 2:
            raise ImageReviewError("contrast_limits must contain exactly two finite numbers.")
        try:
            low = float(contrast_limits[0])
            high = float(contrast_limits[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ImageReviewError("contrast_limits must contain two finite numbers.") from exc
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ImageReviewError("contrast_limits must contain two finite numbers.")
    else:
        low, high = np.percentile(values[finite], (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        return np.full(source.shape, 127, dtype=np.uint8)
    if high <= low:
        midpoint = 127
        if high == low and 0 <= low <= 255:
            midpoint = int(np.rint(low))
        return np.full(source.shape, midpoint, dtype=np.uint8)
    scaled = np.zeros(source.shape, dtype=np.float64)
    scaled[finite] = (values[finite] - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _label_set(values: Iterable[int], description: str) -> tuple[int, ...]:
    try:
        items = list(values)
    except TypeError as exc:
        raise ImageReviewError(f"{description} must be an iterable of positive labels.") from exc
    labels: list[int] = []
    seen: set[int] = set()
    for value in items:
        label = _normalize_cell_label(value, description)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return tuple(labels)


def _label_neighbors(mask: np.ndarray) -> np.ndarray:
    # Exact four-neighbor label comparison keeps touching-label boundaries and
    # marks cells that reach the image edge (padding is outside the image).
    padded = np.pad(mask, 1, mode="constant", constant_values=-1)
    center = padded[1:-1, 1:-1]
    return (
        (center != padded[:-2, 1:-1])
        | (center != padded[2:, 1:-1])
        | (center != padded[1:-1, :-2])
        | (center != padded[1:-1, 2:])
    )


def _blend(base: np.ndarray, region: np.ndarray, color: tuple[int, int, int], opacity: float) -> None:
    if not region.any():
        return
    base[region] = np.rint(
        (1.0 - opacity) * base[region].astype(np.float64) + opacity * np.asarray(color)
    ).astype(np.uint8)


def _mask_boundary(mask: np.ndarray, labels: set[int]) -> np.ndarray:
    if not labels:
        return np.zeros(mask.shape, dtype=bool)
    selected = np.isin(mask, list(labels))
    return selected & _label_neighbors(mask)


def render_review_overlay(
    image: np.ndarray,
    mask: np.ndarray | None,
    selected_labels: Iterable[int],
    *,
    fill: bool = False,
    outline: bool = True,
    all_cells: bool = False,
    opacity: float = 0.35,
    contrast_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render selected label outlines over normalized grayscale RGB.

    Selected cells are vivid cyan ``(0, 220, 255)``; other cells, when
    ``all_cells=True``, are warm orange ``(255, 165, 0)``. Boundaries are
    computed from label-neighbor changes in the saved mask, so boundaries
    between touching labels remain visible. Background pixels that do not
    touch a labeled boundary are unchanged.
    """

    source = _validate_2d_numeric(image)
    labels = _label_set(selected_labels, "selected_labels")
    try:
        alpha = float(opacity)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImageReviewError(f"opacity must be finite in [0, 1]; got {opacity!r}.") from exc
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ImageReviewError(f"opacity must be finite in [0, 1]; got {opacity!r}.")

    normalized = normalize_review_image(source, contrast_limits)
    rgb = np.repeat(normalized[:, :, None], 3, axis=2)
    if mask is None:
        return rgb
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != source.shape:
        raise ImageReviewError(
            f"mask must be a 2D array matching image shape {source.shape}; "
            f"got {getattr(mask, 'shape', None)}."
        )
    if mask.size == 0:
        raise ImageReviewError("mask must not be empty.")
    if (
        not np.issubdtype(mask.dtype, np.number)
        or np.issubdtype(mask.dtype, np.bool_)
        or np.issubdtype(mask.dtype, np.complexfloating)
    ):
        raise ImageReviewError(f"mask must be real numeric; got dtype {mask.dtype}.")
    if not np.isfinite(mask).all():
        raise ImageReviewError("mask must contain only finite label values.")
    if (mask < 0).any() or (mask % 1 != 0).any():
        raise ImageReviewError("mask must contain nonnegative integer cell labels.")

    label_values = np.unique(mask)
    positive = {int(value) for value in label_values if int(value) > 0}
    selected = set(labels)
    present_selected = selected & positive
    if outline:
        selected_boundary = _mask_boundary(mask, present_selected)
        rgb[selected_boundary] = SELECTED_COLOR
        if all_cells:
            other_boundary = _mask_boundary(mask, positive - present_selected)
            rgb[other_boundary & ~selected_boundary] = OTHER_CELL_COLOR
    if fill:
        if all_cells:
            other_region = np.isin(mask, list(positive - present_selected))
            _blend(rgb, other_region, OTHER_CELL_COLOR, alpha)
        if present_selected:
            region = np.isin(mask, list(present_selected))
            _blend(rgb, region, SELECTED_COLOR, alpha)
    return rgb
