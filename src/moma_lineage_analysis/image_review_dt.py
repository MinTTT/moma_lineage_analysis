"""Read and rasterize saved DT spine/rib geometry for image review.

Geometry is joined by saved cell labels, never by parameter-list position. Ribs
are pixel-coordinate rows ``[x_left, y_left, x_right, y_right]`` in the saved
image orientation; their ordered midpoints form the spine. No morphology is
recomputed, and old exports without meshes remain usable with diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .image_review import ImageReviewError, ReviewFrame, _resolve_side_channel

__all__ = ["DTMesh", "DT_RIB_COLOR", "DT_SPINE_COLOR", "get_dt_meshes", "render_dt_overlay"]

DT_RIB_COLOR = (255, 255, 0)  # bright yellow, RGB
DT_SPINE_COLOR = (255, 0, 255)  # bright magenta, RGB


@dataclass(frozen=True)
class DTMesh:
    """One cell's saved ribs; getter results own a read-only float64 array."""

    cell_index: int
    ribs: np.ndarray


def _cell_label(value, description: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise ImageReviewError(f"{description} must be a positive integer cell_index; got {value!r}.")
    return int(value)


def _validated_ribs(value, description: str) -> np.ndarray:
    """Validate without coercing text/complex coordinates or mutating the source."""

    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImageReviewError(f"{description} must be a finite numeric mesh of shape (N, 4), N >= 1.") from exc
    if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] != 4:
        raise ImageReviewError(
            f"{description} must have shape (N, 4), N >= 1; got {source.shape}."
        )
    if source.dtype.kind not in "iuf":
        raise ImageReviewError(f"{description} must contain real numeric pixel coordinates.")
    with np.errstate(over="ignore", invalid="ignore"):
        ribs = np.array(source, dtype=np.float64, copy=True)
    if not np.isfinite(ribs).all():
        raise ImageReviewError(f"{description} must contain only finite pixel coordinates.")
    return ribs


def get_dt_meshes(
    export: dict, frame: ReviewFrame, *, all_cells: bool = False
) -> tuple[tuple[DTMesh, ...], tuple[str, ...]]:
    """Return saved meshes and diagnostics for one exact side-channel/time frame.

    The lookup is
    ``export['cells_sideChannel_image_dict']['fov_cells_pars_dict'][side][time]``.
    Side keys use the existing review resolver (including unique padded numeric
    aliases); time_point and cell_index are exact. By default only snapshot
    observation labels are included. With all_cells, remaining parameter labels
    follow them in saved list order. An ambiguous duplicate label is skipped in
    full, even if one of its records has a usable mesh.

    Missing/malformed optional geometry returns diagnostics. Invalid API inputs
    raise ImageReviewError. Algorithm metadata is optional: a valid saved mesh
    suffices for older exports. No geometry is synthesized from legacy fields.
    """

    if not isinstance(export, Mapping):
        raise ImageReviewError("Export payload must be a mapping.")
    if not isinstance(frame, ReviewFrame):
        raise ImageReviewError("frame must be a ReviewFrame.")
    if not isinstance(frame.time_point, str) or not frame.time_point:
        raise ImageReviewError("ReviewFrame time_point must be a non-empty exact frame name.")
    if not isinstance(all_cells, (bool, np.bool_)):
        raise ImageReviewError("all_cells must be a boolean.")
    try:
        selected = tuple(dict.fromkeys(
            _cell_label(observation.cell_index, "Selected observation")
            for observation in frame.observations
        ))
    except (AttributeError, TypeError) as exc:
        raise ImageReviewError("ReviewFrame observations must contain cell_index labels.") from exc
    if not selected and not all_cells:
        return (), ()

    context = f"side={frame.side_channel!r}, time_point={frame.time_point!r}"

    def unavailable(message: str) -> tuple[tuple[DTMesh, ...], tuple[str, ...]]:
        return (), (f"DT geometry unavailable for {context}: {message}",)

    payload = export.get("cells_sideChannel_image_dict")
    if not isinstance(payload, Mapping):
        return unavailable("missing or malformed cells_sideChannel_image_dict.")
    parameters = payload.get("fov_cells_pars_dict")
    if not isinstance(parameters, Mapping):
        return unavailable("missing or malformed fov_cells_pars_dict; this export may lack saved DT meshes.")
    try:
        side = _resolve_side_channel(parameters, frame.side_channel, "Export DT parameters mapping")
    except ImageReviewError as exc:
        return unavailable(str(exc))
    times = parameters[side]
    if not isinstance(times, Mapping):
        return unavailable("the side-channel parameter entry must be a time mapping.")
    if frame.time_point not in times:
        return unavailable("parameter time_point is missing; time matching is exact.")
    records = times[frame.time_point]
    if not isinstance(records, list):
        return unavailable("the frame parameters must be a list of cell parameter mappings.")

    messages: list[str] = []
    by_label: dict[int, Mapping | None] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            messages.append(f"DT parameter record {index} for {context} is malformed: expected a mapping.")
            continue
        try:
            label = _cell_label(record.get("cell_index"), f"DT parameter record {index}")
        except ImageReviewError as exc:
            messages.append(f"{exc} ({context})")
            continue
        # A duplicate invalidates every record for that label; never choose one.
        by_label[label] = None if label in by_label else record

    targets = list(selected)
    if all_cells:
        targets.extend(label for label in by_label if label not in selected)
    meshes: list[DTMesh] = []
    for label in targets:
        description = f"Cell label {label} for {context}"
        if label not in by_label:
            messages.append(f"{description} has no saved parameter record.")
            continue
        record = by_label[label]
        if record is None:
            messages.append(f"{description} has duplicate parameter records; DT mesh skipped as ambiguous.")
            continue
        if record.get("mesh") is None:
            messages.append(
                f"{description} has no saved DT mesh; legacy geometry is not reconstructed."
            )
            continue
        try:
            ribs = _validated_ribs(record["mesh"], f"{description} DT mesh")
        except ImageReviewError as exc:
            messages.append(str(exc))
            continue
        ribs.flags.writeable = False
        meshes.append(DTMesh(label, ribs))
    return tuple(meshes), tuple(messages)


def _clip_segment(
    start: Sequence[float], end: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip before rasterization, including for extremely large finite endpoints."""

    x0, y0 = map(float, start)
    x1, y1 = map(float, end)
    right, bottom = width - 1, height - 1
    if 0 <= x0 <= right and 0 <= x1 <= right and 0 <= y0 <= bottom and 0 <= y1 <= bottom:
        return x0, y0, x1, y1
    if max(x0, x1) < 0 or min(x0, x1) > right or max(y0, y1) < 0 or min(y0, y1) > bottom:
        return None

    # Exact rational clipping avoids overflow and cancellation when finite
    # float endpoints are far beyond the image. Only out-of-bounds segments
    # take this path; no allocation depends on their original lengths.
    ax, ay, bx, by = map(Fraction, (x0, y0, x1, y1))
    dx, dy = bx - ax, by - ay
    lower, upper = Fraction(0), Fraction(1)
    for p, q in ((-dx, ax), (dx, right - ax), (-dy, ay), (dy, bottom - ay)):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return float(ax + lower * dx), float(ay + lower * dy), float(ax + upper * dx), float(ay + upper * dy)


def _draw_segment(
    rgb: np.ndarray, start: Sequence[float], end: Sequence[float], color: tuple[int, int, int]
) -> None:
    height, width = rgb.shape[:2]
    segment = _clip_segment(start, end, width, height)
    if segment is None:
        return
    x0, y0, x1, y1 = (int(round(value)) for value in segment)
    # Bresenham draws one pixel per major-axis step, bounded by image dimensions.
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        rgb[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def render_dt_overlay(rgb: np.ndarray, meshes: Sequence[DTMesh]) -> np.ndarray:
    """Return an RGB uint8 copy with yellow ribs and magenta midpoint spines.

    Coordinates are used as saved: X is the column and Y is the row. Every rib
    and every adjacent midpoint pair is drawn as a clipped one-pixel line. A
    single-rib cell receives a magenta midpoint. Spines are painted last so they
    remain visible at rib intersections; separate cells are never connected.
    Malformed RGB arrays or meshes raise ImageReviewError, without warnings or
    changes to either input.
    """

    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.shape[0] < 1
        or rgb.shape[1] < 1
        or rgb.dtype != np.uint8
    ):
        raise ImageReviewError("rgb must be a non-empty RGB uint8 array of shape (height, width, 3).")
    if not isinstance(meshes, Sequence) or isinstance(meshes, (str, bytes)):
        raise ImageReviewError("meshes must be a sequence of DTMesh objects.")
    validated: list[np.ndarray] = []
    labels: set[int] = set()
    for index, mesh in enumerate(meshes):
        if not isinstance(mesh, DTMesh):
            raise ImageReviewError(f"meshes[{index}] must be a DTMesh.")
        label = _cell_label(mesh.cell_index, f"meshes[{index}]")
        if label in labels:
            raise ImageReviewError(f"Duplicate cell label {label} in meshes.")
        labels.add(label)
        validated.append(_validated_ribs(mesh.ribs, f"Cell label {label} DT mesh"))

    result = rgb.copy()
    for ribs in validated:
        for rib in ribs:
            _draw_segment(result, rib[:2], rib[2:], DT_RIB_COLOR)
    for ribs in validated:
        # Halve before adding to avoid overflow for large finite endpoints.
        midpoints = ribs[:, :2] * 0.5 + ribs[:, 2:] * 0.5
        if len(midpoints) == 1:
            _draw_segment(result, midpoints[0], midpoints[0], DT_SPINE_COLOR)
        else:
            for start, end in zip(midpoints[:-1], midpoints[1:]):
                _draw_segment(result, start, end, DT_SPINE_COLOR)
    return result
