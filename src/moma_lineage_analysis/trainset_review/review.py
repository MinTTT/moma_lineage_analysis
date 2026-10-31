"""Pure, explicit label edits for the training-review application.

Opening or plotting a trajectory never calls the resolver. Every function returns
copies, and observation IDs are integers rather than row positions or floats.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable
import re

import numpy as np
import pandas as pd

LABELS = ("division_point", "excluded_point")
ACTIONS = ("add_division", "remove_division", "exclude", "restore", "clear")


def observation_id(value: Any, *, from_browser: bool = False) -> int:
    """Parse an opaque integer without a float conversion (including >2**53)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Observation IDs must be integers, not booleans")
    if isinstance(value, Integral):
        return int(value)
    if from_browser and isinstance(value, str) and re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError("Observation IDs must be exact integers; browser IDs may be decimal strings")


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy labels, measurements, and per-trajectory observation IDs.

    Actual bool/np.bool_ values are required, including when columns have object
    dtype. Numeric 0/1 or string booleans are deliberately not accepted as edits.
    The same FOV observation ID may occur in different trajectories.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = {"trajectory_key", "fov_cell_index", "time_h", "spine_length", *LABELS}
    if not frame.columns.is_unique:
        raise ValueError("frame has duplicate column names")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("frame is empty")
    if any(not isinstance(key, str) or not key.strip() for key in frame.trajectory_key):
        raise ValueError("trajectory_key values must be nonempty strings")
    ids = [observation_id(value) for value in frame.fov_cell_index]
    if frame.assign(fov_cell_index=ids).duplicated(["trajectory_key", "fov_cell_index"]).any():
        raise ValueError("Duplicate observation IDs within a trajectory")
    for column in LABELS:
        if not all(isinstance(value, (bool, np.bool_)) for value in frame[column]):
            raise ValueError(f"{column} must contain true boolean values without missing entries")
    for column in ("time_h", "spine_length"):
        if any(isinstance(value, (bool, np.bool_)) for value in frame[column]):
            raise ValueError(f"{column} must contain numeric measurements, not booleans")
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be finite numeric measurements") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite")
        if column == "spine_length" and (values <= 0).any():
            raise ValueError("spine_length must be positive")
    return frame.copy(deep=True)


def table_labels(frame: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    """Apply an edited label table as a label-only copy.

    The browser table sends decimal observation IDs, so IDs are compared as
    exact integers in row order. Measurements never come from the table and
    every label must be a true boolean. Raises before any write on mismatch.
    """
    result = validate_frame(frame)
    if not isinstance(edited, pd.DataFrame):
        raise TypeError("edited table must be a pandas DataFrame")
    required = {"fov_cell_index", *LABELS}
    missing = required.difference(edited.columns)
    if missing:
        raise ValueError(f"Edited table is missing columns: {sorted(missing)}")
    if len(edited) != len(result):
        raise ValueError("Edited table changed the number of observations")
    expected = [observation_id(value) for value in result.fov_cell_index]
    actual = [observation_id(value, from_browser=True) for value in edited.fov_cell_index]
    if actual != expected:
        raise ValueError("Edited table changed observation IDs or their order")
    for label in LABELS:
        values = edited[label].tolist()
        if not all(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"{label} must contain true boolean values without missing entries")
    changed = result.copy(deep=True)
    for label in LABELS:
        changed[label] = np.asarray(edited[label].tolist(), dtype=bool)
    return changed


def edit_labels(frame: pd.DataFrame, ids: Iterable[int], action: str) -> pd.DataFrame:
    """Edit selected observations on a copy; no implicit conflict resolution.

    add_division/remove_division change only division_point; exclude/restore
    change only excluded_point; clear clears both labels. IDs must identify a
    unique row across the supplied frame (normally one displayed trajectory).
    """
    result = validate_frame(frame)
    if action not in ACTIONS:
        raise ValueError(f"Unknown label action: {action!r}; expected one of {ACTIONS}")
    allowed = [observation_id(value) for value in result.fov_cell_index]
    if len(set(allowed)) != len(allowed):
        raise ValueError("Observation IDs are ambiguous across trajectories; edit one trajectory at a time")
    selected = {observation_id(value) for value in ids}
    unknown = selected.difference(allowed)
    if unknown:
        raise ValueError(f"Unknown observation IDs: {sorted(unknown)}")
    mask = np.asarray([value in selected for value in allowed], dtype=bool)
    if action in ("add_division", "remove_division", "clear"):
        result.loc[mask, "division_point"] = action == "add_division"
    if action in ("exclude", "restore", "clear"):
        result.loc[mask, "excluded_point"] = action == "exclude"
    return result


def resolve_labels(frame: pd.DataFrame) -> dict[str, Any]:
    """Explicitly apply the packaged effective-division policy per trajectory.

    Exclusion flags and measurements remain unchanged. Temporary string tokens
    protect large IDs from pandas coercion inside the existing audit resolver.
    decisions includes trajectory_key and original opaque integer IDs.
    """
    from moma_lineage_analysis.trajectory_cleaning.streamlit_api import (
        DIVISION_RESOLUTION_STATUSES, resolve_division_exclusion_conflicts,
    )
    result = validate_frame(frame)
    effective = np.zeros(len(result), dtype=bool)
    decisions = []
    counts = {status: 0 for status in DIVISION_RESOLUTION_STATUSES}
    for key, positions in result.groupby("trajectory_key", sort=False).indices.items():
        group = result.iloc[positions].copy()
        original_ids = [observation_id(value) for value in group.fov_cell_index]
        tokens = [f"observation_{index}" for index in range(len(group))]
        mapping = dict(zip(tokens, original_ids))
        group["fov_cell_index"] = tokens
        divisions = [token for token, flag in zip(tokens, group.division_point) if flag]
        exclusions = [token for token, flag in zip(tokens, group.excluded_point) if flag]
        resolution = resolve_division_exclusion_conflicts(group, divisions, exclusions)
        selected = set(resolution.effective_division_indices)
        effective[positions] = [token in selected for token in tokens]
        audit = resolution.decisions.copy(deep=True)
        for column in ("source_index", "effective_index", "previous_valid_index", "next_valid_index"):
            audit[column] = pd.Series([None if pd.isna(value) else mapping[value]
                                       for value in audit[column]], index=audit.index, dtype=object)
        audit.insert(0, "trajectory_key", key)
        decisions.append(audit)
        for status, count in resolution.summary_counts.items():
            counts[status] += count
    original = result.division_point.to_numpy(dtype=bool)
    removed = int(np.sum(original & ~effective))
    added = int(np.sum(~original & effective))
    result["division_point"] = effective
    return {"frame": result, "decisions": pd.concat(decisions, ignore_index=True),
            "status_counts": counts, "removed": removed, "added": added}
