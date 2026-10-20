"""Pure-Python helpers for integrating trajectory cleaning into a UI.

Despite the module name, this file deliberately does not import Streamlit.  It
translates between the lineage applications' ``trajectory_dict`` structure and
the dataframe-oriented inference API, and it keeps UI state mutation small and
auditable.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

from .training import predict_dataframe


TRAJECTORY_KEY_COLUMN = "trajectory_key"
FOV_INDEX_COLUMN = "fov_cell_index"
REQUIRED_TRAJECTORY_COLUMNS = frozenset(
    {"time_h", "spine_length", FOV_INDEX_COLUMN}
)
PREDICTION_COLUMNS = (
    "pred_division_probability",
    "pred_excluded_probability",
    "pred_division_point",
    "pred_excluded_point",
)
AI_INPUT_VALID_COLUMN = "ai_input_valid"
AI_INPUT_REASON_COLUMN = "ai_input_invalid_reason"
AI_PREDICTION_SOURCE_COLUMN = "ai_prediction_source"
AI_PREDICTION_AUDIT_COLUMNS = (
    AI_INPUT_VALID_COLUMN,
    AI_INPUT_REASON_COLUMN,
    AI_PREDICTION_SOURCE_COLUMN,
)
DIVISION_RESOLUTION_STATUSES = (
    "unchanged",
    "coalesced_previous",
    "shifted_previous",
    "needs_review",
)

CleaningSnapshot: TypeAlias = dict[Hashable, dict[str, set[Hashable]]]


@dataclass(frozen=True)
class DivisionConflictResolution:
    """Non-mutating result of resolving division/exclusion label conflicts.

    ``raw_*`` fields are immutable copies of the labels supplied by the caller.
    ``effective_division_indices`` contains only division boundaries that can be
    used after exclusions.  ``decisions`` has one row per raw division label and
    records how (or whether) it contributed to that effective set.
    """

    raw_division_indices: frozenset[Hashable]
    raw_excluded_indices: frozenset[Hashable]
    effective_division_indices: tuple[Hashable, ...]
    decisions: pd.DataFrame

    @property
    def summary_counts(self) -> dict[str, int]:
        """Return stable per-status counts, including statuses absent in the data."""

        counts = self.decisions["division_resolution"].value_counts()
        return {
            status: int(counts.get(status, 0))
            for status in DIVISION_RESOLUTION_STATUSES
        }


@dataclass(frozen=True)
class AIInputPreflight:
    """Validated model input plus a row-level audit of omitted observations.

    Invalid observations are deliberately retained in ``decisions`` and are
    identified by their original ``fov_cell_index``.  ``valid_trajectory`` is
    the only frame that is safe to pass to the sequence model.
    """

    valid_trajectory: pd.DataFrame
    invalid_indices: frozenset[Hashable]
    decisions: pd.DataFrame

    @property
    def valid_count(self) -> int:
        return int(len(self.valid_trajectory))

    @property
    def invalid_count(self) -> int:
        return int(len(self.invalid_indices))

    @property
    def inference_ready(self) -> bool:
        return self.valid_count >= 2


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _materialize_keys(keys: Iterable[Hashable] | Hashable) -> list[Hashable]:
    if isinstance(keys, (str, bytes)):
        values: list[Hashable] = [keys]
    else:
        try:
            values = list(keys)  # type: ignore[arg-type]
        except TypeError:
            values = [keys]  # type: ignore[list-item]

    if not values:
        raise ValueError("At least one trajectory key is required")

    seen: set[Hashable] = set()
    for key in values:
        if _is_missing_scalar(key):
            raise ValueError("Trajectory keys cannot be missing")
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError(f"Trajectory key {key!r} is not hashable") from exc
        if key in seen:
            raise ValueError(f"Duplicate trajectory key requested: {key!r}")
        seen.add(key)
    return values


def _trajectory_entry(
    trajectory_dict: Mapping[Hashable, Any],
    key: Hashable,
    *,
    mutable: bool = False,
) -> tuple[Mapping[str, Any] | MutableMapping[str, Any], pd.DataFrame]:
    if not isinstance(trajectory_dict, Mapping):
        raise TypeError("trajectory_dict must be a mapping")
    if key not in trajectory_dict:
        raise KeyError(f"Trajectory key not found: {key!r}")

    entry = trajectory_dict[key]
    expected = MutableMapping if mutable else Mapping
    if not isinstance(entry, expected):
        qualifier = "mutable " if mutable else ""
        raise TypeError(f"Trajectory entry {key!r} must be a {qualifier}mapping")
    if "trajectory" not in entry:
        raise KeyError(f"Trajectory entry {key!r} has no 'trajectory' dataframe")

    frame = entry["trajectory"]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Trajectory entry {key!r}['trajectory'] must be a DataFrame")
    if frame.empty:
        raise ValueError(f"Trajectory {key!r} is empty")

    missing_columns = sorted(REQUIRED_TRAJECTORY_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Trajectory {key!r} is missing required columns: {missing_columns}"
        )
    _validate_fov_indices(frame, context=f"Trajectory {key!r}")
    return entry, frame


def _validate_fov_indices(frame: pd.DataFrame, *, context: str) -> set[Hashable]:
    if FOV_INDEX_COLUMN not in frame.columns:
        raise ValueError(f"{context} is missing required column {FOV_INDEX_COLUMN!r}")

    series = frame[FOV_INDEX_COLUMN]
    if series.isna().any():
        raise ValueError(f"{context} contains missing {FOV_INDEX_COLUMN} values")
    if series.duplicated().any():
        duplicates = series.loc[series.duplicated(keep=False)].unique()[:3].tolist()
        raise ValueError(
            f"{context} contains duplicate {FOV_INDEX_COLUMN} values: {duplicates}"
        )

    values: set[Hashable] = set()
    for value in series.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"{context} contains an unhashable {FOV_INDEX_COLUMN}: {value!r}"
            ) from exc
        values.add(value)
    return values


def _validate_threshold(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a numeric probability, not a boolean")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return threshold


def _validate_threshold_pair(thresholds: Iterable[float]) -> tuple[float, float]:
    try:
        values = tuple(thresholds)
    except TypeError as exc:
        raise TypeError("checkpoint_thresholds must contain two probabilities") from exc
    if len(values) != 2:
        raise ValueError("checkpoint_thresholds must contain exactly two probabilities")
    return (
        _validate_threshold(values[0], "division threshold"),
        _validate_threshold(values[1], "excluded threshold"),
    )


def _validate_prediction_probabilities(prediction_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError("prediction_df must be a DataFrame")
    required = {
        FOV_INDEX_COLUMN,
        "pred_division_probability",
        "pred_excluded_probability",
    }
    missing = sorted(required.difference(prediction_df.columns))
    if missing:
        raise ValueError(f"Prediction dataframe is missing required columns: {missing}")
    if prediction_df.empty:
        raise ValueError("Prediction dataframe is empty")

    _validate_fov_indices(prediction_df, context="Prediction dataframe")
    validated = prediction_df.copy()
    for column in ("pred_division_probability", "pred_excluded_probability"):
        values = pd.to_numeric(validated[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{column!r} must contain only finite numeric values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{column!r} must contain probabilities between 0 and 1")
        validated[column] = values
    return validated


def _coerce_index_set(
    value: Any,
    *,
    field: str,
    key: Hashable,
    allowed: set[Hashable],
) -> set[Hashable]:
    if value is None:
        indices: set[Hashable] = set()
    elif isinstance(value, (str, bytes)):
        raise TypeError(f"{field} for trajectory {key!r} must be an iterable of IDs")
    else:
        try:
            indices = set(value)
        except TypeError as exc:
            raise TypeError(
                f"{field} for trajectory {key!r} must be an iterable of hashable IDs"
            ) from exc

    unknown = indices.difference(allowed)
    if unknown:
        preview = list(unknown)[:3]
        raise ValueError(
            f"{field} for trajectory {key!r} contains unknown "
            f"{FOV_INDEX_COLUMN} values: {preview}"
        )
    return indices


def _ordered_indices(frame: pd.DataFrame, selected: set[Hashable]) -> list[Hashable]:
    return [
        value for value in frame[FOV_INDEX_COLUMN].tolist() if value in selected
    ]


def preflight_ai_trajectory_inputs(trajectory_df: pd.DataFrame) -> AIInputPreflight:
    """Identify unusable model inputs without changing the source dataframe.

    Time offsets may be negative or zero, but must be finite.  Spine lengths
    must be finite and strictly positive.  When two otherwise usable rows have
    exactly the same timestamp, every row in that ambiguous timestamp group is
    excluded instead of arbitrarily choosing one observation.
    """

    if not isinstance(trajectory_df, pd.DataFrame):
        raise TypeError("trajectory_df must be a DataFrame")
    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty")
    missing = sorted(REQUIRED_TRAJECTORY_COLUMNS.difference(trajectory_df.columns))
    if missing:
        raise ValueError(f"Trajectory dataframe is missing required columns: {missing}")
    _validate_fov_indices(trajectory_df, context="Trajectory dataframe")

    time_values = pd.to_numeric(trajectory_df["time_h"], errors="coerce")
    length_values = pd.to_numeric(trajectory_df["spine_length"], errors="coerce")
    time_array = time_values.to_numpy(dtype=np.float64)
    length_array = length_values.to_numpy(dtype=np.float64)
    finite_time = np.isfinite(time_array)
    finite_length = np.isfinite(length_array)
    positive_length = finite_length & (length_array > 0.0)
    base_valid = finite_time & positive_length

    duplicate_time = np.zeros(len(trajectory_df), dtype=bool)
    valid_positions = np.flatnonzero(base_valid)
    duplicate_time[valid_positions] = pd.Series(
        time_array[valid_positions]
    ).duplicated(keep=False).to_numpy(dtype=bool)
    valid_mask = base_valid & ~duplicate_time

    reasons: list[str] = []
    for position in range(len(trajectory_df)):
        row_reasons: list[str] = []
        if not bool(finite_time[position]):
            row_reasons.append("nonfinite_time")
        if not bool(finite_length[position]):
            row_reasons.append("nonfinite_spine_length")
        elif not bool(positive_length[position]):
            row_reasons.append("nonpositive_spine_length")
        if bool(duplicate_time[position]):
            row_reasons.append("duplicate_time")
        reasons.append(";".join(row_reasons))

    decisions = pd.DataFrame(
        {
            FOV_INDEX_COLUMN: trajectory_df[FOV_INDEX_COLUMN].tolist(),
            AI_INPUT_VALID_COLUMN: valid_mask,
            AI_INPUT_REASON_COLUMN: reasons,
        }
    )
    invalid_indices = frozenset(
        decisions.loc[
            ~decisions[AI_INPUT_VALID_COLUMN], FOV_INDEX_COLUMN
        ].tolist()
    )
    valid_trajectory = trajectory_df.iloc[np.flatnonzero(valid_mask)].copy()
    valid_trajectory["time_h"] = time_array[valid_mask]
    valid_trajectory["spine_length"] = length_array[valid_mask]
    return AIInputPreflight(
        valid_trajectory=valid_trajectory,
        invalid_indices=invalid_indices,
        decisions=decisions,
    )


def _prediction_input_valid_mask(prediction: pd.DataFrame) -> pd.Series:
    """Return the model-scored row mask, validating an optional audit column."""

    if AI_INPUT_VALID_COLUMN not in prediction.columns:
        return pd.Series(True, index=prediction.index, dtype=bool)
    values = prediction[AI_INPUT_VALID_COLUMN]
    if values.isna().any():
        raise ValueError(f"{AI_INPUT_VALID_COLUMN!r} cannot contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"{AI_INPUT_VALID_COLUMN!r} must contain only booleans")
    return numeric.astype(bool)


def _validate_prediction_alignment(
    trajectory: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    key: Hashable,
) -> None:
    """Require prediction coordinates to match every model-scored source row."""

    identity_columns = {TRAJECTORY_KEY_COLUMN, "time_h", "spine_length"}
    missing = sorted(identity_columns.difference(prediction.columns))
    if missing:
        raise ValueError(
            f"Predictions for trajectory {key!r} are missing identity columns: {missing}"
        )

    predicted_keys = prediction[TRAJECTORY_KEY_COLUMN]
    if predicted_keys.isna().any() or not predicted_keys.eq(key).fillna(False).all():
        examples = (
            predicted_keys.loc[~predicted_keys.eq(key).fillna(False)]
            .head(3)
            .tolist()
        )
        raise ValueError(
            f"Predictions supplied for trajectory {key!r} contain a different "
            f"trajectory_key: {examples}"
        )

    if TRAJECTORY_KEY_COLUMN in trajectory.columns:
        trajectory_keys = trajectory[TRAJECTORY_KEY_COLUMN]
        if trajectory_keys.isna().any() or not trajectory_keys.eq(key).fillna(False).all():
            raise ValueError(
                f"Trajectory {key!r} contains conflicting trajectory_key values"
            )

    expected_preflight = preflight_ai_trajectory_inputs(trajectory)
    prediction_valid = _prediction_input_valid_mask(prediction)
    if AI_INPUT_VALID_COLUMN in prediction.columns:
        expected_valid = prediction[FOV_INDEX_COLUMN].map(
            expected_preflight.decisions.set_index(FOV_INDEX_COLUMN)[
                AI_INPUT_VALID_COLUMN
            ]
        )
        if expected_valid.isna().any() or not np.array_equal(
            prediction_valid.to_numpy(dtype=bool),
            expected_valid.to_numpy(dtype=bool),
        ):
            raise ValueError(
                f"Predictions for trajectory {key!r} contain stale or invalid "
                "AI input-validation flags"
            )
    elif expected_preflight.invalid_indices:
        raise ValueError(
            f"Predictions for trajectory {key!r} must identify invalid AI input rows"
        )

    lookup = prediction.set_index(FOV_INDEX_COLUMN)
    trajectory_lookup = trajectory.set_index(FOV_INDEX_COLUMN)
    scored_ids = prediction.loc[prediction_valid, FOV_INDEX_COLUMN].tolist()
    for coordinate in ("time_h", "spine_length"):
        trajectory_values = pd.to_numeric(
            trajectory_lookup.loc[scored_ids, coordinate], errors="coerce"
        ).to_numpy(dtype=np.float64)
        predicted_values = pd.to_numeric(
            lookup.loc[scored_ids, coordinate], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(trajectory_values).all()
            or not np.isfinite(predicted_values).all()
            or not np.allclose(
                trajectory_values,
                predicted_values,
                rtol=1e-7,
                atol=1e-9,
            )
        ):
            raise ValueError(
                f"Prediction {coordinate!r} values do not match trajectory {key!r} "
                f"rows identified by {FOV_INDEX_COLUMN!r}"
            )


def build_scope_dataframe(
    trajectory_dict: Mapping[Hashable, Any],
    keys: Iterable[Hashable] | Hashable,
) -> pd.DataFrame:
    """Concatenate requested trajectory dataframes for model inference.

    A pre-existing ``trajectory_key`` column is accepted only when every value
    agrees with the enclosing dictionary key.  This prevents a stale exported
    key from silently routing predictions to the wrong UI trajectory.
    """

    requested_keys = _materialize_keys(keys)
    frames: list[pd.DataFrame] = []
    for key in requested_keys:
        _, trajectory = _trajectory_entry(trajectory_dict, key)
        frame = trajectory.copy()
        if TRAJECTORY_KEY_COLUMN in frame.columns:
            existing = frame[TRAJECTORY_KEY_COLUMN]
            if existing.isna().any() or not existing.eq(key).fillna(False).all():
                examples = existing.loc[~existing.eq(key).fillna(False)].head(3).tolist()
                raise ValueError(
                    f"Trajectory {key!r} contains conflicting trajectory_key "
                    f"values: {examples}"
                )
        frame[TRAJECTORY_KEY_COLUMN] = pd.Series(
            [key] * len(frame), index=frame.index, dtype=object
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def predict_trajectory_scope(
    model: Any,
    normalizer: Any,
    checkpoint_thresholds: Iterable[float],
    trajectory_dict: Mapping[Hashable, Any],
    keys: Iterable[Hashable] | Hashable,
    device: Any = "cpu",
    batch_size: int = 32,
) -> dict[Hashable, pd.DataFrame]:
    """Predict a scope after omitting objectively invalid feature rows.

    The returned frames still cover every original cell ID.  Rows omitted from
    inference carry deterministic validation-rule labels (never a division,
    always excluded) and explicit audit columns so they cannot be mistaken for
    neural-network scores.
    """

    requested_keys = _materialize_keys(keys)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise TypeError("batch_size must be a positive integer")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    thresholds = _validate_threshold_pair(checkpoint_thresholds)
    preflights: dict[Hashable, AIInputPreflight] = {}
    model_trajectories: dict[Hashable, dict[str, pd.DataFrame]] = {}
    for key in requested_keys:
        _, trajectory = _trajectory_entry(trajectory_dict, key)
        if TRAJECTORY_KEY_COLUMN in trajectory.columns:
            existing = trajectory[TRAJECTORY_KEY_COLUMN]
            if existing.isna().any() or not existing.eq(key).fillna(False).all():
                examples = existing.loc[
                    ~existing.eq(key).fillna(False)
                ].head(3).tolist()
                raise ValueError(
                    f"Trajectory {key!r} contains conflicting trajectory_key "
                    f"values: {examples}"
                )
        preflight = preflight_ai_trajectory_inputs(trajectory)
        if not preflight.inference_ready:
            raise ValueError(
                f"Trajectory {key!r} has {preflight.valid_count} valid AI input "
                "point(s) after preflight; at least two are required"
            )
        preflights[key] = preflight
        model_trajectories[key] = {"trajectory": preflight.valid_trajectory}

    scope = build_scope_dataframe(model_trajectories, requested_keys)
    predicted = predict_dataframe(
        model,
        scope,
        normalizer=normalizer,
        thresholds=thresholds,
        trajectory_column=TRAJECTORY_KEY_COLUMN,
        time_column="time_h",
        length_column="spine_length",
        batch_size=int(batch_size),
        device=device,
    )

    # Retain coordinates needed by the Streamlit plot overlay as well as IDs.
    output_columns = [
        TRAJECTORY_KEY_COLUMN,
        FOV_INDEX_COLUMN,
        "time_h",
        "spine_length",
        *PREDICTION_COLUMNS,
    ]
    result: dict[Hashable, pd.DataFrame] = {}
    for key in requested_keys:
        mask = predicted[TRAJECTORY_KEY_COLUMN].eq(key).fillna(False)
        model_prediction = predicted.loc[mask, output_columns].reset_index(drop=True)
        _, trajectory = _trajectory_entry(trajectory_dict, key)
        preflight = preflights[key]
        if len(model_prediction) != preflight.valid_count:
            raise RuntimeError(
                f"Inference returned {len(model_prediction)} rows for trajectory "
                f"{key!r}; expected {preflight.valid_count} valid rows"
            )
        _validate_fov_indices(
            model_prediction, context=f"Predictions for trajectory {key!r}"
        )
        expected_model_ids = set(
            preflight.valid_trajectory[FOV_INDEX_COLUMN].tolist()
        )
        actual_model_ids = set(model_prediction[FOV_INDEX_COLUMN].tolist())
        if actual_model_ids != expected_model_ids:
            raise RuntimeError(
                f"Inference returned mismatched {FOV_INDEX_COLUMN} values for "
                f"trajectory {key!r}"
            )

        key_predictions = trajectory.loc[
            :, [FOV_INDEX_COLUMN, "time_h", "spine_length"]
        ].copy()
        key_predictions.insert(0, TRAJECTORY_KEY_COLUMN, key)
        model_lookup = model_prediction.set_index(FOV_INDEX_COLUMN)
        model_valid = key_predictions[FOV_INDEX_COLUMN].isin(expected_model_ids)
        for column in PREDICTION_COLUMNS:
            key_predictions[column] = key_predictions[FOV_INDEX_COLUMN].map(
                model_lookup[column]
            )

        # Validation-rule rows remain visible and exactly aligned, but they do
        # not claim neural-network confidence or create division boundaries.
        key_predictions.loc[
            ~model_valid, "pred_division_probability"
        ] = 0.0
        key_predictions.loc[
            ~model_valid, "pred_excluded_probability"
        ] = 1.0
        key_predictions.loc[~model_valid, "pred_division_point"] = False
        key_predictions.loc[~model_valid, "pred_excluded_point"] = True
        key_predictions["pred_division_point"] = key_predictions[
            "pred_division_point"
        ].astype(bool)
        key_predictions["pred_excluded_point"] = key_predictions[
            "pred_excluded_point"
        ].astype(bool)

        decision_lookup = preflight.decisions.set_index(FOV_INDEX_COLUMN)
        key_predictions[AI_INPUT_VALID_COLUMN] = key_predictions[
            FOV_INDEX_COLUMN
        ].map(decision_lookup[AI_INPUT_VALID_COLUMN]).astype(bool)
        key_predictions[AI_INPUT_REASON_COLUMN] = key_predictions[
            FOV_INDEX_COLUMN
        ].map(decision_lookup[AI_INPUT_REASON_COLUMN])
        key_predictions[AI_PREDICTION_SOURCE_COLUMN] = np.where(
            key_predictions[AI_INPUT_VALID_COLUMN],
            "model",
            "validation_rule",
        )
        result[key] = key_predictions.reset_index(drop=True)
    return result


def suggested_indices(
    prediction_df: pd.DataFrame,
    division_threshold: float,
    excluded_threshold: float,
) -> tuple[set[Hashable], set[Hashable]]:
    """Return ``(division_indices, excluded_indices)`` at custom thresholds."""

    division_cutoff = _validate_threshold(division_threshold, "division_threshold")
    excluded_cutoff = _validate_threshold(excluded_threshold, "excluded_threshold")
    prediction = _validate_prediction_probabilities(prediction_df)
    model_scored = _prediction_input_valid_mask(prediction)
    division = set(
        prediction.loc[
            model_scored
            & (prediction["pred_division_probability"] >= division_cutoff),
            FOV_INDEX_COLUMN,
        ].tolist()
    )
    excluded = set(
        prediction.loc[
            (~model_scored)
            | (prediction["pred_excluded_probability"] >= excluded_cutoff),
            FOV_INDEX_COLUMN,
        ].tolist()
    )
    return division, excluded


def resolve_division_exclusion_conflicts(
    trajectory_df: pd.DataFrame,
    division_indices: Iterable[Hashable] | None,
    excluded_indices: Iterable[Hashable] | None,
    *,
    division_drop_ratio: float = 0.75,
) -> DivisionConflictResolution:
    """Resolve division labels that coincide with excluded measurements.

    Raw labels are copied into the result and are never mutated.  Division
    sources are processed in stable ``time_h`` order.  A contiguous run of raw
    division sources represents at most one biological event:

    * the earliest non-excluded source in the run remains the boundary;
    * an all-excluded run coalesces onto an already labelled previous valid
      point; otherwise it shifts to that point only when the next/previous
      valid length ratio is strictly below ``division_drop_ratio``;
    * a run without either form of evidence remains ``needs_review``.

    The source-level audit table uses the exact statuses ``unchanged``,
    ``coalesced_previous``, ``shifted_previous``, and ``needs_review``.  A
    shifted run assigns ``shifted_previous`` to its first source and coalesces
    the remaining sources onto the same effective point. A raw division whose
    timestamp is nonfinite is retained as ``needs_review`` with no effective
    boundary because it cannot be placed in temporal order.
    """

    if not isinstance(trajectory_df, pd.DataFrame):
        raise TypeError("trajectory_df must be a DataFrame")
    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty")
    required = {FOV_INDEX_COLUMN, "time_h", "spine_length"}
    missing = sorted(required.difference(trajectory_df.columns))
    if missing:
        raise ValueError(
            f"Trajectory dataframe is missing required columns: {missing}"
        )

    allowed = _validate_fov_indices(
        trajectory_df, context="Trajectory dataframe"
    )
    raw_divisions = _coerce_index_set(
        division_indices,
        field="division_indices",
        key="conflict resolution",
        allowed=allowed,
    )
    raw_exclusions = _coerce_index_set(
        excluded_indices,
        field="excluded_indices",
        key="conflict resolution",
        allowed=allowed,
    )

    if isinstance(division_drop_ratio, (bool, np.bool_)):
        raise TypeError("division_drop_ratio must be numeric, not a boolean")
    try:
        drop_ratio = float(division_drop_ratio)
    except (TypeError, ValueError) as exc:
        raise TypeError("division_drop_ratio must be numeric") from exc
    if not np.isfinite(drop_ratio) or not 0.0 < drop_ratio <= 1.0:
        raise ValueError(
            "division_drop_ratio must be finite and greater than 0 and at most 1"
        )

    time_values = pd.to_numeric(
        trajectory_df["time_h"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    finite_time = np.isfinite(time_values)
    source_ids = trajectory_df[FOV_INDEX_COLUMN].tolist()
    invalid_time_ids = {
        source_ids[position]
        for position in np.flatnonzero(~finite_time).tolist()
    }
    length_values = pd.to_numeric(
        trajectory_df["spine_length"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    # Stable sorting makes "previous" temporal rather than dependent on cell ID
    # or an accidentally shuffled dataframe, while preserving tied time points.
    finite_positions = np.flatnonzero(finite_time)
    order = finite_positions[
        np.argsort(time_values[finite_positions], kind="stable")
    ]
    ordered_ids = trajectory_df.iloc[order][FOV_INDEX_COLUMN].tolist()
    ordered_lengths = length_values[order]
    position_for_id = {
        cell_id: position for position, cell_id in enumerate(ordered_ids)
    }

    ordered_raw_divisions = raw_divisions.difference(invalid_time_ids)
    division_positions = sorted(
        position_for_id[index] for index in ordered_raw_divisions
    )
    runs: list[list[int]] = []
    for position in division_positions:
        if runs and position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])

    effective: set[Hashable] = set()
    resolved_target_by_source: dict[Hashable, Hashable | None] = {}
    records: list[dict[str, Any]] = []
    for run in runs:
        sources = [ordered_ids[position] for position in run]
        non_excluded_positions = [
            position
            for position in run
            if ordered_ids[position] not in raw_exclusions
        ]

        previous_position = run[0] - 1
        while (
            previous_position >= 0
            and ordered_ids[previous_position] in raw_exclusions
        ):
            previous_position -= 1
        next_position = run[-1] + 1
        while (
            next_position < len(ordered_ids)
            and ordered_ids[next_position] in raw_exclusions
        ):
            next_position += 1

        previous_index = (
            ordered_ids[previous_position] if previous_position >= 0 else None
        )
        next_index = (
            ordered_ids[next_position]
            if next_position < len(ordered_ids)
            else None
        )
        length_ratio = np.nan
        if previous_index is not None and next_index is not None:
            previous_length = ordered_lengths[previous_position]
            next_length = ordered_lengths[next_position]
            if (
                np.isfinite(previous_length)
                and np.isfinite(next_length)
                and previous_length > 0.0
                and next_length > 0.0
            ):
                length_ratio = float(next_length / previous_length)

        if non_excluded_positions:
            target = ordered_ids[non_excluded_positions[0]]
            effective.add(target)
            statuses = [
                "unchanged" if source == target else "coalesced_previous"
                for source in sources
            ]
        else:
            target = None
            previous_event_target = (
                resolved_target_by_source.get(previous_index)
                if previous_index is not None
                else None
            )
            if previous_event_target is None and previous_index in effective:
                previous_event_target = previous_index
            drop_is_evidence = (
                previous_index is not None
                and next_index is not None
                and np.isfinite(length_ratio)
                and length_ratio < drop_ratio
            )
            if previous_event_target is not None:
                target = previous_event_target
                effective.add(target)
                statuses = ["coalesced_previous"] * len(sources)
            elif drop_is_evidence:
                target = previous_index
                effective.add(target)
                statuses = [
                    "shifted_previous",
                    *(["coalesced_previous"] * (len(sources) - 1)),
                ]
            else:
                statuses = ["needs_review"] * len(sources)

        for source, status in zip(sources, statuses, strict=True):
            resolved_target_by_source[source] = target
            records.append(
                {
                    "source_index": source,
                    "effective_index": target,
                    "division_resolution": status,
                    "previous_valid_index": previous_index,
                    "next_valid_index": next_index,
                    "length_ratio": length_ratio,
                }
            )

    # A row with no finite timestamp cannot be placed before or after another
    # biological observation. Preserve its raw division label for audit/manual
    # editing, but never invent an effective boundary for it. The Streamlit AI
    # preflight separately labels such a row as excluded before inference.
    for source in source_ids:
        if source in raw_divisions and source in invalid_time_ids:
            records.append(
                {
                    "source_index": source,
                    "effective_index": None,
                    "division_resolution": "needs_review",
                    "previous_valid_index": None,
                    "next_valid_index": None,
                    "length_ratio": np.nan,
                }
            )

    decision_columns = [
        "source_index",
        "effective_index",
        "division_resolution",
        "previous_valid_index",
        "next_valid_index",
        "length_ratio",
    ]
    decisions = pd.DataFrame.from_records(records, columns=decision_columns)
    # Keep opaque cell IDs opaque even when pandas could coerce them to floats
    # because some rows have no effective/neighbor ID.
    for column in (
        "source_index",
        "effective_index",
        "previous_valid_index",
        "next_valid_index",
    ):
        decisions[column] = decisions[column].astype(object)

    ordered_effective = tuple(
        cell_id for cell_id in ordered_ids if cell_id in effective
    )
    return DivisionConflictResolution(
        raw_division_indices=frozenset(raw_divisions),
        raw_excluded_indices=frozenset(raw_exclusions),
        effective_division_indices=ordered_effective,
        decisions=decisions,
    )


def capture_cleaning_snapshot(
    trajectory_dict: Mapping[Hashable, Any],
    keys: Iterable[Hashable] | Hashable,
) -> CleaningSnapshot:
    """Capture independent copies of the current division/exclusion ID sets."""

    snapshot: CleaningSnapshot = {}
    for key in _materialize_keys(keys):
        entry, trajectory = _trajectory_entry(trajectory_dict, key)
        allowed = _validate_fov_indices(trajectory, context=f"Trajectory {key!r}")
        snapshot[key] = {
            "div_indices": _coerce_index_set(
                entry.get("div_indices", set()),
                field="div_indices",
                key=key,
                allowed=allowed,
            ),
            "excluded_indices": _coerce_index_set(
                entry.get("excluded_indices", set()),
                field="excluded_indices",
                key=key,
                allowed=allowed,
            ),
        }
    return snapshot


def apply_ai_predictions(
    trajectory_dict: MutableMapping[Hashable, Any],
    predictions: Mapping[Hashable, pd.DataFrame],
    keys: Iterable[Hashable] | Hashable | None = None,
    *,
    apply_divisions: bool = True,
    apply_exclusions: bool = True,
    policy: Literal["merge", "replace"] = "merge",
    division_threshold: float = 0.5,
    excluded_threshold: float = 0.5,
) -> CleaningSnapshot:
    """Apply AI suggestions and return the exact pre-application snapshot.

    ``merge`` unions suggestions with current labels.  ``replace`` replaces
    only the enabled label types.  The function intentionally does not update
    cached segments, generation times, growth rates, or any other UI state.
    """

    if not isinstance(trajectory_dict, MutableMapping):
        raise TypeError("trajectory_dict must be a mutable mapping")
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a key-to-DataFrame mapping")
    if not isinstance(apply_divisions, (bool, np.bool_)):
        raise TypeError("apply_divisions must be boolean")
    if not isinstance(apply_exclusions, (bool, np.bool_)):
        raise TypeError("apply_exclusions must be boolean")
    if policy not in {"merge", "replace"}:
        raise ValueError("policy must be either 'merge' or 'replace'")

    requested_keys = _materialize_keys(predictions.keys() if keys is None else keys)
    division_cutoff = _validate_threshold(division_threshold, "division_threshold")
    excluded_cutoff = _validate_threshold(excluded_threshold, "excluded_threshold")
    snapshot = capture_cleaning_snapshot(trajectory_dict, requested_keys)

    # Validate and calculate every update before mutating any trajectory.
    updates: dict[Hashable, tuple[pd.DataFrame, set[Hashable], set[Hashable]]] = {}
    for key in requested_keys:
        if key not in predictions:
            raise KeyError(f"No prediction dataframe supplied for trajectory {key!r}")
        prediction = predictions[key]
        _, trajectory = _trajectory_entry(trajectory_dict, key, mutable=True)
        allowed = _validate_fov_indices(trajectory, context=f"Trajectory {key!r}")
        predicted_ids = _validate_fov_indices(
            prediction, context=f"Predictions for trajectory {key!r}"
        )
        if predicted_ids != allowed:
            missing = list(allowed.difference(predicted_ids))[:3]
            extra = list(predicted_ids.difference(allowed))[:3]
            raise ValueError(
                f"Predictions for trajectory {key!r} do not cover exactly its "
                f"{FOV_INDEX_COLUMN} values (missing={missing}, extra={extra})"
            )
        _validate_prediction_alignment(trajectory, prediction, key=key)
        proposed_divisions, proposed_exclusions = suggested_indices(
            prediction, division_cutoff, excluded_cutoff
        )

        current_divisions = snapshot[key]["div_indices"]
        current_exclusions = snapshot[key]["excluded_indices"]
        if policy == "merge":
            proposed_divisions = current_divisions | proposed_divisions
            proposed_exclusions = current_exclusions | proposed_exclusions
        updates[key] = (trajectory, proposed_divisions, proposed_exclusions)

    for key in requested_keys:
        entry, _ = _trajectory_entry(trajectory_dict, key, mutable=True)
        trajectory, proposed_divisions, proposed_exclusions = updates[key]
        if apply_divisions:
            entry["div_indices"] = _ordered_indices(trajectory, proposed_divisions)
        if apply_exclusions:
            entry["excluded_indices"] = set(proposed_exclusions)

    return snapshot


def restore_cleaning_snapshot(
    trajectory_dict: MutableMapping[Hashable, Any],
    snapshot: Mapping[Hashable, Mapping[str, Iterable[Hashable]]],
) -> None:
    """Restore a snapshot captured by :func:`capture_cleaning_snapshot`."""

    if not isinstance(trajectory_dict, MutableMapping):
        raise TypeError("trajectory_dict must be a mutable mapping")
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    requested_keys = _materialize_keys(snapshot.keys())

    restored: dict[Hashable, tuple[pd.DataFrame, set[Hashable], set[Hashable]]] = {}
    for key in requested_keys:
        state = snapshot[key]
        if not isinstance(state, Mapping):
            raise TypeError(f"Snapshot entry {key!r} must be a mapping")
        missing = {"div_indices", "excluded_indices"}.difference(state)
        if missing:
            raise ValueError(
                f"Snapshot entry {key!r} is missing fields: {sorted(missing)}"
            )
        _, trajectory = _trajectory_entry(trajectory_dict, key, mutable=True)
        allowed = _validate_fov_indices(trajectory, context=f"Trajectory {key!r}")
        divisions = _coerce_index_set(
            state["div_indices"], field="div_indices", key=key, allowed=allowed
        )
        exclusions = _coerce_index_set(
            state["excluded_indices"],
            field="excluded_indices",
            key=key,
            allowed=allowed,
        )
        restored[key] = (trajectory, divisions, exclusions)

    for key in requested_keys:
        entry, _ = _trajectory_entry(trajectory_dict, key, mutable=True)
        trajectory, divisions, exclusions = restored[key]
        entry["div_indices"] = _ordered_indices(trajectory, divisions)
        entry["excluded_indices"] = set(exclusions)


def attach_ai_predictions_to_export(
    export_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach AI predictions and constant provenance columns by cell ID.

    This helper is intended for one trajectory at a time, before the lineage
    apps concatenate trajectory exports.  Both dataframes must therefore have
    unique ``fov_cell_index`` values.  ``provenance`` maps exact output column
    names (for example ``ai_checkpoint`` or ``ai_division_threshold``) to a
    constant value that will be repeated for every exported row.
    """

    if not isinstance(export_df, pd.DataFrame):
        raise TypeError("export_df must be a DataFrame")
    if export_df.empty:
        raise ValueError("export_df is empty")
    export_ids = _validate_fov_indices(export_df, context="Export dataframe")
    prediction = _validate_prediction_probabilities(prediction_df)

    missing_prediction_columns = sorted(set(PREDICTION_COLUMNS).difference(prediction))
    if missing_prediction_columns:
        raise ValueError(
            "Prediction dataframe is missing required columns: "
            f"{missing_prediction_columns}"
        )
    prediction_ids = _validate_fov_indices(
        prediction, context="Prediction dataframe"
    )
    missing_ids = export_ids.difference(prediction_ids)
    extra_ids = prediction_ids.difference(export_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "Prediction and export dataframes must contain exactly the same "
            f"{FOV_INDEX_COLUMN} values (missing={list(missing_ids)[:3]}, "
            f"extra={list(extra_ids)[:3]})"
        )

    export_has_key = TRAJECTORY_KEY_COLUMN in export_df.columns
    prediction_has_key = TRAJECTORY_KEY_COLUMN in prediction.columns
    if export_has_key != prediction_has_key:
        raise ValueError(
            "trajectory_key must be present in both export and prediction "
            "dataframes when present in either"
        )
    if export_has_key:
        export_keys = export_df[TRAJECTORY_KEY_COLUMN].dropna().unique().tolist()
        prediction_keys = (
            prediction[TRAJECTORY_KEY_COLUMN].dropna().unique().tolist()
        )
        if (
            len(export_keys) != 1
            or len(prediction_keys) != 1
            or export_keys[0] != prediction_keys[0]
        ):
            raise ValueError(
                "Export and prediction dataframes must describe the same single "
                f"trajectory_key (export={export_keys}, prediction={prediction_keys})"
            )

    expected_preflight = preflight_ai_trajectory_inputs(export_df)
    prediction_valid = _prediction_input_valid_mask(prediction)
    if AI_INPUT_VALID_COLUMN in prediction.columns:
        expected_valid = prediction[FOV_INDEX_COLUMN].map(
            expected_preflight.decisions.set_index(FOV_INDEX_COLUMN)[
                AI_INPUT_VALID_COLUMN
            ]
        )
        if expected_valid.isna().any() or not np.array_equal(
            prediction_valid.to_numpy(dtype=bool),
            expected_valid.to_numpy(dtype=bool),
        ):
            raise ValueError(
                "Prediction AI input-validation flags do not match the export rows"
            )
    elif expected_preflight.invalid_indices:
        raise ValueError("Prediction must identify invalid AI input rows")

    lookup = prediction.set_index(FOV_INDEX_COLUMN)
    result = export_df.copy()
    scored_ids = prediction.loc[prediction_valid, FOV_INDEX_COLUMN].tolist()
    export_lookup = export_df.set_index(FOV_INDEX_COLUMN)
    for coordinate in ("time_h", "spine_length"):
        if coordinate in export_df.columns and coordinate in prediction.columns:
            export_values = pd.to_numeric(
                export_lookup.loc[scored_ids, coordinate], errors="coerce"
            ).to_numpy(dtype=np.float64)
            predicted_values = pd.to_numeric(
                lookup.loc[scored_ids, coordinate], errors="coerce"
            ).to_numpy(dtype=np.float64)
            if (
                not np.isfinite(export_values).all()
                or not np.isfinite(predicted_values).all()
                or not np.allclose(
                    export_values,
                    predicted_values,
                    rtol=1e-7,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    f"Prediction {coordinate!r} values do not match the export "
                    f"rows identified by {FOV_INDEX_COLUMN!r}"
                )
    for column in PREDICTION_COLUMNS:
        result[column] = result[FOV_INDEX_COLUMN].map(lookup[column]).to_numpy()
    for column in AI_PREDICTION_AUDIT_COLUMNS:
        if column in prediction.columns:
            result[column] = result[FOV_INDEX_COLUMN].map(lookup[column]).to_numpy()

    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping of column names to values")
        protected = {
            FOV_INDEX_COLUMN,
            *PREDICTION_COLUMNS,
            *AI_PREDICTION_AUDIT_COLUMNS,
        }
        for column, value in provenance.items():
            if not isinstance(column, str) or not column:
                raise ValueError("Provenance column names must be non-empty strings")
            if column in protected:
                raise ValueError(f"Provenance column {column!r} is reserved")
            result[column] = pd.Series(
                [deepcopy(value) for _ in range(len(result))],
                index=result.index,
            )
    return result


# A short alias for callers that describe the operation in terms of columns.
attach_ai_columns_to_export = attach_ai_predictions_to_export


__all__ = [
    "AIInputPreflight",
    "AI_INPUT_REASON_COLUMN",
    "AI_INPUT_VALID_COLUMN",
    "AI_PREDICTION_AUDIT_COLUMNS",
    "AI_PREDICTION_SOURCE_COLUMN",
    "CleaningSnapshot",
    "DIVISION_RESOLUTION_STATUSES",
    "DivisionConflictResolution",
    "FOV_INDEX_COLUMN",
    "PREDICTION_COLUMNS",
    "REQUIRED_TRAJECTORY_COLUMNS",
    "TRAJECTORY_KEY_COLUMN",
    "apply_ai_predictions",
    "attach_ai_columns_to_export",
    "attach_ai_predictions_to_export",
    "build_scope_dataframe",
    "capture_cleaning_snapshot",
    "predict_trajectory_scope",
    "preflight_ai_trajectory_inputs",
    "resolve_division_exclusion_conflicts",
    "restore_cleaning_snapshot",
    "suggested_indices",
]
