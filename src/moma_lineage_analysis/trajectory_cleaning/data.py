"""Data validation, normalization, grouping, and padding for trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


LABEL_COLUMNS = ("division_point", "excluded_point")

# Per-observation model input channels, in tensor column order. ``time`` and
# ``length`` are the v0.1.0 channels; ``length_delta``/``log_growth_rate`` are
# the boundary-safe physiological features added in round R002 to keep the
# division signature stationary with respect to absolute experiment time, and
# ``time_since_last_drop`` encodes cell-cycle phase without absolute time.
FEATURE_COLUMNS = (
    "time",
    "length",
    "length_delta",
    "log_growth_rate",
    "time_since_last_drop",
)


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    """Parse common CSV boolean encodings without treating strings as truthy."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        invalid = values.notna() & ~values.isin([0, 1])
        if invalid.any():
            examples = series.loc[invalid].astype(str).unique()[:3].tolist()
            raise ValueError(f"{column!r} contains non-boolean values: {examples}")
        if values.isna().any():
            raise ValueError(f"{column!r} contains missing values")
        return values.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        examples = series.loc[parsed.isna()].astype(str).unique()[:3].tolist()
        raise ValueError(f"{column!r} contains non-boolean values: {examples}")
    return parsed.astype(bool)


def validate_trajectory_dataframe(
    dataframe: pd.DataFrame,
    *,
    trajectory_column: str = "trajectory_key",
    time_column: str = "time_h",
    length_column: str = "spine_length",
    require_labels: bool = True,
) -> pd.DataFrame:
    """Return a validated copy of a trajectory dataframe.

    Rows are not reordered here. Sorting happens independently inside each
    trajectory so inference output can retain the input CSV's original order.
    """

    required = {trajectory_column, time_column, length_column}
    if require_labels:
        required.update(LABEL_COLUMNS)
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = dataframe.copy()
    if result.empty:
        raise ValueError("The trajectory dataframe is empty")
    if result[trajectory_column].isna().any():
        raise ValueError(f"{trajectory_column!r} contains missing values")

    for column in (time_column, length_column):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column!r} must contain only finite numeric values")

    for column in LABEL_COLUMNS:
        if column in result:
            result[column] = _as_bool(result[column], column)
        elif not require_labels:
            result[column] = False

    return result


def load_trajectory_csv(
    path: str | Path,
    *,
    trajectory_column: str = "trajectory_key",
    time_column: str = "time_h",
    length_column: str = "spine_length",
    require_labels: bool = True,
) -> pd.DataFrame:
    """Load and validate a labeled or unlabeled trajectory CSV."""

    dataframe = pd.read_csv(Path(path))
    return validate_trajectory_dataframe(
        dataframe,
        trajectory_column=trajectory_column,
        time_column=time_column,
        length_column=length_column,
        require_labels=require_labels,
    )


@dataclass(frozen=True)
class TrajectoryNormalizer:
    """Unit- and scale-invariant normalization performed per trajectory.

    Time is converted to elapsed median sampling intervals and then divided by
    a fixed numerical scale. Therefore hours, seconds, and a constant rescaling
    of either produce the same feature values while skipped-frame gaps remain
    visible. Spine length
    is converted to a log ratio relative to the trajectory median and divided
    by a robust MAD scale. This is invariant to multiplicative changes such as
    micrometres versus pixels or larger cells in a different growth condition.

    The full trajectory is available in the intended offline cleaning workflow.
    A future online/causal workflow should estimate these quantities only from
    past observations.
    """

    epsilon: float = 1e-6
    mad_consistency: float = 1.4826
    clip_value: float = 12.0
    time_step_scale: float = 100.0
    minimum_length_scale: float = 0.05
    # ``legacy`` reproduces the v0.1.0 two-channel (time, length) layout so
    # checkpoints trained before round R002 keep their original behavior.
    feature_compatibility: str = "extended"

    def __post_init__(self) -> None:
        if self.feature_compatibility not in {"extended", "legacy"}:
            raise ValueError("feature_compatibility must be 'extended' or 'legacy'")

    def transform(self, time: Sequence[float], spine_length: Sequence[float]) -> np.ndarray:
        time_array = np.asarray(time, dtype=np.float64)
        length_array = np.asarray(spine_length, dtype=np.float64)
        if time_array.ndim != 1 or length_array.ndim != 1:
            raise ValueError("time and spine_length must be one-dimensional")
        if len(time_array) != len(length_array) or len(time_array) == 0:
            raise ValueError("time and spine_length must be non-empty and equally sized")
        if not np.isfinite(time_array).all() or not np.isfinite(length_array).all():
            raise ValueError("time and spine_length must contain only finite values")
        if len(time_array) < 2:
            raise ValueError("Each trajectory must contain at least two time points")
        if np.any(np.diff(time_array) <= 0):
            raise ValueError("time must be strictly increasing within each trajectory")
        if self.time_step_scale <= 0:
            raise ValueError("time_step_scale must be positive")

        elapsed = time_array - time_array[0]
        positive_dt = np.diff(time_array)
        positive_dt = positive_dt[positive_dt > 0]
        sampling_interval = float(np.median(positive_dt)) if positive_dt.size else 1.0
        frame_time = elapsed / max(sampling_interval, self.epsilon)
        time_feature = frame_time / self.time_step_scale

        positive = length_array > 0
        if not positive.all():
            warnings.warn(
                "Nonpositive spine lengths were mapped to a finite floor; verify their "
                "excluded_point labels.",
                RuntimeWarning,
                stacklevel=2,
            )
        if positive.any():
            median_length = max(float(np.median(length_array[positive])), self.epsilon)
            length_ratio = length_array / median_length
            safe_ratio = np.where(positive, np.maximum(length_ratio, self.epsilon), self.epsilon)
            log_ratio = np.log(safe_ratio)
            positive_log = log_ratio[positive]
            center = float(np.median(positive_log))
            mad = float(np.median(np.abs(positive_log - center)))
            robust_scale = self.mad_consistency * mad
            if robust_scale <= self.epsilon:
                q25, q75 = np.quantile(positive_log, [0.25, 0.75])
                robust_scale = float((q75 - q25) / 1.349)
            if robust_scale <= self.epsilon:
                robust_scale = float(np.std(positive_log))
            robust_scale = max(robust_scale, self.minimum_length_scale)
            length_feature = (log_ratio - center) / robust_scale
        else:
            length_feature = np.full_like(length_array, -self.clip_value)

        # Round-R002 boundary-safe dynamics features. They are computed from
        # the raw, unclipped length feature so the division signature stays
        # stationary with respect to absolute experiment time.
        raw_length_feature = length_feature
        length_delta = np.diff(raw_length_feature, prepend=raw_length_feature[0])
        growth_rate = np.diff(raw_length_feature, prepend=raw_length_feature[0]) / np.maximum(
            np.diff(frame_time, prepend=frame_time[0] + 1.0), 1.0
        )
        # Time (in median sampling intervals) since the previous strong length
        # drop — a boundary-safe proxy for cell-cycle phase.
        drop_indices = np.flatnonzero(length_delta <= -1.0)
        positions = np.arange(len(length_array), dtype=np.float64)
        if drop_indices.size:
            previous_drop = np.searchsorted(drop_indices, positions, side="right") - 1
            safe_previous = np.clip(previous_drop, 0, None)
            last_drop = np.where(previous_drop >= 0, drop_indices[safe_previous], -1.0)
        else:
            last_drop = np.full(len(length_array), -1.0)
        time_since_last_drop = np.clip(
            (positions - last_drop) / self.time_step_scale, 0.0, self.clip_value
        )

        feature_columns = [
            np.clip(time_feature, -self.clip_value, self.clip_value),
            np.clip(length_feature, -self.clip_value, self.clip_value),
        ]
        if self.feature_compatibility != "legacy":
            feature_columns.extend(
                [
                    np.clip(length_delta, -self.clip_value, self.clip_value),
                    np.clip(growth_rate, -self.clip_value, self.clip_value),
                    np.clip(time_since_last_drop, -self.clip_value, self.clip_value),
                ]
            )
        features = np.column_stack(feature_columns)
        return features.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrajectoryNormalizer":
        restored = dict(values)
        # Checkpoints saved before round R002 predate ``feature_compatibility``
        # and two-channel models; restore them in ``legacy`` mode.
        restored.setdefault("feature_compatibility", "legacy")
        return cls(**restored)


@dataclass
class TrajectoryRecord:
    key: Any
    features: torch.Tensor
    targets: torch.Tensor
    row_positions: np.ndarray

    @property
    def length(self) -> int:
        return int(self.features.shape[0])


class TrajectoryDataset(Dataset[TrajectoryRecord]):
    """One dataset item per ``trajectory_key`` rather than per CSV row."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        keys: Iterable[Any] | None = None,
        normalizer: TrajectoryNormalizer | None = None,
        trajectory_column: str = "trajectory_key",
        time_column: str = "time_h",
        length_column: str = "spine_length",
    ) -> None:
        self.trajectory_column = trajectory_column
        self.time_column = time_column
        self.length_column = length_column
        self.normalizer = normalizer or TrajectoryNormalizer()

        validated = validate_trajectory_dataframe(
            dataframe,
            trajectory_column=trajectory_column,
            time_column=time_column,
            length_column=length_column,
            require_labels=False,
        )
        validated = validated.copy()
        validated["__row_position__"] = np.arange(len(validated), dtype=np.int64)

        if keys is not None:
            key_set = set(keys)
            validated = validated[validated[trajectory_column].isin(key_set)]
            missing_keys = key_set.difference(validated[trajectory_column].unique().tolist())
            if missing_keys:
                raise ValueError(f"Unknown trajectory keys: {list(missing_keys)[:3]}")

        self.records: list[TrajectoryRecord] = []
        for key, group in validated.groupby(trajectory_column, sort=False, dropna=False):
            ordered = group.sort_values(
                [time_column, "__row_position__"], kind="mergesort"
            )
            features = self.normalizer.transform(
                ordered[time_column].to_numpy(),
                ordered[length_column].to_numpy(),
            )
            targets = ordered.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float32)
            self.records.append(
                TrajectoryRecord(
                    key=key,
                    features=torch.from_numpy(features),
                    targets=torch.from_numpy(targets),
                    row_positions=ordered["__row_position__"].to_numpy(dtype=np.int64),
                )
            )
        if not self.records:
            raise ValueError("No trajectories remain after key filtering")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrajectoryRecord:
        return self.records[index]


def collate_trajectories(batch: Sequence[TrajectoryRecord]) -> dict[str, Any]:
    """Pad variable-length trajectories and return a validity mask."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    max_length = max(record.length for record in batch)
    feature_count = int(batch[0].features.shape[1])
    target_count = int(batch[0].targets.shape[1])
    features = torch.zeros(batch_size, max_length, feature_count, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_length, target_count, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for index, record in enumerate(batch):
        length = record.length
        features[index, :length] = record.features
        targets[index, :length] = record.targets
        mask[index, :length] = True

    return {
        "features": features,
        "targets": targets,
        "mask": mask,
        "lengths": mask.sum(dim=1),
        "keys": [record.key for record in batch],
        "row_positions": [record.row_positions for record in batch],
    }


def _split_strata(
    dataframe: pd.DataFrame,
    keys: Sequence[Any],
    trajectory_column: str,
) -> np.ndarray | None:
    """Create coarse strata, falling back safely for small datasets."""

    summary = (
        dataframe[dataframe[trajectory_column].isin(keys)]
        .groupby(trajectory_column, sort=False)
        .agg(
            has_excluded=("excluded_point", "max"),
            division_count=("division_point", "sum"),
        )
        .reindex(keys)
    )
    division_bin = pd.cut(
        summary["division_count"],
        bins=[-np.inf, 4, 6, np.inf],
        labels=False,
    ).fillna(0).astype(int)
    detailed = summary["has_excluded"].astype(int).astype(str) + "_" + division_bin.astype(str)
    if detailed.value_counts().min() >= 2:
        return detailed.to_numpy()
    excluded_only = summary["has_excluded"].astype(int)
    if excluded_only.nunique() > 1 and excluded_only.value_counts().min() >= 2:
        return excluded_only.to_numpy()
    return None


def _safe_train_test_split(
    keys: Sequence[Any],
    *,
    test_fraction: float,
    seed: int,
    dataframe: pd.DataFrame,
    trajectory_column: str,
) -> tuple[list[Any], list[Any]]:
    if test_fraction <= 0:
        return list(keys), []
    if len(keys) < 2:
        raise ValueError("At least two trajectories are required for a split")

    strata = _split_strata(dataframe, keys, trajectory_column)
    n_test = max(1, int(round(len(keys) * test_fraction)))
    n_test = min(n_test, len(keys) - 1)
    if strata is not None:
        n_classes = len(np.unique(strata))
        if n_test < n_classes or (len(keys) - n_test) < n_classes:
            strata = None
    train_keys, test_keys = train_test_split(
        list(keys),
        test_size=n_test,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    return list(train_keys), list(test_keys)


def split_trajectory_keys(
    dataframe: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    trajectory_column: str = "trajectory_key",
    source_column: str | None = None,
) -> dict[str, list[Any]]:
    """Split whole trajectories, optionally within every acquisition source."""

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    required = {trajectory_column, *LABEL_COLUMNS}
    if source_column is not None:
        required.add(source_column)
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keys = dataframe[trajectory_column].drop_duplicates().tolist()
    relative_validation = validation_fraction / max(
        1.0 - test_fraction, np.finfo(float).eps
    )
    if source_column is not None:
        if dataframe[source_column].isna().any():
            raise ValueError(f"{source_column!r} contains missing values")
        source_counts = dataframe.groupby(
            trajectory_column, sort=False, dropna=False
        )[source_column].nunique(dropna=False)
        ambiguous = source_counts[source_counts != 1]
        if not ambiguous.empty:
            examples = ambiguous.index.astype(str).tolist()[:5]
            raise ValueError(
                "Every trajectory must belong to exactly one source; "
                f"ambiguous trajectory keys include {examples}"
            )

        split_by_source: dict[str, list[Any]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        source_values = dataframe[source_column].drop_duplicates().tolist()
        source_values.sort(key=str)
        required_partitions = (
            1 + int(validation_fraction > 0) + int(test_fraction > 0)
        )
        for source_index, source_value in enumerate(source_values):
            source_keys = dataframe.loc[
                dataframe[source_column].eq(source_value), trajectory_column
            ].drop_duplicates().tolist()
            if len(source_keys) < required_partitions:
                raise ValueError(
                    f"Source {source_value!r} has {len(source_keys)} trajectories; "
                    f"at least {required_partitions} are required for the requested "
                    "train/validation/test split"
                )
            source_seed = seed + source_index * 1009
            train_validation, source_test = _safe_train_test_split(
                source_keys,
                test_fraction=test_fraction,
                seed=source_seed,
                dataframe=dataframe,
                trajectory_column=trajectory_column,
            )
            source_train, source_validation = _safe_train_test_split(
                train_validation,
                test_fraction=relative_validation,
                seed=source_seed + 1,
                dataframe=dataframe,
                trajectory_column=trajectory_column,
            )
            split_by_source["train"].extend(source_train)
            split_by_source["validation"].extend(source_validation)
            split_by_source["test"].extend(source_test)
        return split_by_source

    train_validation, test = _safe_train_test_split(
        keys,
        test_fraction=test_fraction,
        seed=seed,
        dataframe=dataframe,
        trajectory_column=trajectory_column,
    )
    train, validation = _safe_train_test_split(
        train_validation,
        test_fraction=relative_validation,
        seed=seed + 1,
        dataframe=dataframe,
        trajectory_column=trajectory_column,
    )
    return {"train": train, "validation": validation, "test": test}
