"""Read-only checkpoint evaluation and review-priority summaries."""
from __future__ import annotations

import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .review import LABELS, validate_frame

PROBABILITY_COLUMNS = ("pred_division_probability", "pred_excluded_probability")
PREDICTION_COLUMNS = ("pred_division_point", "pred_excluded_point")


def checkpoint_sha256(checkpoint_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_pair(thresholds: Iterable[float]) -> np.ndarray:
    values = list(thresholds)
    if len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise ValueError("Exactly two numeric thresholds (division, exclusion) are required")
    raw = np.asarray(values, dtype=float)
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Thresholds must be finite and between 0 and 1")
    # predict_dataframe uses float32 thresholds; use the identical thresholds
    # for evaluation, per-trajectory scoring, and displayed markers.
    return raw.astype(np.float32)


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    result = validate_frame(predictions)
    missing = set(PROBABILITY_COLUMNS).difference(result.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")
    for column in PROBABILITY_COLUMNS:
        try:
            values = result[column].to_numpy(dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{column} must contain finite probabilities in [0, 1]") from error
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must contain finite probabilities in [0, 1]")
    return result


def evaluate_frame(frame: pd.DataFrame, checkpoint_path: str | Path,
                   device: str = "cpu", thresholds: Iterable[float] | None = None) -> dict[str, Any]:
    """Evaluate existing labels; no calibration, relabeling, selection, or writes.

    predictions retains the exact input row order/index and opaque observation
    IDs, with four standard prediction columns. metrics is the existing
    training.evaluate_model output, including complete-trajectory event metrics.
    """
    from moma_lineage_analysis.trajectory_cleaning import training
    from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryDataset

    source = validate_frame(frame)
    override = None if thresholds is None else threshold_pair(thresholds)
    digest = checkpoint_sha256(checkpoint_path)
    selected_device = training.resolve_device(device)
    model, normalizer, stored, _ = training.load_checkpoint(checkpoint_path, device=selected_device)
    selected = threshold_pair(stored) if override is None else override
    predictions = training.predict_dataframe(model, source, normalizer=normalizer,
                                               thresholds=selected, device=selected_device)
    validate_predictions(predictions)
    if not predictions.index.equals(source.index) or len(predictions) != len(source):
        raise RuntimeError("Inference changed source row alignment")
    for column in ("trajectory_key", "fov_cell_index", "time_h", "spine_length", *LABELS):
        if not predictions[column].equals(source[column]):
            raise RuntimeError(f"Inference changed source column {column}")
    dataset = TrajectoryDataset(source, normalizer=normalizer)
    metrics = training.evaluate_model(model, dataset, thresholds=selected, device=selected_device,
                                      division_tolerance_frames=1)
    if checkpoint_sha256(checkpoint_path) != digest:
        raise RuntimeError("Checkpoint changed during evaluation; discard this result")
    return {"predictions": predictions, "metrics": metrics,
            "thresholds": selected.tolist(), "checkpoint_sha256": digest}


def trajectory_scores(predictions: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    """Descriptive point metrics/disagreements for review ordering, never selection.

    mean_uncertainty = mean(2*min(p, 1-p)) across rows and both labels (0–1).
    f1 fields are point F1; event clustering is only in evaluate_frame metrics.
    An absent positive label has NaN AP and zero P/R/F1, matching training code.
    """
    from moma_lineage_analysis.trajectory_cleaning.training import _point_metrics

    frame = validate_predictions(predictions)
    selected = threshold_pair(thresholds)
    rows = []
    for key, group in frame.groupby("trajectory_key", sort=False):
        targets = group.loc[:, LABELS].to_numpy(dtype=bool)
        probabilities = group.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
        flags = probabilities >= selected
        metrics = _point_metrics(targets, probabilities, selected)
        row = {"trajectory_key": key, "rows": len(group),
               "division_points": int(targets[:, 0].sum()), "excluded_points": int(targets[:, 1].sum()),
               "division_disagreements": int(np.sum(flags[:, 0] != targets[:, 0])),
               "exclusion_disagreements": int(np.sum(flags[:, 1] != targets[:, 1])),
               "mean_uncertainty": float(np.mean(2 * np.minimum(probabilities, 1 - probabilities)))}
        for prefix, label in (("division", "division_point"), ("exclusion", "excluded_point")):
            for metric in ("precision", "recall", "f1", "average_precision", "support"):
                row[f"{prefix}_{metric}"] = metrics[label][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def split_membership(checkpoint_path: str | Path, keys: Iterable[str]) -> dict[str, str]:
    """Read companion splits.json; absent companion means unknown/unseen keys.

    Membership describes the checkpoint's recorded partition, not a guarantee of
    biological independence. A malformed or overlapping companion is an error.
    """
    values = list(keys)
    if any(not isinstance(key, str) for key in values):
        raise ValueError("Trajectory keys must be strings")
    companion = Path(checkpoint_path).with_name("splits.json")
    if not companion.exists():
        return dict.fromkeys(values, "unseen")
    splits = json.loads(companion.read_text(encoding="utf-8-sig"))
    lookup = {}
    for partition in ("train", "validation", "test"):
        partition_keys = splits.get(partition)
        if not isinstance(partition_keys, list) or any(not isinstance(key, str) for key in partition_keys):
            raise ValueError(f"splits.json must contain a list of string keys for {partition}")
        for key in partition_keys:
            if key in lookup:
                raise ValueError(f"Duplicate/overlapping trajectory in splits.json: {key}")
            lookup[key] = partition
    return {key: lookup.get(key, "unseen") for key in values}


def cache_key(frame: pd.DataFrame, checkpoint_path: str | Path, device: str = "cpu",
              thresholds: Iterable[float] | None = None, *, include_labels: bool = True) -> str:
    """Key cached inference by actual ordered frame content and checkpoint bytes.

    Includes columns, dtypes, index, row order, labels, and measurements; dirty
    working data cannot reuse an old prediction by lineage/name alone. Pass
    include_labels=False only for prediction caches that are invariant to label
    edits (model inputs use measurements, not label columns). The checkpoint hash
    covers its stored thresholds when no override is supplied. This is a local
    cache identity, not a persisted scientific corpus hash.
    """
    source = validate_frame(frame)
    if not include_labels:
        source = source.drop(columns=list(LABELS))
    header = json.dumps({"columns": list(source.columns), "dtypes": [str(d) for d in source.dtypes],
                         "index_names": [str(name) for name in source.index.names]}, sort_keys=True).encode()
    content = hashlib.sha256(header + pd.util.hash_pandas_object(source, index=True).to_numpy().tobytes()).hexdigest()
    record = {"input_content_sha256": content, "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
              "device": str(device), "thresholds": None if thresholds is None else threshold_pair(thresholds).tolist()}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
