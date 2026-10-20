"""Training, metrics, checkpointing, and dataframe inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    LABEL_COLUMNS,
    TrajectoryDataset,
    TrajectoryNormalizer,
    collate_trajectories,
)
from .models import ModelConfig, build_model


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    focal_gamma: float = 1.5
    maximum_positive_weight: float = 50.0
    patience: int = 15
    seed: int = 42
    device: str = "auto"
    division_tolerance_frames: int = 1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.focal_gamma < 0 or self.maximum_positive_weight < 1:
            raise ValueError("Invalid focal or positive-weight setting")
        if self.patience <= 0 or self.division_tolerance_frames < 0:
            raise ValueError("patience must be positive and tolerance nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MaskedFocalBCEWithLogitsLoss(nn.Module):
    """Independent binary losses for division and exclusion, ignoring padding."""

    def __init__(self, positive_weight: torch.Tensor, gamma: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("positive_weight", positive_weight.to(dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes")
        if mask.shape != logits.shape[:2]:
            raise ValueError("mask must match the batch and time dimensions")
        losses = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.positive_weight,
        )
        if self.gamma > 0:
            probability = torch.sigmoid(logits)
            target_probability = targets * probability + (1 - targets) * (1 - probability)
            losses = losses * (1 - target_probability).pow(self.gamma)
        valid = mask.unsqueeze(-1).expand_as(losses)
        return losses.masked_select(valid).mean()


def calculate_positive_weight(
    dataset: TrajectoryDataset,
    maximum: float = 50.0,
) -> torch.Tensor:
    targets = torch.cat([record.targets for record in dataset.records], dim=0)
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    weights = negatives / positives.clamp_min(1.0)
    return weights.clamp(min=1.0, max=maximum).to(dtype=torch.float32)


def _loader(
    dataset: TrajectoryDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_trajectories,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: MaskedFocalBCEWithLogitsLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_points = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features, mask)
            loss = loss_function(logits, targets, mask)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            point_count = int(mask.sum().item())
            total_loss += float(loss.detach().cpu()) * point_count
            total_points += point_count
    return total_loss / max(total_points, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataset: TrajectoryDataset,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    selected_device = resolve_device(device) if isinstance(device, str) else device
    model = model.to(selected_device)
    model.eval()
    results: list[dict[str, Any]] = []
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    for batch in loader:
        features = batch["features"].to(selected_device)
        mask = batch["mask"].to(selected_device)
        probabilities = torch.sigmoid(model(features, mask)).cpu().numpy()
        targets = batch["targets"].numpy()
        lengths = batch["lengths"].numpy()
        for index, length_value in enumerate(lengths):
            length = int(length_value)
            results.append(
                {
                    "key": batch["keys"][index],
                    "row_positions": batch["row_positions"][index],
                    "probabilities": probabilities[index, :length],
                    "targets": targets[index, :length],
                }
            )
    return results


def best_f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    if target.sum() == 0:
        return 0.5
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        prediction = probability >= threshold
        _, _, f1, _ = precision_recall_fscore_support(
            target,
            prediction,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def _match_event_indices(
    true_indices: Iterable[int],
    predicted_indices: Iterable[int],
    tolerance: int,
) -> tuple[int, int, int, list[int]]:
    true_values = sorted(int(value) for value in true_indices)
    predicted_values = sorted(int(value) for value in predicted_indices)
    unmatched = set(predicted_values)
    errors: list[int] = []
    for true_value in true_values:
        candidates = [
            value for value in unmatched if abs(value - true_value) <= tolerance
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda value: (abs(value - true_value), value))
        unmatched.remove(selected)
        errors.append(selected - true_value)
    true_positive = len(errors)
    return (
        true_positive,
        len(unmatched),
        len(true_values) - true_positive,
        errors,
    )


def cluster_event_indices(
    positive: np.ndarray,
    *,
    score: np.ndarray | None = None,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Collapse consecutive positive points to one biological event index.

    For labeled events, a non-excluded point is preferred. For predictions,
    the highest-scoring candidate in the consecutive run is selected.
    """

    positive = np.asarray(positive, dtype=bool)
    if positive.ndim != 1:
        raise ValueError("positive must be one-dimensional")
    if score is not None and np.asarray(score).shape != positive.shape:
        raise ValueError("score must have the same shape as positive")
    if excluded is not None and np.asarray(excluded).shape != positive.shape:
        raise ValueError("excluded must have the same shape as positive")
    indices = np.flatnonzero(positive)
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, boundaries)
    representatives: list[int] = []
    for run in runs:
        candidates = run
        if excluded is not None:
            non_excluded = run[~np.asarray(excluded, dtype=bool)[run]]
            if non_excluded.size:
                candidates = non_excluded
        if score is not None:
            values = np.asarray(score, dtype=float)[candidates]
            representatives.append(int(candidates[int(np.argmax(values))]))
        else:
            representatives.append(int(candidates[0]))
    return np.asarray(representatives, dtype=np.int64)


def _point_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for label_index, label in enumerate(LABEL_COLUMNS):
        target = targets[:, label_index].astype(bool)
        probability = probabilities[:, label_index]
        prediction = probability >= thresholds[label_index]
        precision, recall, f1, _ = precision_recall_fscore_support(
            target,
            prediction,
            average="binary",
            zero_division=0,
        )
        average_precision = (
            float(average_precision_score(target, probability))
            if target.any()
            else float("nan")
        )
        metrics[label] = {
            "threshold": float(thresholds[label_index]),
            "support": int(target.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "average_precision": average_precision,
        }
    return metrics


def evaluate_model(
    model: nn.Module,
    dataset: TrajectoryDataset,
    *,
    thresholds: Iterable[float] = (0.5, 0.5),
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    division_tolerance_frames: int = 1,
) -> dict[str, Any]:
    threshold_array = np.asarray(list(thresholds), dtype=float)
    if threshold_array.shape != (2,):
        raise ValueError("Exactly two thresholds are required")
    records = collect_predictions(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    targets = np.concatenate([record["targets"] for record in records], axis=0)
    probabilities = np.concatenate(
        [record["probabilities"] for record in records], axis=0
    )
    metrics = {"point": _point_metrics(targets, probabilities, threshold_array)}

    total_tp = total_fp = total_fn = 0
    errors: list[int] = []
    for record in records:
        truth = cluster_event_indices(
            record["targets"][:, 0] >= 0.5,
            excluded=record["targets"][:, 1] >= 0.5,
        )
        prediction = cluster_event_indices(
            record["probabilities"][:, 0] >= threshold_array[0],
            score=record["probabilities"][:, 0],
        )
        tp, fp, fn, record_errors = _match_event_indices(
            truth,
            prediction,
            division_tolerance_frames,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        errors.extend(record_errors)
    event_precision = total_tp / max(total_tp + total_fp, 1)
    event_recall = total_tp / max(total_tp + total_fn, 1)
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    metrics["division_event"] = {
        "tolerance_frames": int(division_tolerance_frames),
        "true_positive": int(total_tp),
        "false_positive": int(total_fp),
        "false_negative": int(total_fn),
        "precision": float(event_precision),
        "recall": float(event_recall),
        "f1": float(event_f1),
        "mean_absolute_error_frames": (
            float(np.mean(np.abs(errors))) if errors else float("nan")
        ),
    }

    # Round R002: position-stratified division recall. The pre-R002 failure
    # mode was collapsing recall in the last part of long trajectories, so the
    # tail deciles are now first-class acceptance metrics.
    division_index = LABEL_COLUMNS.index("division_point")
    division_targets = targets[:, division_index].astype(bool)
    division_probabilities = probabilities[:, division_index]
    division_predictions = division_probabilities >= threshold_array[division_index]
    offsets = np.concatenate(
        [[0], np.cumsum([len(record["targets"]) for record in records])[:-1]]
    )
    relative_positions: list[np.ndarray] = []
    for record, offset in zip(records, offsets):
        length = len(record["targets"])
        denominator = max(length - 1, 1)
        relative_positions.append(np.arange(length, dtype=np.float64) / denominator)
    relative_position = np.concatenate(relative_positions)

    position_metrics: dict[str, Any] = {}
    for name, lower, upper in (("head", 0.0, 0.2), ("tail", 0.8, 1.0)):
        mask = (
            (relative_position >= lower)
            & (relative_position <= upper)
            & division_targets
        )
        support = int(mask.sum())
        position_metrics[f"division_recall_{name}_quintile"] = {
            "support": support,
            "recall": (
                float(division_predictions[mask].mean()) if support else float("nan")
            ),
        }

    # Fraction of trajectories whose LAST true division event is missed —
    # the user-reported failure mode on the 21.6 h RDM-Gluc dataset.
    last_missed = 0
    with_division = 0
    for record in records:
        truth = cluster_event_indices(
            record["targets"][:, 0] >= 0.5,
            excluded=record["targets"][:, 1] >= 0.5,
        )
        if truth.size == 0:
            continue
        with_division += 1
        prediction = cluster_event_indices(
            record["probabilities"][:, 0] >= threshold_array[0],
            score=record["probabilities"][:, 0],
        )
        last_truth = truth[-1]
        matched = any(
            abs(predicted - last_truth) <= division_tolerance_frames
            for predicted in prediction
        )
        if not matched:
            last_missed += 1
    position_metrics["last_division_event"] = {
        "trajectories_with_division": int(with_division),
        "last_event_missed": int(last_missed),
        "miss_rate": float(last_missed / with_division) if with_division else float("nan"),
    }
    metrics["position"] = position_metrics
    return metrics


def train_model(
    model: nn.Module,
    train_dataset: TrajectoryDataset,
    validation_dataset: TrajectoryDataset,
    *,
    config: TrainConfig | None = None,
) -> tuple[nn.Module, list[dict[str, float]], np.ndarray]:
    """Train with early stopping and calibrate one threshold per output."""

    config = config or TrainConfig()
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    model = model.to(device)
    train_loader = _loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    positive_weight = calculate_positive_weight(
        train_dataset,
        maximum=config.maximum_positive_weight,
    ).to(device)
    loss_function = MaskedFocalBCEWithLogitsLoss(
        positive_weight,
        gamma=config.focal_gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")
    model.load_state_dict(best_state)

    validation_predictions = collect_predictions(
        model,
        validation_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    validation_targets = np.concatenate(
        [record["targets"] for record in validation_predictions], axis=0
    )
    validation_probabilities = np.concatenate(
        [record["probabilities"] for record in validation_predictions], axis=0
    )
    thresholds = np.asarray(
        [
            best_f1_threshold(
                validation_targets[:, label_index],
                validation_probabilities[:, label_index],
            )
            for label_index in range(2)
        ],
        dtype=np.float32,
    )
    return model, history, thresholds


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_config: ModelConfig,
    normalizer: TrajectoryNormalizer,
    thresholds: Iterable[float],
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format_version": 1,
        "model_config": model_config.to_dict(),
        "normalizer": normalizer.to_dict(),
        "thresholds": [float(value) for value in thresholds],
        "metadata": metadata or {},
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, TrajectoryNormalizer, np.ndarray, dict[str, Any]]:
    selected_device = resolve_device(device)
    try:
        checkpoint = torch.load(
            Path(path),
            map_location=selected_device,
            weights_only=True,
        )
    except TypeError:  # Compatibility with older supported PyTorch 2.x releases.
        checkpoint = torch.load(Path(path), map_location=selected_device)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported trajectory-cleaner checkpoint format")
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(model_config).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalizer = TrajectoryNormalizer.from_dict(checkpoint["normalizer"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    metadata = dict(checkpoint.get("metadata", {}))
    metadata["model_config"] = model_config.to_dict()
    return model, normalizer, thresholds, metadata


def predict_dataframe(
    model: nn.Module,
    dataframe: pd.DataFrame,
    *,
    normalizer: TrajectoryNormalizer | None = None,
    thresholds: Iterable[float] = (0.5, 0.5),
    trajectory_column: str = "trajectory_key",
    time_column: str = "time_h",
    length_column: str = "spine_length",
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Append probabilities and predictions while preserving input row order."""

    dataset = TrajectoryDataset(
        dataframe,
        normalizer=normalizer or TrajectoryNormalizer(),
        trajectory_column=trajectory_column,
        time_column=time_column,
        length_column=length_column,
    )
    records = collect_predictions(model, dataset, batch_size=batch_size, device=device)
    probability = np.full((len(dataframe), 2), np.nan, dtype=np.float32)
    for record in records:
        probability[record["row_positions"]] = record["probabilities"]
    if np.isnan(probability).any():
        raise RuntimeError("Not every dataframe row received a prediction")
    threshold_array = np.asarray(list(thresholds), dtype=np.float32)
    if threshold_array.shape != (2,):
        raise ValueError("Exactly two thresholds are required")

    result = dataframe.copy()
    result["pred_division_probability"] = probability[:, 0]
    result["pred_excluded_probability"] = probability[:, 1]
    result["pred_division_point"] = probability[:, 0] >= threshold_array[0]
    result["pred_excluded_point"] = probability[:, 1] >= threshold_array[1]
    return result
