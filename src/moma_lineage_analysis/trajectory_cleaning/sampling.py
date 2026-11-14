"""Opt-in training windows sliced after full-trajectory normalization.

An epoch contains as many windows as the original training set has trajectories,
so batch size and optimizer-step count are unchanged. Each update sees fewer
rows, and event oversampling changes the effective objective (no importance
correction). This is not equivalent to full-trajectory training.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .data import TrajectoryDataset, TrajectoryRecord, collate_trajectories
from .models import CNNLabeler, ResNetLabeler


def convolution_halo(model: nn.Module) -> int:
    """Require finite, centered convolutional context; sequence-wide LSTM is unsafe."""
    receptive_field = getattr(model, "receptive_field", None)
    if (
        not isinstance(model, (CNNLabeler, ResNetLabeler))
        or isinstance(receptive_field, bool)
        or not isinstance(receptive_field, Integral)
        or receptive_field <= 0
        or receptive_field % 2 != 1
    ):
        raise ValueError(
            "exclusion_windows requires a CNN/ResNet with a finite odd receptive_field; "
            "LSTM sequence-wide context is not supported"
        )
    return (int(receptive_field) - 1) // 2


@dataclass(frozen=True)
class WindowSelection:
    record_index: int
    center: int
    core_start: int
    core_stop: int
    start: int
    stop: int
    event_centered: bool


@dataclass
class TrajectoryWindow(TrajectoryRecord):
    loss_mask: torch.Tensor


class ExclusionWindowDataset(Dataset[TrajectoryWindow]):
    """Sample event midpoints and uniformly sampled observation centers.

    The event pool contains one lower midpoint per contiguous exclusion run,
    irrespective of run length. The uniform pool contains every training row,
    including entirely clean trajectories. Sampling is with replacement. The
    event quota is round-half-up(len(dataset) * positive_fraction); an odd
    epoch size cannot represent exactly 50%. Call set_epoch before each epoch.
    Crops are clipped at real trajectory boundaries, never shifted or normalized
    again. Only the core contributes loss; all real halo rows reach the model.
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        model: nn.Module,
        *,
        core: int = 65,
        positive_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        if isinstance(core, bool) or not isinstance(core, Integral) or core <= 0 or core % 2 != 1:
            raise ValueError("window_core must be a positive odd integer")
        if not np.isfinite(positive_fraction) or not 0 <= positive_fraction <= 1:
            raise ValueError("window_positive_fraction must be finite and in [0, 1]")
        self.halo = convolution_halo(model)
        self.records = dataset.records
        self.core = int(core)
        self.positive_fraction = float(positive_fraction)
        self.seed = int(seed)
        self.lengths = np.asarray([record.length for record in self.records], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.events: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            indices = np.flatnonzero(record.targets[:, 1].numpy() >= 0.5)
            for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if run.size:
                    self.events.append((record_index, int((run[0] + run[-1]) // 2)))
        if self.positive_fraction > 0 and not self.events:
            raise ValueError("exclusion_windows requires training exclusion events when window_positive_fraction > 0")
        self.selections: list[WindowSelection] = []
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.records)

    def select(self, record_index: int, center: int, *, event_centered: bool = False) -> WindowSelection:
        """Describe one clipped crop using positions within its source trajectory."""
        length = self.records[record_index].length
        if not 0 <= center < length:
            raise ValueError("window center must be a real observation")
        core_start = max(0, center - self.core // 2)
        core_stop = min(length, center + self.core // 2 + 1)
        return WindowSelection(
            record_index=record_index, center=center,
            core_start=core_start, core_stop=core_stop,
            start=max(0, core_start - self.halo),
            stop=min(length, core_stop + self.halo),
            event_centered=event_centered,
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self.epoch)
        positive_count = int(np.floor(len(self) * self.positive_fraction + 0.5))
        selections = []
        for _ in range(positive_count):
            record_index, center = self.events[int(rng.integers(len(self.events)))]
            selections.append(self.select(record_index, center, event_centered=True))
        for _ in range(len(self) - positive_count):
            observation = int(rng.integers(int(self.cumulative_lengths[-1])))
            record_index = int(np.searchsorted(self.cumulative_lengths, observation, side="right"))
            offset = int(self.cumulative_lengths[record_index - 1]) if record_index else 0
            selections.append(self.select(record_index, observation - offset))
        rng.shuffle(selections)
        self.selections = selections

    def __getitem__(self, index: int) -> TrajectoryWindow:
        selection = self.selections[index]
        record = self.records[selection.record_index]
        loss_mask = torch.zeros(selection.stop - selection.start, dtype=torch.bool)
        loss_mask[selection.core_start - selection.start:selection.core_stop - selection.start] = True
        return TrajectoryWindow(
            key=record.key,
            features=record.features[selection.start:selection.stop],
            targets=record.targets[selection.start:selection.stop],
            row_positions=record.row_positions[selection.start:selection.stop],
            loss_mask=loss_mask,
        )

    def sample_manifest(self) -> list[dict[str, Any]]:
        """Keep source keys and exact dataframe row positions for epoch replay."""
        manifest = []
        for selection in self.selections:
            record = self.records[selection.record_index]
            manifest.append({
                "key": record.key,
                "center": selection.center,
                "center_row_position": int(record.row_positions[selection.center]),
                "core_start": selection.core_start, "core_stop": selection.core_stop,
                "start": selection.start, "stop": selection.stop,
                "row_positions": record.row_positions[selection.start:selection.stop].tolist(),
                "event_centered": selection.event_centered,
            })
        return manifest


def collate_windows(batch: Sequence[TrajectoryWindow]) -> dict[str, Any]:
    """Use the existing real-row mask for features and a separate mask for loss."""
    result = collate_trajectories(batch)
    loss_mask = torch.zeros_like(result["mask"])
    for index, record in enumerate(batch):
        loss_mask[index, :record.length] = record.loss_mask
    result["loss_mask"] = loss_mask
    return result
