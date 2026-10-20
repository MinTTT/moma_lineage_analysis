"""Sequence models for cleaning mother-machine cell trajectories.

The public API intentionally stays small: load a labeled CSV, split it by
``trajectory_key``, build a trajectory dataset, and train or load one of the
supported PyTorch sequence models.
"""

from .data import (
    LABEL_COLUMNS,
    TrajectoryDataset,
    TrajectoryNormalizer,
    collate_trajectories,
    load_trajectory_csv,
    split_trajectory_keys,
)
from .models import ModelConfig, build_model
from .streamlit_api import (
    AIInputPreflight,
    DIVISION_RESOLUTION_STATUSES,
    DivisionConflictResolution,
    preflight_ai_trajectory_inputs,
    resolve_division_exclusion_conflicts,
)
from .training import (
    TrainConfig,
    evaluate_model,
    load_checkpoint,
    predict_dataframe,
    save_checkpoint,
    train_model,
)

__all__ = [
    "AIInputPreflight",
    "LABEL_COLUMNS",
    "DIVISION_RESOLUTION_STATUSES",
    "DivisionConflictResolution",
    "ModelConfig",
    "TrainConfig",
    "TrajectoryDataset",
    "TrajectoryNormalizer",
    "build_model",
    "collate_trajectories",
    "evaluate_model",
    "load_checkpoint",
    "load_trajectory_csv",
    "predict_dataframe",
    "preflight_ai_trajectory_inputs",
    "resolve_division_exclusion_conflicts",
    "save_checkpoint",
    "split_trajectory_keys",
    "train_model",
]
