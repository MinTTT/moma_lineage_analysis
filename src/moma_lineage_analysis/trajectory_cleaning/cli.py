"""Train or apply a trajectory-cleaning sequence model.

Examples
--------
Train the recommended residual 1D CNN::

    moma-trajectory-cleaning train --input labeled.csv --output-dir run

Apply a saved checkpoint without deleting any source rows::

    moma-trajectory-cleaning predict --input trajectories.csv \
        --checkpoint run/best_model.pt --output predictions.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    LABEL_COLUMNS,
    TrajectoryDataset,
    TrajectoryNormalizer,
    load_trajectory_csv,
    split_trajectory_keys,
    validate_trajectory_dataframe,
)
from .dataset_store import connect_readonly, load_training_frame
from .models import ModelConfig, build_model
from .training import (
    TrainConfig,
    evaluate_model,
    load_checkpoint,
    predict_dataframe,
    save_checkpoint,
    set_reproducible_seed,
    train_model,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, values: Any) -> None:
    path.write_text(
        json.dumps(values, indent=2, default=_json_default, allow_nan=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset_manifest(
    path: str | Path | None,
    *,
    require_human_verified: bool,
) -> dict[str, Any] | None:
    """Load and validate optional training-data provenance."""

    if path is None:
        if require_human_verified:
            raise ValueError(
                "--require-human-verified also requires --dataset-manifest"
            )
        return None
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must contain a JSON object")

    target_columns = manifest.get(
        "canonical_target_columns", manifest.get("target_columns")
    )
    if target_columns != list(LABEL_COLUMNS):
        raise ValueError(
            "Dataset manifest must declare exactly the canonical targets "
            f"{list(LABEL_COLUMNS)!r}; received {target_columns!r}"
        )
    if require_human_verified and manifest.get("human_verified") is not True:
        raise ValueError(
            "Dataset manifest is not explicitly attested as human verified"
        )
    manifest = dict(manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def _summarize_splits(
    dataframe: pd.DataFrame,
    splits: dict[str, list[Any]],
    *,
    trajectory_column: str,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split_name, keys in splits.items():
        subset = dataframe[dataframe[trajectory_column].isin(keys)]
        summary: dict[str, Any] = {
            "trajectories": int(subset[trajectory_column].nunique()),
            "rows": int(len(subset)),
            "division_points": int(subset[LABEL_COLUMNS[0]].sum()),
            "excluded_points": int(subset[LABEL_COLUMNS[1]].sum()),
        }
        if "source_dataset" in subset.columns:
            summary["trajectories_by_source"] = (
                subset.groupby("source_dataset", sort=True)[trajectory_column]
                .nunique()
                .astype(int)
                .to_dict()
            )
        summaries[split_name] = summary
    return summaries


def _load_store_training_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the training frame from the SQLite dataset store (read-only)."""

    store_path = Path(args.store)
    connection = connect_readonly(store_path)
    try:
        dataframe, lineage_block = load_training_frame(
            connection,
            args.dataset_lineage,
            require_human_verified=args.require_human_verified,
            store_path=store_path,
        )
    finally:
        connection.close()
    dataframe = validate_trajectory_dataframe(
        dataframe,
        trajectory_column=args.trajectory_column,
        time_column=args.time_column,
        length_column=args.length_column,
        require_labels=True,
    )
    return dataframe, lineage_block


def train_command(args: argparse.Namespace) -> None:
    if args.dataset_lineage and args.input:
        raise ValueError("pass either --input or --dataset-lineage, not both")
    if not args.dataset_lineage and not args.input:
        raise ValueError("one of --input or --dataset-lineage is required")
    if args.dataset_lineage and args.dataset_manifest:
        raise ValueError(
            "--dataset-manifest is only for CSV input; store lineages carry "
            "their own attestation"
        )

    dataset_manifest: dict[str, Any] | None = None
    dataset_lineage: dict[str, Any] | None = None
    input_path: Path | None = None
    if args.dataset_lineage:
        dataframe, dataset_lineage = _load_store_training_data(args)
    else:
        dataset_manifest = _load_dataset_manifest(
            args.dataset_manifest,
            require_human_verified=args.require_human_verified,
        )
        input_path = Path(args.input)
        dataframe = load_trajectory_csv(
            input_path,
            trajectory_column=args.trajectory_column,
            time_column=args.time_column,
            length_column=args.length_column,
            require_labels=True,
        )
    splits = split_trajectory_keys(
        dataframe,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        trajectory_column=args.trajectory_column,
        source_column=args.source_column,
    )
    normalizer = TrajectoryNormalizer()
    dataset_arguments = {
        "normalizer": normalizer,
        "trajectory_column": args.trajectory_column,
        "time_column": args.time_column,
        "length_column": args.length_column,
    }
    train_dataset = TrajectoryDataset(
        dataframe,
        keys=splits["train"],
        **dataset_arguments,
    )
    validation_dataset = TrajectoryDataset(
        dataframe,
        keys=splits["validation"],
        **dataset_arguments,
    )
    test_dataset = TrajectoryDataset(
        dataframe,
        keys=splits["test"],
        **dataset_arguments,
    )

    model_config = ModelConfig(
        model_type=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        bidirectional=not args.unidirectional,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        maximum_positive_weight=args.maximum_positive_weight,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        division_tolerance_frames=args.division_tolerance_frames,
    )
    set_reproducible_seed(args.seed)
    model = build_model(model_config)
    model, history, thresholds = train_model(
        model,
        train_dataset,
        validation_dataset,
        config=train_config,
    )
    validation_metrics = evaluate_model(
        model,
        validation_dataset,
        thresholds=thresholds,
        batch_size=args.batch_size,
        device=args.device,
        division_tolerance_frames=args.division_tolerance_frames,
    )
    test_metrics = evaluate_model(
        model,
        test_dataset,
        thresholds=thresholds,
        batch_size=args.batch_size,
        device=args.device,
        division_tolerance_frames=args.division_tolerance_frames,
    )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    training_data: dict[str, Any]
    if input_path is not None:
        training_data = {
            "source": "csv",
            "path": str(input_path.resolve()),
            "sha256": _sha256_file(input_path),
            "rows": int(len(dataframe)),
            "trajectories": int(dataframe[args.trajectory_column].nunique()),
            "target_columns": list(LABEL_COLUMNS),
        }
    else:
        training_data = {
            "source": "sqlite_dataset_store",
            "store_path": dataset_lineage["store_path"],
            "rows": int(len(dataframe)),
            "trajectories": int(dataframe[args.trajectory_column].nunique()),
            "target_columns": list(LABEL_COLUMNS),
            "lineage_ids": list(dataset_lineage["composition"]["lineage_ids"]),
        }
    metadata = {
        "trajectory_column": args.trajectory_column,
        "time_column": args.time_column,
        "length_column": args.length_column,
        "source_column": args.source_column,
        "train_config": train_config.to_dict(),
        "training_data": training_data,
    }
    if dataset_manifest is not None:
        metadata["dataset_manifest"] = dataset_manifest
    if dataset_lineage is not None:
        metadata["dataset_lineage"] = dataset_lineage
    save_checkpoint(
        output_directory / "best_model.pt",
        model,
        model_config=model_config,
        normalizer=normalizer,
        thresholds=thresholds,
        metadata=metadata,
    )
    pd.DataFrame(history).to_csv(output_directory / "history.csv", index=False)
    _write_json(output_directory / "splits.json", splits)
    metrics = {
        "thresholds": {
            "division_point": float(thresholds[0]),
            "excluded_point": float(thresholds[1]),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "training_data": metadata["training_data"],
        "split_summary": _summarize_splits(
            dataframe,
            splits,
            trajectory_column=args.trajectory_column,
        ),
    }
    if dataset_manifest is not None:
        metrics["dataset_manifest"] = dataset_manifest
    if dataset_lineage is not None:
        metrics["dataset_lineage"] = dataset_lineage
    _write_json(output_directory / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, default=_json_default, allow_nan=True))


def predict_command(args: argparse.Namespace) -> None:
    model, normalizer, thresholds, metadata = load_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    trajectory_column = args.trajectory_column or metadata.get(
        "trajectory_column", "trajectory_key"
    )
    time_column = args.time_column or metadata.get("time_column", "time_h")
    length_column = args.length_column or metadata.get("length_column", "spine_length")
    dataframe = pd.read_csv(args.input)
    result = predict_dataframe(
        model,
        dataframe,
        normalizer=normalizer,
        thresholds=thresholds,
        trajectory_column=trajectory_column,
        time_column=time_column,
        length_column=length_column,
        batch_size=args.batch_size,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} labeled rows to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and evaluate a sequence model")
    train.add_argument(
        "--input",
        help="Labeled trajectory CSV (mutually exclusive with --dataset-lineage)",
    )
    train.add_argument(
        "--dataset-lineage",
        action="append",
        default=[],
        help=(
            "Frozen dataset lineage id from the SQLite store, e.g. "
            "verified_merged_20261017@v1. Repeat to compose versions (duplicate "
            "observations are collapsed). Lineage ids are recorded in metrics.json."
        ),
    )
    train.add_argument(
        "--store",
        default="datasets/store/lineage.sqlite3",
        help="SQLite dataset store path (default: datasets/store/lineage.sqlite3)",
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--model", choices=["resnet", "cnn", "lstm"], default="resnet")
    train.add_argument("--trajectory-column", default="trajectory_key")
    train.add_argument("--time-column", default="time_h")
    train.add_argument("--length-column", default="spine_length")
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=4)
    train.add_argument("--kernel-size", type=int, default=5)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--unidirectional", action="store_true")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--focal-gamma", type=float, default=1.5)
    train.add_argument("--maximum-positive-weight", type=float, default=50.0)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--test-fraction", type=float, default=0.15)
    train.add_argument(
        "--source-column",
        help=(
            "Optional acquisition-source column. When provided, each source is "
            "split independently so it appears in train, validation, and test."
        ),
    )
    train.add_argument("--division-tolerance-frames", type=int, default=1)
    train.add_argument(
        "--dataset-manifest",
        help="JSON provenance manifest for the prepared training CSV",
    )
    train.add_argument(
        "--require-human-verified",
        action="store_true",
        help=(
            "Refuse training unless the manifest explicitly declares "
            "human_verified=true and the canonical target columns"
        ),
    )
    train.set_defaults(function=train_command)

    predict = subparsers.add_parser("predict", help="Append predictions to a CSV")
    predict.add_argument("--input", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--trajectory-column")
    predict.add_argument("--time-column")
    predict.add_argument("--length-column")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(function=predict_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
