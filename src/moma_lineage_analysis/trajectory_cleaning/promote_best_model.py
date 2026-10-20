"""Select and promote the best trajectory-cleaning architecture.

Model selection is deliberately restricted to validation results.  The held-out
test metrics may be present in ``metrics.json``, but this module never reads them
when ranking architectures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ARCHITECTURES = ("cnn", "resnet", "lstm")
PROMOTED_ARTIFACTS = ("best_model.pt", "history.csv", "metrics.json", "splits.json")


@dataclass(frozen=True)
class ValidationScore:
    """Validation-only values used to rank one architecture."""

    architecture: str
    mean_f1: float
    excluded_point_f1: float
    division_event_f1: float
    minimum_validation_loss: float

    def ranking_key(self) -> tuple[float, float, float, float, str]:
        """Return an ascending key that implements the documented ranking."""

        return (
            -self.mean_f1,
            -self.excluded_point_f1,
            -self.division_event_f1,
            self.minimum_validation_loss,
            self.architecture,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_f1": self.mean_f1,
            "excluded_point_f1": self.excluded_point_f1,
            "division_event_f1": self.division_event_f1,
            "minimum_validation_loss": self.minimum_validation_loss,
        }


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required model artifact is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _finite_number(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} in {path} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} in {path} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} in {path} must be finite")
    return number


def _nested_value(document: Mapping[str, Any], keys: Sequence[str], *, path: Path) -> Any:
    value: Any = document
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            dotted = ".".join(keys)
            raise ValueError(f"Missing {dotted} in {path}")
        value = value[key]
    return value


def _minimum_validation_loss(path: Path) -> float:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "validation_loss" not in reader.fieldnames:
                raise ValueError(f"Missing validation_loss column in {path}")
            losses = [
                _finite_number(row.get("validation_loss"), field="validation_loss", path=path)
                for row in reader
            ]
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required model artifact is missing: {path}") from error
    if not losses:
        raise ValueError(f"No training epochs found in {path}")
    return min(losses)


def _score_architecture(run_dir: Path, architecture: str) -> ValidationScore:
    model_dir = run_dir / architecture
    for filename in PROMOTED_ARTIFACTS:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")

    metrics_path = model_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Expected a JSON object in {metrics_path}")

    division_f1 = _finite_number(
        _nested_value(metrics, ("validation", "division_event", "f1"), path=metrics_path),
        field="validation.division_event.f1",
        path=metrics_path,
    )
    excluded_f1 = _finite_number(
        _nested_value(
            metrics,
            ("validation", "point", "excluded_point", "f1"),
            path=metrics_path,
        ),
        field="validation.point.excluded_point.f1",
        path=metrics_path,
    )
    return ValidationScore(
        architecture=architecture,
        mean_f1=(division_f1 + excluded_f1) / 2.0,
        excluded_point_f1=excluded_f1,
        division_event_f1=division_f1,
        minimum_validation_loss=_minimum_validation_loss(model_dir / "history.csv"),
    )


def _validate_identical_splits(run_dir: Path) -> tuple[Any, Path]:
    reference_architecture = ARCHITECTURES[0]
    reference_path = run_dir / reference_architecture / "splits.json"
    reference = _read_json(reference_path)
    for architecture in ARCHITECTURES[1:]:
        candidate_path = run_dir / architecture / "splits.json"
        candidate = _read_json(candidate_path)
        if candidate != reference:
            raise ValueError(
                "Split manifests must be exactly identical across models; "
                f"{candidate_path} differs from {reference_path}"
            )
    return reference, reference_path


def select_best_model(run_dir: str | os.PathLike[str]) -> tuple[ValidationScore, dict[str, ValidationScore]]:
    """Return the deterministic validation winner and all validation scores."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    _validate_identical_splits(root)
    scores = {
        architecture: _score_architecture(root, architecture)
        for architecture in ARCHITECTURES
    }
    selected = min(scores.values(), key=ValidationScore.ranking_key)
    return selected, scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_timestamped_backup(destination_dir: Path, timestamp: datetime) -> Path:
    backup_root = destination_dir / "promotion_backups"
    stem = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = backup_root / stem
    suffix = 1
    while candidate.exists():
        candidate = backup_root / f"{stem}-{suffix}"
        suffix += 1
    return candidate


def _stage_copy(source: Path, destination_dir: Path, destination_name: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_name}.", suffix=".tmp", dir=destination_dir
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_json_stage(document: Mapping[str, Any], destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".promotion.json.", suffix=".tmp", dir=destination_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _backup_existing_artifacts(
    destination_dir: Path,
    backup_dir: Path | None,
    timestamp: datetime,
) -> tuple[Path | None, set[str]]:
    existing = {
        filename
        for filename in PROMOTED_ARTIFACTS
        if (destination_dir / filename).is_file()
    }
    if not existing:
        return None, existing

    target_dir = backup_dir or _new_timestamped_backup(destination_dir, timestamp)
    target_dir = target_dir.expanduser().resolve()
    if target_dir == destination_dir:
        raise ValueError("Backup directory must differ from the destination directory")
    target_dir.mkdir(parents=True, exist_ok=True)

    conflicting = [filename for filename in existing if (target_dir / filename).exists()]
    if conflicting:
        raise FileExistsError(
            "Refusing to overwrite existing backup artifacts: "
            + ", ".join(str(target_dir / filename) for filename in sorted(conflicting))
        )

    staged: dict[str, Path] = {}
    try:
        for filename in sorted(existing):
            staged[filename] = _stage_copy(
                destination_dir / filename, target_dir, filename
            )
        for filename, temporary_path in staged.items():
            os.replace(temporary_path, target_dir / filename)
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)
    return target_dir, existing


def _restore_after_failure(
    destination_dir: Path,
    backup_path: Path | None,
    originally_existing: set[str],
    replaced: Sequence[str],
) -> None:
    """Best-effort rollback if an artifact replacement fails."""

    for filename in reversed(replaced):
        destination = destination_dir / filename
        try:
            if filename in originally_existing and backup_path is not None:
                staged = _stage_copy(backup_path / filename, destination_dir, filename)
                os.replace(staged, destination)
            else:
                destination.unlink(missing_ok=True)
        except Exception:
            # Preserve the original promotion exception.  The backup remains on disk
            # for manual recovery if this best-effort rollback also encounters an error.
            pass


def promote_best_model(
    run_dir: str | os.PathLike[str],
    *,
    destination_dir: str | os.PathLike[str] | None = None,
    backup_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Select by validation metrics and copy the winning artifacts to the root.

    Existing root artifacts are preserved before any replacement.  The returned
    dictionary is also written to ``promotion.json`` in ``destination_dir``.
    """

    root = Path(run_dir).expanduser().resolve()
    destination = (
        Path(destination_dir).expanduser().resolve()
        if destination_dir is not None
        else root
    )
    explicit_backup = (
        Path(backup_dir).expanduser().resolve() if backup_dir is not None else None
    )
    selected, scores = select_best_model(root)
    _, common_splits_path = _validate_identical_splits(root)

    source_dir = root / selected.architecture
    source_hashes = {
        filename: _sha256(source_dir / filename) for filename in PROMOTED_ARTIFACTS
    }
    timestamp = datetime.now(timezone.utc)
    promoted_at = timestamp.isoformat().replace("+00:00", "Z")

    destination.mkdir(parents=True, exist_ok=True)
    backup_path, originally_existing = _backup_existing_artifacts(
        destination, explicit_backup, timestamp
    )

    artifacts = {
        filename: {
            "source": str(source_dir / filename),
            "destination": str(destination / filename),
            "source_sha256": source_hashes[filename],
            "destination_sha256": source_hashes[filename],
        }
        for filename in PROMOTED_ARTIFACTS
    }
    promotion: dict[str, Any] = {
        "schema_version": 1,
        "promoted_at_utc": promoted_at,
        "run_dir": str(root),
        "destination_dir": str(destination),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "selected_model": selected.architecture,
        "selection": {
            "uses_test_metrics": False,
            "criterion": {
                "primary": (
                    "arithmetic mean of validation.division_event.f1 and "
                    "validation.point.excluded_point.f1"
                ),
                "tie_breakers": [
                    "higher validation.point.excluded_point.f1",
                    "higher validation.division_event.f1",
                    "lower minimum validation_loss from history.csv",
                    "architecture name in ascending lexical order",
                ],
            },
            "validation_scores": {
                architecture: scores[architecture].as_dict()
                for architecture in sorted(scores)
            },
        },
        "common_split_manifest": {
            "source": str(common_splits_path),
            "sha256": _sha256(common_splits_path),
        },
        "artifacts": artifacts,
    }

    staged = {
        filename: _stage_copy(source_dir / filename, destination, filename)
        for filename in PROMOTED_ARTIFACTS
    }
    promotion_stage = _write_json_stage(promotion, destination)
    replaced: list[str] = []
    try:
        for filename in PROMOTED_ARTIFACTS:
            os.replace(staged[filename], destination / filename)
            replaced.append(filename)
        for filename in PROMOTED_ARTIFACTS:
            destination_hash = _sha256(destination / filename)
            if destination_hash != source_hashes[filename]:
                raise OSError(
                    f"Hash verification failed after promoting {destination / filename}"
                )
        os.replace(promotion_stage, destination / "promotion.json")
    except Exception:
        _restore_after_failure(
            destination, backup_path, originally_existing, replaced
        )
        raise
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)
        promotion_stage.unlink(missing_ok=True)

    return promotion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the best CNN/ResNet/LSTM checkpoint using validation metrics only."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing cnn, resnet, and lstm training subdirectories.",
    )
    parser.add_argument(
        "--destination-dir",
        help="Destination for promoted root artifacts (default: --run-dir).",
    )
    parser.add_argument(
        "--backup-dir",
        help=(
            "Optional explicit directory for existing destination artifacts. "
            "By default a timestamped directory is created under promotion_backups."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = promote_best_model(
        arguments.run_dir,
        destination_dir=arguments.destination_dir,
        backup_dir=arguments.backup_dir,
    )
    print(f"Promoted {result['selected_model']} model")
    print(f"Promotion manifest: {Path(result['destination_dir']) / 'promotion.json'}")
    if result["backup_path"]:
        print(f"Previous artifacts backed up to: {result['backup_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
