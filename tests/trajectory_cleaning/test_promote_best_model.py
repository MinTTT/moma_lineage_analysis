"""Focused tests for deterministic validation-only model promotion."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from moma_lineage_analysis.trajectory_cleaning.promote_best_model import (
    PROMOTED_ARTIFACTS,
    promote_best_model,
)


ARCHITECTURES = ("cnn", "resnet", "lstm")


def _write_training_run(
    run_dir: Path,
    *,
    validation: dict[str, tuple[float, float]],
    losses: dict[str, float] | None = None,
    test_scores: dict[str, float] | None = None,
) -> None:
    split = {
        "train": ["trajectory_1", "trajectory_2"],
        "validation": ["trajectory_3"],
        "test": ["trajectory_4"],
    }
    losses = losses or {architecture: 0.2 for architecture in ARCHITECTURES}
    test_scores = test_scores or {architecture: 0.5 for architecture in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        model_dir = run_dir / architecture
        model_dir.mkdir(parents=True)
        division_f1, excluded_f1 = validation[architecture]
        metrics = {
            "validation": {
                "division_event": {"f1": division_f1},
                "point": {"excluded_point": {"f1": excluded_f1}},
            },
            # Intentionally present to ensure promotion can never rank by it.
            "test": {
                "division_event": {"f1": test_scores[architecture]},
                "point": {
                    "excluded_point": {"f1": test_scores[architecture]}
                },
            },
        }
        (model_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        with (model_dir / "history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("epoch", "train_loss", "validation_loss")
            )
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": 1,
                    "train_loss": 0.4,
                    "validation_loss": losses[architecture] + 0.1,
                }
            )
            writer.writerow(
                {
                    "epoch": 2,
                    "train_loss": 0.3,
                    "validation_loss": losses[architecture],
                }
            )
        (model_dir / "splits.json").write_text(
            json.dumps(split), encoding="utf-8"
        )
        (model_dir / "best_model.pt").write_bytes(
            f"checkpoint:{architecture}".encode("ascii")
        )


class PromotionTests(unittest.TestCase):
    def test_selects_highest_mean_validation_f1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            destination = Path(temporary_directory) / "active"
            _write_training_run(
                run_dir,
                validation={
                    "cnn": (0.95, 0.45),  # mean 0.70
                    "resnet": (0.82, 0.78),  # mean 0.80
                    "lstm": (0.98, 0.52),  # mean 0.75
                },
            )

            result = promote_best_model(run_dir, destination_dir=destination)

            self.assertEqual(result["selected_model"], "resnet")
            self.assertEqual(
                (destination / "best_model.pt").read_bytes(), b"checkpoint:resnet"
            )
            manifest = json.loads(
                (destination / "promotion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["selected_model"], "resnet")
            self.assertFalse(manifest["selection"]["uses_test_metrics"])
            self.assertEqual(
                set(manifest["selection"]["validation_scores"]),
                set(ARCHITECTURES),
            )

    def test_test_metrics_are_ignored_and_name_breaks_complete_tie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            _write_training_run(
                run_dir,
                validation={architecture: (0.8, 0.6) for architecture in ARCHITECTURES},
                losses={architecture: 0.2 for architecture in ARCHITECTURES},
                test_scores={"cnn": 0.0, "resnet": 1.0, "lstm": 0.9},
            )

            result = promote_best_model(run_dir)

            # Alphabetical architecture name is the final deterministic tie-breaker.
            self.assertEqual(result["selected_model"], "cnn")
            self.assertEqual(
                (run_dir / "best_model.pt").read_bytes(), b"checkpoint:cnn"
            )

    def test_rejects_different_split_manifests_before_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            destination = Path(temporary_directory) / "active"
            _write_training_run(
                run_dir,
                validation={architecture: (0.8, 0.6) for architecture in ARCHITECTURES},
            )
            mismatched = json.loads(
                (run_dir / "lstm" / "splits.json").read_text(encoding="utf-8")
            )
            mismatched["validation"] = ["different_trajectory"]
            (run_dir / "lstm" / "splits.json").write_text(
                json.dumps(mismatched), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Split manifests"):
                promote_best_model(run_dir, destination_dir=destination)

            self.assertFalse(destination.exists())

    def test_existing_destination_artifacts_are_preserved_in_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "run"
            destination = root / "active"
            backup = root / "explicit_backup"
            _write_training_run(
                run_dir,
                validation={
                    "cnn": (0.9, 0.8),
                    "resnet": (0.8, 0.6),
                    "lstm": (0.7, 0.5),
                },
            )
            destination.mkdir()
            previous = {
                filename: f"previous:{filename}".encode("ascii")
                for filename in PROMOTED_ARTIFACTS
            }
            for filename, contents in previous.items():
                (destination / filename).write_bytes(contents)

            result = promote_best_model(
                run_dir, destination_dir=destination, backup_dir=backup
            )

            self.assertEqual(result["backup_path"], str(backup.resolve()))
            for filename, contents in previous.items():
                with self.subTest(filename=filename):
                    self.assertEqual((backup / filename).read_bytes(), contents)
                    self.assertEqual(
                        (destination / filename).read_bytes(),
                        (run_dir / "cnn" / filename).read_bytes(),
                    )
                    artifact = result["artifacts"][filename]
                    self.assertEqual(
                        artifact["source_sha256"], artifact["destination_sha256"]
                    )


if __name__ == "__main__":
    unittest.main()
