"""Contract-integrity tests for the trajectory-cleaning governance surface.

These tests pin the promoted checkpoint, run artifacts, dataset manifest, and
the Methods contract (METHOD_MANIFEST) so that silent drift in any of them
fails loudly. They complement the behavioral suites in this directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
from pathlib import Path

from moma_lineage_analysis.paths import PACKAGE_MODELS_RUN_DIR

REPO_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

RUN_DIR = PACKAGE_MODELS_RUN_DIR
PROMOTED_CHECKPOINT = RUN_DIR / "best_model.pt"
# SHA-256 recorded at promotion time (R010 CNN checkpoint, run merged_20261017,
# promoted 2026-10-17 as the validation-only rule winner; previous R008 CNN
# backed up under promotion_backups/20261017T124420.980675Z).
PROMOTED_SHA256 = "6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b"
PROMOTED_RUN = RUN_DIR / "no_longruns_20260916"
VERIFIED_RUN = REPO_ROOT / "datasets" / "verified_20260916"
METHODS_FILE = REPO_ROOT / "docs" / "METHODS_TRAJECTORY_CLEANING_MODEL.md"

MANIFEST_RE = re.compile(
    r"<!-- METHOD_MANIFEST_START -->\s*```json\s*(.*?)\s*```\s*<!-- METHOD_MANIFEST_END -->",
    re.DOTALL,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_manifest(text: str) -> dict:
    match = MANIFEST_RE.search(text)
    if match is None:
        raise AssertionError("METHOD_MANIFEST block not found in Methods contract")
    return json.loads(match.group(1))


class PromotedCheckpointTests(unittest.TestCase):
    def test_promoted_checkpoint_hash_matches_record(self) -> None:
        self.assertTrue(PROMOTED_CHECKPOINT.exists(), "promoted checkpoint missing")
        self.assertEqual(_sha256(PROMOTED_CHECKPOINT), PROMOTED_SHA256)

    def test_checkpoint_schema_and_internal_consistency(self) -> None:
        from moma_lineage_analysis.trajectory_cleaning import load_checkpoint

        model, normalizer, thresholds, metadata = load_checkpoint(PROMOTED_CHECKPOINT)
        import numpy as np

        self.assertIsInstance(thresholds, np.ndarray)
        self.assertEqual(thresholds.shape, (2,))
        for value in thresholds:
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0)
        self.assertIsNotNone(normalizer)
        self.assertIsInstance(metadata, dict)

    def test_promoted_thresholds_match_run_metrics(self) -> None:
        from moma_lineage_analysis.trajectory_cleaning import load_checkpoint

        metrics_path = PROMOTED_RUN / "cnn" / "metrics.json"
        if not metrics_path.exists():
            self.skipTest("no_longruns_20260916/cnn/metrics.json not present")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _, _, thresholds, _ = load_checkpoint(PROMOTED_CHECKPOINT)
        metric_thresholds = metrics.get("thresholds") or metrics.get("best_thresholds")
        self.assertIsNotNone(metric_thresholds, "metrics.json has no thresholds record")
        # checkpoint threshold order follows LABEL_COLUMNS: division_point, excluded_point
        self.assertAlmostEqual(
            float(thresholds[0]), float(metric_thresholds["division_point"]), places=6
        )
        self.assertAlmostEqual(
            float(thresholds[1]), float(metric_thresholds["excluded_point"]), places=6
        )


class RunArtifactTests(unittest.TestCase):
    def test_split_partitions_are_disjoint_and_complete(self) -> None:
        splits_path = VERIFIED_RUN / "cnn" / "splits.json"
        if not splits_path.exists():
            self.skipTest("splits.json not present")
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        train, val, test = (set(splits[name]) for name in ("train", "validation", "test"))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(len(train), 356)
        self.assertEqual(len(val), 77)
        self.assertEqual(len(test), 77)

    def test_dataset_manifest_schema(self) -> None:
        manifest_path = VERIFIED_RUN / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertTrue(manifest["human_verified"])
        self.assertEqual(
            manifest["canonical_target_columns"], ["division_point", "excluded_point"]
        )
        self.assertEqual(manifest["feature_columns"], ["time_h", "spine_length"])


class MethodsContractTests(unittest.TestCase):
    def test_manifest_block_parses(self) -> None:
        manifest = _extract_manifest(METHODS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(manifest["analysis_id"], "trajectory_cleaning_model")
        self.assertRegex(manifest["method_version"], r"^\d+\.\d+\.\d+$")

    def test_manifest_entry_points_exist(self) -> None:
        manifest = _extract_manifest(METHODS_FILE.read_text(encoding="utf-8"))
        for entry_point in manifest["entry_points"]:
            path = REPO_ROOT / entry_point.split(" ")[0]
            self.assertTrue(path.exists(), f"declared entry point missing: {entry_point}")

    def test_manifest_declared_inputs_exist(self) -> None:
        manifest = _extract_manifest(METHODS_FILE.read_text(encoding="utf-8"))
        for declared_input in manifest["inputs"]:
            self.assertTrue(
                (REPO_ROOT / declared_input).exists(),
                f"declared input missing: {declared_input}",
            )


if __name__ == "__main__":
    unittest.main()
