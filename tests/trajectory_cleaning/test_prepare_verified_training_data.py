"""Tests for preparing human-verified trajectory training data."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from moma_lineage_analysis.trajectory_cleaning.prepare_verified_training_data import (
    OUTPUT_COLUMNS,
    build_verified_training_dataset,
    main,
    prepare_verified_training_data,
)


def source_frame(key: str = "shared") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trajectory_key": [key, key, key],
            "fov_cell_index": [10, 11, 12],
            "time_h": [0.0, 1.0, 2.0],
            "spine_length": [2.0, 3.0, 1.7],
            "division_point": ["false", "true", "false"],
            "excluded_point": [0, 0, 1],
            "pred_division_point": [True, False, True],
            "ai_excluded_point": [True, True, False],
            "effective_division_point": [True, False, True],
            "division_conflict_status": ["ai", "ai", "ai"],
        }
    )


class VerifiedDatasetPreparationTests(unittest.TestCase):
    def test_only_canonical_labels_are_copied_and_keys_are_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            source_frame().to_csv(first, index=False)
            altered = source_frame()
            altered["spine_length"] += 0.2
            altered.to_csv(second, index=False)

            prepared = build_verified_training_dataset(
                [first, second],
                dataset_ids=["condition_a", "condition_b"],
                verification_basis="user_attested_test",
            )

            self.assertEqual(list(prepared.dataframe.columns), list(OUTPUT_COLUMNS))
            self.assertEqual(
                set(prepared.dataframe["trajectory_key"]),
                {"condition_a::shared", "condition_b::shared"},
            )
            first_rows = prepared.dataframe[
                prepared.dataframe["source_dataset"] == "condition_a"
            ]
            self.assertEqual(first_rows["division_point"].tolist(), [False, True, False])
            self.assertEqual(first_rows["excluded_point"].tolist(), [False, False, True])
            self.assertTrue(prepared.manifest["human_verified"])
            self.assertEqual(
                prepared.manifest["namespace"][
                    "raw_keys_present_in_multiple_sources"
                ],
                1,
            )
            forbidden = ("pred_", "ai_", "effective_")
            self.assertFalse(
                any(column.startswith(forbidden) for column in prepared.dataframe)
            )
            self.assertFalse(
                any(
                    "division" in column and column != "division_point"
                    for column in prepared.dataframe
                )
            )

    def test_preferred_source_supersedes_only_shared_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            older = root / "older.csv"
            preferred = root / "preferred.csv"
            older_frame = pd.concat(
                [source_frame("shared"), source_frame("older_only")],
                ignore_index=True,
            )
            preferred_shared = source_frame("shared")
            preferred_shared["division_point"] = [True, False, False]
            preferred_frame = pd.concat(
                [preferred_shared, source_frame("preferred_only")],
                ignore_index=True,
            )
            older_frame.to_csv(older, index=False)
            preferred_frame.to_csv(preferred, index=False)

            prepared = build_verified_training_dataset(
                [older, preferred],
                dataset_ids=["older", "preferred"],
                supersessions=["older=preferred"],
                verification_basis="reviewed",
            )

            self.assertEqual(len(prepared.dataframe), 9)
            self.assertEqual(
                set(prepared.dataframe["trajectory_key"]),
                {
                    "older::older_only",
                    "preferred::shared",
                    "preferred::preferred_only",
                },
            )
            shared = prepared.dataframe[
                prepared.dataframe["trajectory_key"] == "preferred::shared"
            ]
            self.assertEqual(
                shared["division_point"].tolist(), [True, False, False]
            )
            self.assertEqual(
                prepared.manifest["summary"]["omitted_superseded_rows"], 3
            )
            self.assertEqual(
                prepared.manifest["summary"][
                    "omitted_superseded_trajectories"
                ],
                1,
            )
            rule = prepared.manifest["supersession"]["rules"][0]
            self.assertEqual(rule["superseded_dataset"], "older")
            self.assertEqual(rule["preferred_dataset"], "preferred")
            self.assertEqual(rule["trajectory_keys"], ["shared"])
            self.assertEqual(
                prepared.manifest["namespace"][
                    "raw_keys_present_in_multiple_sources"
                ],
                0,
            )

    def test_invalid_supersession_rules_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            source_frame("first").to_csv(first, index=False)
            source_frame("second").to_csv(second, index=False)
            arguments = {
                "sources": [first, second],
                "dataset_ids": ["first", "second"],
                "verification_basis": "reviewed",
            }
            with self.assertRaisesRegex(ValueError, "unknown dataset IDs"):
                build_verified_training_dataset(
                    **arguments, supersessions=["missing=second"]
                )
            with self.assertRaisesRegex(ValueError, "cycle"):
                build_verified_training_dataset(
                    **arguments,
                    supersessions=["first=second", "second=first"],
                )

    def test_identical_source_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "one.csv"
            duplicate = root / "copy.csv"
            source_frame().to_csv(first, index=False)
            shutil.copyfile(first, duplicate)
            with self.assertRaisesRegex(ValueError, "double-counted"):
                build_verified_training_dataset(
                    [first, duplicate],
                    verification_basis="reviewed",
                )

    def test_invalid_features_are_omitted_and_described_without_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.csv"
            dataframe = pd.concat(
                [source_frame(), source_frame("other")], ignore_index=True
            )
            dataframe.loc[0, "spine_length"] = 0
            dataframe["time_h"] = dataframe["time_h"].astype(object)
            dataframe.loc[1, "time_h"] = "not-a-time"
            dataframe.loc[3, "spine_length"] = np.inf
            dataframe.to_csv(path, index=False)

            prepared = build_verified_training_dataset(
                [path], verification_basis="reviewed"
            )

            self.assertEqual(len(prepared.dataframe), 3)
            self.assertEqual(
                prepared.manifest["summary"]["omitted_invalid_feature_rows"], 3
            )
            details = prepared.manifest["invalid_feature_rows"]
            self.assertEqual([item["csv_line_number"] for item in details], [2, 3, 5])
            self.assertEqual(
                prepared.manifest["summary"]["retained_label_counts"],
                {
                    "division_point": 1,
                    "excluded_point": 2,
                    "division_and_excluded_overlap": 0,
                },
            )
            self.assertTrue(np.isfinite(prepared.dataframe["time_h"]).all())
            self.assertTrue((prepared.dataframe["spine_length"] > 0).all())

    def test_non_boolean_labels_and_duplicate_coordinates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_label = root / "invalid_label.csv"
            dataframe = source_frame()
            dataframe.loc[0, "division_point"] = "maybe"
            dataframe.to_csv(invalid_label, index=False)
            with self.assertRaisesRegex(ValueError, "non-boolean"):
                build_verified_training_dataset(
                    [invalid_label], verification_basis="reviewed"
                )

            duplicate_time = root / "duplicate_time.csv"
            dataframe = source_frame()
            dataframe.loc[1, "time_h"] = 0.0
            dataframe.to_csv(duplicate_time, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate time_h"):
                build_verified_training_dataset(
                    [duplicate_time], verification_basis="reviewed"
                )

            duplicate_id = root / "duplicate_id.csv"
            dataframe = source_frame()
            dataframe.loc[1, "fov_cell_index"] = 10
            dataframe.to_csv(duplicate_id, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate fov_cell_index"):
                build_verified_training_dataset(
                    [duplicate_id], verification_basis="reviewed"
                )

    def test_required_attestation_and_schema_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "source.csv"
            source_frame().drop(columns="excluded_point").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "verification_basis"):
                build_verified_training_dataset([path], verification_basis="  ")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                build_verified_training_dataset([path], verification_basis="reviewed")

    def test_writer_and_cli_create_csv_and_json_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            source_frame("trajectory").to_csv(source, index=False)
            output = root / "prepared" / "training.csv"
            manifest = root / "prepared" / "manifest.json"

            exit_code = main(
                [
                    "--source",
                    str(source),
                    "--dataset-id",
                    "verified",
                    "--output-csv",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--verification-basis",
                    "user_attested_test",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["retained_rows"], 3)
            self.assertEqual(payload["verification_basis"], "user_attested_test")
            self.assertEqual(len(payload["output"]["training_csv_sha256"]), 64)
            loaded = pd.read_csv(output)
            self.assertEqual(list(loaded.columns), list(OUTPUT_COLUMNS))

            # Exercise the direct writing API as well as the CLI wrapper.
            second_output = root / "direct.csv"
            second_manifest = root / "direct.json"
            direct = prepare_verified_training_data(
                [source],
                output_csv=second_output,
                manifest_path=second_manifest,
                verification_basis="direct_review",
            )
            self.assertEqual(len(direct.dataframe), 3)


if __name__ == "__main__":
    unittest.main()
