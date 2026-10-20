"""Focused stdlib tests for the Streamlit trajectory-cleaning adapter."""

from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from moma_lineage_analysis.trajectory_cleaning.streamlit_api import (
    DIVISION_RESOLUTION_STATUSES,
    PREDICTION_COLUMNS,
    apply_ai_predictions,
    attach_ai_predictions_to_export,
    build_scope_dataframe,
    capture_cleaning_snapshot,
    predict_trajectory_scope,
    preflight_ai_trajectory_inputs,
    resolve_division_exclusion_conflicts,
    restore_cleaning_snapshot,
    suggested_indices,
)


class ProbabilityLogitNormalizer:
    """Encode the two source values as logits for the deterministic CNN."""

    def transform(self, time, spine_length) -> np.ndarray:
        probability = np.column_stack([time, spine_length]).astype(np.float64)
        if np.any((probability <= 0.0) | (probability >= 1.0)):
            raise ValueError("Test probabilities must be strictly between zero and one")
        return np.log(probability / (1.0 - probability)).astype(np.float32)


class IdentityPointwiseCNN(nn.Module):
    """A deterministic 1x1 CNN whose sigmoid output echoes source values."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(2, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.convolution.weight.zero_()
            self.convolution.weight[0, 0, 0] = 1.0
            self.convolution.weight[1, 1, 0] = 1.0

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del mask
        return self.convolution(features.transpose(1, 2)).transpose(1, 2)


def make_trajectory_dict() -> dict[str, dict[str, object]]:
    trajectory_a = pd.DataFrame(
        {
            # Deliberately not in time order: prediction results must map by row ID.
            "fov_cell_index": [103, 101, 102],
            "time_h": [0.9, 0.1, 0.6],
            "spine_length": [0.9, 0.7, 0.2],
            "source_value": ["a3", "a1", "a2"],
        },
        index=[8, 3, 5],
    )
    trajectory_b = pd.DataFrame(
        {
            "fov_cell_index": [202, 201],
            "time_h": [0.8, 0.3],
            "spine_length": [0.4, 0.8],
            "source_value": ["b2", "b1"],
        },
        index=[12, 11],
    )
    return {
        "trajectory_a": {
            "trajectory": trajectory_a,
            "div_indices": [103, 101],
            "excluded_indices": {102},
            "unrelated_state": {"keep": True},
        },
        "trajectory_b": {
            "trajectory": trajectory_b,
            "div_indices": [202],
            "excluded_indices": {201},
            "unrelated_state": {"keep": True},
        },
    }


def probability_prediction(
    frame: pd.DataFrame, trajectory_key: str | None = None
) -> pd.DataFrame:
    prediction = frame.loc[
        :, ["fov_cell_index", "time_h", "spine_length"]
    ].copy()
    prediction["pred_division_probability"] = prediction["time_h"]
    prediction["pred_excluded_probability"] = prediction["spine_length"]
    prediction["pred_division_point"] = (
        prediction["pred_division_probability"] >= 0.5
    )
    prediction["pred_excluded_point"] = (
        prediction["pred_excluded_probability"] >= 0.65
    )
    if trajectory_key is not None:
        prediction["trajectory_key"] = trajectory_key
    return prediction


class AIInputPreflightTests(unittest.TestCase):
    def test_nonfinite_time_is_invalid_but_negative_finite_time_is_retained(self) -> None:
        frame = pd.DataFrame(
            {
                "fov_cell_index": ["negative", "zero", "nan", "positive_inf", "negative_inf"],
                "time_h": [-2.0, 0.0, np.nan, np.inf, -np.inf],
                "spine_length": [2.0, 2.5, 3.0, 3.5, 4.0],
            }
        )
        original = frame.copy(deep=True)

        result = preflight_ai_trajectory_inputs(frame)

        self.assertEqual(
            result.valid_trajectory["fov_cell_index"].tolist(),
            ["negative", "zero"],
        )
        self.assertEqual(
            result.invalid_indices,
            frozenset({"nan", "positive_inf", "negative_inf"}),
        )
        decisions = result.decisions.set_index("fov_cell_index")
        self.assertTrue(bool(decisions.loc["negative", "ai_input_valid"]))
        self.assertEqual(
            decisions.loc["negative", "ai_input_invalid_reason"], ""
        )
        for cell_id in ("nan", "positive_inf", "negative_inf"):
            self.assertFalse(bool(decisions.loc[cell_id, "ai_input_valid"]))
            self.assertEqual(
                decisions.loc[cell_id, "ai_input_invalid_reason"],
                "nonfinite_time",
            )
        pd.testing.assert_frame_equal(frame, original)

    def test_nonfinite_and_nonpositive_lengths_are_invalid(self) -> None:
        frame = pd.DataFrame(
            {
                "fov_cell_index": ["nan", "positive_inf", "negative_inf", "zero", "negative", "valid"],
                "time_h": np.arange(6, dtype=float),
                "spine_length": [np.nan, np.inf, -np.inf, 0.0, -1.0, 2.0],
            }
        )
        original = frame.copy(deep=True)

        result = preflight_ai_trajectory_inputs(frame)

        self.assertEqual(
            result.valid_trajectory["fov_cell_index"].tolist(), ["valid"]
        )
        self.assertEqual(
            result.invalid_indices,
            frozenset({"nan", "positive_inf", "negative_inf", "zero", "negative"}),
        )
        decisions = result.decisions.set_index("fov_cell_index")
        for cell_id in ("nan", "positive_inf", "negative_inf"):
            self.assertEqual(
                decisions.loc[cell_id, "ai_input_invalid_reason"],
                "nonfinite_spine_length",
            )
        for cell_id in ("zero", "negative"):
            self.assertEqual(
                decisions.loc[cell_id, "ai_input_invalid_reason"],
                "nonpositive_spine_length",
            )
        self.assertFalse(result.inference_ready)
        pd.testing.assert_frame_equal(frame, original)

    def test_all_rows_at_an_ambiguous_duplicate_time_are_invalid(self) -> None:
        frame = pd.DataFrame(
            {
                "fov_cell_index": ["early", "duplicate_a", "duplicate_b", "late"],
                "time_h": [-1.0, 0.5, 0.5, 2.0],
                "spine_length": [2.0, 3.0, 4.0, 5.0],
            }
        )
        original = frame.copy(deep=True)

        result = preflight_ai_trajectory_inputs(frame)

        self.assertEqual(
            result.valid_trajectory["fov_cell_index"].tolist(), ["early", "late"]
        )
        self.assertEqual(
            result.invalid_indices, frozenset({"duplicate_a", "duplicate_b"})
        )
        decisions = result.decisions.set_index("fov_cell_index")
        for cell_id in ("duplicate_a", "duplicate_b"):
            self.assertFalse(bool(decisions.loc[cell_id, "ai_input_valid"]))
            self.assertEqual(
                decisions.loc[cell_id, "ai_input_invalid_reason"],
                "duplicate_time",
            )
        self.assertTrue(result.inference_ready)
        pd.testing.assert_frame_equal(frame, original)


class ScopeAndInferenceTests(unittest.TestCase):
    def test_scope_concatenates_requested_keys_without_mutating_inputs(self) -> None:
        trajectories = make_trajectory_dict()
        original_a = trajectories["trajectory_a"]["trajectory"].copy(deep=True)
        original_b = trajectories["trajectory_b"]["trajectory"].copy(deep=True)

        scope = build_scope_dataframe(
            trajectories, ["trajectory_b", "trajectory_a"]
        )

        self.assertEqual(
            scope["trajectory_key"].tolist(),
            ["trajectory_b", "trajectory_b", "trajectory_a", "trajectory_a", "trajectory_a"],
        )
        self.assertEqual(scope["fov_cell_index"].tolist(), [202, 201, 103, 101, 102])
        self.assertIsInstance(scope.index, pd.RangeIndex)
        pd.testing.assert_frame_equal(
            trajectories["trajectory_a"]["trajectory"], original_a
        )
        pd.testing.assert_frame_equal(
            trajectories["trajectory_b"]["trajectory"], original_b
        )
        self.assertNotIn("trajectory_key", original_a.columns)

    def test_scope_rejects_conflicts_duplicates_and_missing_columns(self) -> None:
        conflicting = make_trajectory_dict()
        conflicting_frame = conflicting["trajectory_a"]["trajectory"]
        conflicting_frame["trajectory_key"] = ["trajectory_a", "wrong", "trajectory_a"]
        with self.assertRaisesRegex(ValueError, "conflicting trajectory_key"):
            build_scope_dataframe(conflicting, "trajectory_a")

        duplicate_ids = make_trajectory_dict()
        duplicate_ids["trajectory_a"]["trajectory"].loc[:, "fov_cell_index"] = [1, 1, 2]
        with self.assertRaisesRegex(ValueError, "duplicate fov_cell_index"):
            build_scope_dataframe(duplicate_ids, "trajectory_a")

        missing_column = make_trajectory_dict()
        missing_column["trajectory_b"]["trajectory"] = missing_column[
            "trajectory_b"
        ]["trajectory"].drop(columns="spine_length")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_scope_dataframe(missing_column, "trajectory_b")

        with self.assertRaisesRegex(ValueError, "Duplicate trajectory key"):
            build_scope_dataframe(make_trajectory_dict(), ["trajectory_a", "trajectory_a"])

    def test_prediction_probabilities_map_back_to_shuffled_source_rows(self) -> None:
        trajectories = make_trajectory_dict()
        original = {
            key: entry["trajectory"].copy(deep=True)
            for key, entry in trajectories.items()
        }
        predictions = predict_trajectory_scope(
            IdentityPointwiseCNN(),
            ProbabilityLogitNormalizer(),
            (0.5, 0.65),
            trajectories,
            ["trajectory_b", "trajectory_a"],
            device="cpu",
            batch_size=2,
        )

        self.assertEqual(list(predictions), ["trajectory_b", "trajectory_a"])
        for key in predictions:
            source = trajectories[key]["trajectory"]
            predicted = predictions[key]
            self.assertEqual(
                predicted["fov_cell_index"].tolist(),
                source["fov_cell_index"].tolist(),
            )
            np.testing.assert_allclose(
                predicted["pred_division_probability"], source["time_h"], atol=1e-6
            )
            np.testing.assert_allclose(
                predicted["pred_excluded_probability"],
                source["spine_length"],
                atol=1e-6,
            )
        self.assertEqual(
            predictions["trajectory_a"]["pred_division_point"].tolist(),
            [True, False, True],
        )
        self.assertEqual(
            predictions["trajectory_a"]["pred_excluded_point"].tolist(),
            [True, True, False],
        )
        for key, frame in original.items():
            pd.testing.assert_frame_equal(trajectories[key]["trajectory"], frame)


class SuggestionAndMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectories = make_trajectory_dict()
        self.prediction_a = probability_prediction(
            self.trajectories["trajectory_a"]["trajectory"], "trajectory_a"
        )

    def test_threshold_suggestions_keep_overlapping_labels(self) -> None:
        original = self.prediction_a.copy(deep=True)
        divisions, exclusions = suggested_indices(self.prediction_a, 0.5, 0.65)
        self.assertEqual(divisions, {102, 103})
        self.assertEqual(exclusions, {101, 103})
        self.assertEqual(divisions & exclusions, {103})
        pd.testing.assert_frame_equal(self.prediction_a, original)

    def test_merge_and_replace_have_distinct_label_semantics(self) -> None:
        merge_snapshot = apply_ai_predictions(
            self.trajectories,
            {"trajectory_a": self.prediction_a},
            policy="merge",
            division_threshold=0.5,
            excluded_threshold=0.65,
        )
        self.assertEqual(merge_snapshot["trajectory_a"]["div_indices"], {101, 103})
        self.assertEqual(
            self.trajectories["trajectory_a"]["div_indices"], [103, 101, 102]
        )
        self.assertEqual(
            self.trajectories["trajectory_a"]["excluded_indices"], {101, 102, 103}
        )

        restore_cleaning_snapshot(self.trajectories, merge_snapshot)
        apply_ai_predictions(
            self.trajectories,
            {"trajectory_a": self.prediction_a},
            policy="replace",
            division_threshold=0.5,
            excluded_threshold=0.65,
        )
        self.assertEqual(self.trajectories["trajectory_a"]["div_indices"], [103, 102])
        self.assertEqual(
            self.trajectories["trajectory_a"]["excluded_indices"], {101, 103}
        )

    def test_apply_is_atomic_when_a_later_prediction_is_invalid(self) -> None:
        prediction_b = probability_prediction(
            self.trajectories["trajectory_b"]["trajectory"], "trajectory_b"
        ).iloc[:1].copy()
        prediction_a_before = self.prediction_a.copy(deep=True)
        prediction_b_before = prediction_b.copy(deep=True)
        state_before = capture_cleaning_snapshot(
            self.trajectories, ["trajectory_a", "trajectory_b"]
        )
        frames_before = {
            key: entry["trajectory"].copy(deep=True)
            for key, entry in self.trajectories.items()
        }

        with self.assertRaisesRegex(ValueError, "do not cover exactly"):
            apply_ai_predictions(
                self.trajectories,
                {
                    "trajectory_a": self.prediction_a,
                    "trajectory_b": prediction_b,
                },
                ["trajectory_a", "trajectory_b"],
                policy="replace",
            )

        self.assertEqual(
            capture_cleaning_snapshot(
                self.trajectories, ["trajectory_a", "trajectory_b"]
            ),
            state_before,
        )
        for key, frame in frames_before.items():
            pd.testing.assert_frame_equal(self.trajectories[key]["trajectory"], frame)
        pd.testing.assert_frame_equal(self.prediction_a, prediction_a_before)
        pd.testing.assert_frame_equal(prediction_b, prediction_b_before)

    def test_apply_rejects_foreign_prediction_when_local_ids_overlap(self) -> None:
        foreign_source = self.trajectories["trajectory_a"]["trajectory"].copy()
        foreign_source.loc[:, "time_h"] = [0.2, 0.4, 0.8]
        foreign_prediction = probability_prediction(foreign_source, "trajectory_b")
        before = capture_cleaning_snapshot(self.trajectories, "trajectory_a")

        with self.assertRaisesRegex(ValueError, "different trajectory_key"):
            apply_ai_predictions(
                self.trajectories,
                {"trajectory_a": foreign_prediction},
                policy="replace",
            )

        spoofed_prediction = foreign_prediction.copy()
        spoofed_prediction.loc[:, "trajectory_key"] = "trajectory_a"
        with self.assertRaisesRegex(ValueError, "time_h"):
            apply_ai_predictions(
                self.trajectories,
                {"trajectory_a": spoofed_prediction},
                policy="replace",
            )

        self.assertEqual(
            capture_cleaning_snapshot(self.trajectories, "trajectory_a"), before
        )

    def test_snapshot_restores_exact_labels_and_leaves_other_state_untouched(self) -> None:
        labels_before = deepcopy(
            {
                key: {
                    "div_indices": entry["div_indices"],
                    "excluded_indices": entry["excluded_indices"],
                }
                for key, entry in self.trajectories.items()
            }
        )
        unrelated_before = deepcopy(
            {
                key: entry["unrelated_state"]
                for key, entry in self.trajectories.items()
            }
        )
        snapshot = capture_cleaning_snapshot(
            self.trajectories, ["trajectory_a", "trajectory_b"]
        )
        predictions = {
            key: probability_prediction(entry["trajectory"], key)
            for key, entry in self.trajectories.items()
        }
        apply_ai_predictions(self.trajectories, predictions, policy="replace")
        restore_cleaning_snapshot(self.trajectories, snapshot)

        for key in self.trajectories:
            self.assertEqual(
                self.trajectories[key]["div_indices"], labels_before[key]["div_indices"]
            )
            self.assertEqual(
                self.trajectories[key]["excluded_indices"],
                labels_before[key]["excluded_indices"],
            )
            self.assertEqual(
                self.trajectories[key]["unrelated_state"], unrelated_before[key]
            )
        self.assertEqual(snapshot["trajectory_a"]["div_indices"], {101, 103})


class DivisionConflictResolutionTests(unittest.TestCase):
    @staticmethod
    def make_frame(
        lengths: list[float],
        *,
        ids: list[str] | None = None,
    ) -> pd.DataFrame:
        if ids is None:
            ids = [f"p{position}" for position in range(len(lengths))]
        return pd.DataFrame(
            {
                "fov_cell_index": ids,
                "time_h": np.arange(len(lengths), dtype=float),
                "spine_length": lengths,
            }
        )

    def test_excluded_run_shifts_once_with_length_drop(self) -> None:
        frame = self.make_frame([4.0, 20.0, 18.0, 2.96])
        raw_divisions = {"p1", "p2"}
        raw_exclusions = {"p1", "p2"}
        frame_before = frame.copy(deep=True)
        divisions_before = set(raw_divisions)
        exclusions_before = set(raw_exclusions)

        result = resolve_division_exclusion_conflicts(
            frame,
            raw_divisions,
            raw_exclusions,
        )

        self.assertEqual(result.raw_division_indices, frozenset({"p1", "p2"}))
        self.assertEqual(result.raw_excluded_indices, frozenset({"p1", "p2"}))
        self.assertEqual(result.effective_division_indices, ("p0",))
        self.assertEqual(
            result.decisions["source_index"].tolist(), ["p1", "p2"]
        )
        self.assertEqual(
            result.decisions["effective_index"].tolist(), ["p0", "p0"]
        )
        self.assertEqual(
            result.decisions["division_resolution"].tolist(),
            ["shifted_previous", "coalesced_previous"],
        )
        np.testing.assert_allclose(result.decisions["length_ratio"], [0.74, 0.74])
        self.assertEqual(
            result.summary_counts,
            {
                "unchanged": 0,
                "coalesced_previous": 1,
                "shifted_previous": 1,
                "needs_review": 0,
            },
        )
        pd.testing.assert_frame_equal(frame, frame_before)
        self.assertEqual(raw_divisions, divisions_before)
        self.assertEqual(raw_exclusions, exclusions_before)

    def test_threshold_is_strict_and_false_spikes_need_review(self) -> None:
        for ratio in (0.75, 0.76, 1.10):
            with self.subTest(ratio=ratio):
                frame = self.make_frame([4.0, 20.0, 4.0 * ratio])
                result = resolve_division_exclusion_conflicts(
                    frame,
                    {"p1"},
                    {"p1"},
                )

                self.assertEqual(result.effective_division_indices, ())
                self.assertEqual(
                    result.decisions["division_resolution"].tolist(),
                    ["needs_review"],
                )
                self.assertIsNone(result.decisions["effective_index"].iloc[0])
                self.assertAlmostEqual(
                    result.decisions["length_ratio"].iloc[0], ratio
                )

    def test_previous_valid_raw_division_coalesces_without_next_neighbor(self) -> None:
        frame = self.make_frame([4.0, 10.0, 25.0], ids=["mother", "gap", "source"])

        result = resolve_division_exclusion_conflicts(
            frame,
            {"mother", "source"},
            {"gap", "source"},
        )

        self.assertEqual(result.effective_division_indices, ("mother",))
        self.assertEqual(
            result.decisions["division_resolution"].tolist(),
            ["unchanged", "coalesced_previous"],
        )
        source_decision = result.decisions.iloc[1]
        self.assertEqual(source_decision["effective_index"], "mother")
        self.assertEqual(source_decision["previous_valid_index"], "mother")
        self.assertIsNone(source_decision["next_valid_index"])
        self.assertTrue(np.isnan(source_decision["length_ratio"]))

    def test_adjacent_run_prefers_earliest_non_excluded_source(self) -> None:
        frame = self.make_frame([8.0, 4.0, 3.0, 2.0])

        result = resolve_division_exclusion_conflicts(
            frame,
            {"p0", "p1", "p2"},
            {"p0"},
        )

        self.assertEqual(result.effective_division_indices, ("p1",))
        self.assertEqual(
            result.decisions["effective_index"].tolist(), ["p1", "p1", "p1"]
        )
        self.assertEqual(
            result.decisions["division_resolution"].tolist(),
            ["coalesced_previous", "unchanged", "coalesced_previous"],
        )

    def test_later_conflict_coalesces_to_prior_runs_effective_boundary(self) -> None:
        frame = self.make_frame([8.0, 7.0, 30.0, 25.0])

        result = resolve_division_exclusion_conflicts(
            frame,
            {"p0", "p1", "p3"},
            {"p2", "p3"},
        )

        self.assertEqual(result.effective_division_indices, ("p0",))
        self.assertEqual(
            result.decisions["effective_index"].tolist(), ["p0", "p0", "p0"]
        )
        self.assertEqual(
            result.decisions["division_resolution"].tolist(),
            ["unchanged", "coalesced_previous", "coalesced_previous"],
        )

    def test_missing_previous_neighbor_requires_review(self) -> None:
        frame = self.make_frame([20.0, 2.0, 3.0])

        result = resolve_division_exclusion_conflicts(
            frame,
            {"p0"},
            {"p0"},
        )

        self.assertEqual(result.effective_division_indices, ())
        self.assertEqual(
            result.decisions["division_resolution"].tolist(), ["needs_review"]
        )
        self.assertIsNone(result.decisions["previous_valid_index"].iloc[0])

    def test_nonfinite_time_division_needs_review_with_or_without_exclusion(self) -> None:
        frame = pd.DataFrame(
            {
                "fov_cell_index": ["early", "invalid_time", "late"],
                "time_h": [0.0, np.nan, 2.0],
                "spine_length": [4.0, 20.0, 2.0],
            }
        )
        original = frame.copy(deep=True)

        unexcluded = resolve_division_exclusion_conflicts(
            frame,
            {"invalid_time"},
            set(),
        )
        self.assertEqual(unexcluded.raw_excluded_indices, frozenset())
        self.assertEqual(unexcluded.effective_division_indices, ())
        self.assertEqual(
            unexcluded.decisions["division_resolution"].tolist(),
            ["needs_review"],
        )

        result = resolve_division_exclusion_conflicts(
            frame,
            {"invalid_time"},
            {"invalid_time"},
        )

        self.assertEqual(result.raw_division_indices, frozenset({"invalid_time"}))
        self.assertEqual(result.raw_excluded_indices, frozenset({"invalid_time"}))
        self.assertEqual(result.effective_division_indices, ())
        self.assertEqual(
            result.decisions["source_index"].tolist(), ["invalid_time"]
        )
        self.assertEqual(
            result.decisions["division_resolution"].tolist(), ["needs_review"]
        )
        decision = result.decisions.iloc[0]
        self.assertIsNone(decision["effective_index"])
        self.assertIsNone(decision["previous_valid_index"])
        self.assertIsNone(decision["next_valid_index"])
        self.assertTrue(np.isnan(decision["length_ratio"]))
        pd.testing.assert_frame_equal(frame, original)

    def test_previous_and_next_are_determined_by_time_not_dataframe_order(self) -> None:
        frame = pd.DataFrame(
            {
                "fov_cell_index": ["late", "early", "source"],
                "time_h": [2.0, 0.0, 1.0],
                "spine_length": [2.0, 4.0, 20.0],
            }
        )

        result = resolve_division_exclusion_conflicts(
            frame,
            {"source"},
            {"source"},
        )

        self.assertEqual(result.effective_division_indices, ("early",))
        decision = result.decisions.iloc[0]
        self.assertEqual(decision["previous_valid_index"], "early")
        self.assertEqual(decision["next_valid_index"], "late")
        self.assertEqual(decision["division_resolution"], "shifted_previous")

    def test_recomputing_after_exclusion_reset_restores_raw_boundary(self) -> None:
        frame = self.make_frame([4.0, 18.0, 2.0])

        excluded_result = resolve_division_exclusion_conflicts(
            frame,
            {"p1"},
            {"p1"},
        )
        reset_result = resolve_division_exclusion_conflicts(
            frame,
            {"p1"},
            set(),
        )

        self.assertEqual(excluded_result.effective_division_indices, ("p0",))
        self.assertEqual(
            excluded_result.decisions["division_resolution"].tolist(),
            ["shifted_previous"],
        )
        self.assertEqual(reset_result.effective_division_indices, ("p1",))
        self.assertEqual(
            reset_result.decisions["division_resolution"].tolist(), ["unchanged"]
        )
        self.assertEqual(reset_result.raw_division_indices, frozenset({"p1"}))
        self.assertEqual(reset_result.raw_excluded_indices, frozenset())

    def test_empty_division_set_has_stable_empty_audit_schema(self) -> None:
        result = resolve_division_exclusion_conflicts(
            self.make_frame([1.0, 2.0]),
            set(),
            {"p1"},
        )

        self.assertEqual(result.effective_division_indices, ())
        self.assertEqual(
            result.decisions.columns.tolist(),
            [
                "source_index",
                "effective_index",
                "division_resolution",
                "previous_valid_index",
                "next_valid_index",
                "length_ratio",
            ],
        )
        self.assertTrue(result.decisions.empty)
        self.assertEqual(
            result.summary_counts,
            {status: 0 for status in DIVISION_RESOLUTION_STATUSES},
        )


class ExportTests(unittest.TestCase):
    def test_export_alignment_provenance_and_input_immutability(self) -> None:
        trajectories = make_trajectory_dict()
        prediction = probability_prediction(
            trajectories["trajectory_a"]["trajectory"]
        ).sort_values("fov_cell_index", ascending=True)
        export = trajectories["trajectory_a"]["trajectory"].iloc[[2, 0, 1]].copy()
        export_before = export.copy(deep=True)
        prediction_before = prediction.copy(deep=True)
        provenance_value = {"format": 1}

        result = attach_ai_predictions_to_export(
            export,
            prediction,
            provenance={
                "ai_checkpoint": "deterministic-test.pt",
                "ai_division_threshold": 0.5,
                "ai_metadata": provenance_value,
            },
        )

        self.assertEqual(result["fov_cell_index"].tolist(), [102, 103, 101])
        np.testing.assert_allclose(
            result["pred_division_probability"], [0.6, 0.9, 0.1]
        )
        np.testing.assert_allclose(
            result["pred_excluded_probability"], [0.2, 0.9, 0.7]
        )
        self.assertEqual(result["pred_division_point"].tolist(), [True, True, False])
        self.assertEqual(result["pred_excluded_point"].tolist(), [False, True, True])
        self.assertEqual(result["ai_checkpoint"].unique().tolist(), ["deterministic-test.pt"])
        self.assertEqual(result["ai_division_threshold"].unique().tolist(), [0.5])
        self.assertTrue(all(value == provenance_value for value in result["ai_metadata"]))
        self.assertIsNot(result["ai_metadata"].iloc[0], result["ai_metadata"].iloc[1])
        pd.testing.assert_frame_equal(export, export_before)
        pd.testing.assert_frame_equal(prediction, prediction_before)
        self.assertNotIn("pred_division_probability", export.columns)
        for column in PREDICTION_COLUMNS:
            self.assertIn(column, result.columns)

    def test_export_rejects_wrong_scope_or_coordinates(self) -> None:
        trajectories = make_trajectory_dict()
        export = trajectories["trajectory_a"]["trajectory"].copy()
        export["trajectory_key"] = "trajectory_a"
        prediction = probability_prediction(export)
        prediction["trajectory_key"] = "trajectory_a"

        extra = prediction.iloc[[0]].copy()
        extra["fov_cell_index"] = 999
        with self.assertRaisesRegex(ValueError, "exactly the same"):
            attach_ai_predictions_to_export(
                export,
                pd.concat([prediction, extra], ignore_index=True),
            )

        wrong_key = prediction.copy()
        wrong_key["trajectory_key"] = "trajectory_b"
        with self.assertRaisesRegex(ValueError, "same single trajectory_key"):
            attach_ai_predictions_to_export(export, wrong_key)

        wrong_time = prediction.copy()
        wrong_time.loc[wrong_time.index[0], "time_h"] += 0.01
        with self.assertRaisesRegex(ValueError, "'time_h' values do not match"):
            attach_ai_predictions_to_export(export, wrong_time)


if __name__ == "__main__":
    unittest.main()
