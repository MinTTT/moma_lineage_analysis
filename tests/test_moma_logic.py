"""Small, deterministic regressions for the public lineage-analysis API."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from moma_lineage_analysis.moma_logic import (
    compute_traj_stats,
    get_cell_lineage,
    get_one_trajectory,
    process_channel_worker,
    tracing_lineage,
)


def linked_cells(frames, division_threshold=0.8):
    """Link frames whose per-frame lengths are already in cell-index order."""
    time_point = np.concatenate(
        [[f"t{index}.tif"] * len(lengths) for index, lengths in enumerate(frames)]
    )
    cell_index = np.concatenate(
        [np.arange(len(lengths)) for lengths in frames]
    )
    fov_cell_index = np.arange(len(time_point))
    length = np.concatenate([np.asarray(lengths, dtype=float) for lengths in frames])
    ancestor = get_cell_lineage(
        [f"t{index}.tif" for index in range(len(frames))],
        time_point,
        fov_cell_index,
        cell_index,
        length,
        division_threshold=division_threshold,
    )
    return fov_cell_index, ancestor


def physical_cells(elapsed_hours=(0.0, 0.5, 1.0, 1.5, 2.0), unit="s"):
    factor = {"s": 3600.0, "min": 60.0, "h": 1.0, "d": 1.0 / 24.0}[unit]
    hours = np.asarray(elapsed_hours, dtype=float)
    return pd.DataFrame(
        {
            # A nonzero origin distinguishes relative-hour conversion from a
            # conversion that happens to work only for timestamps near zero.
            "time_s": 7200.0 + hours * factor,
            "time_point": [f"t{index}.tif" for index in range(len(hours))],
            "channels": "A",
            "cell_index": np.zeros(len(hours), dtype=int),
            "fov_cell_index": np.arange(len(hours)),
            "spine_length": 2.0 + hours,
        }
    )


class DivisionRegressionTests(unittest.TestCase):
    def test_root_self_edge_is_not_a_child(self):
        for child_count, expected_divisions in ((0, 0), (1, 0), (2, 1)):
            with self.subTest(non_self_children=child_count):
                cells = np.arange(child_count + 1)
                ancestors = np.zeros(child_count + 1, dtype=int)
                ends = [0] if child_count == 0 else [1, 2][:child_count]
                traced = tracing_lineage(cells, ancestors, ends)
                for lineage in traced.values():
                    self.assertEqual(lineage[1], expected_divisions)

        traced = tracing_lineage(
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 0, 1, 1, 2]),
            [4],
        )[4]
        self.assertEqual(traced[:2], (3, 1))
        self.assertEqual(traced[2], [4, 2, 1, 0])
        self.assertEqual(traced[3], [1])

    def test_exact_halving_is_accepted_for_root_and_nonroot_parents(self):
        for frames, parent, daughters in (
            ([[5.0], [2.5, 2.5]], 0, [1, 2]),
            ([[5.0], [5.0], [2.5, 2.5]], 1, [2, 3]),
            ([[5.0], [2.0, 3.0]], 0, [1, 2]),
            ([[5.0], [5.0], [2.0, 3.0]], 1, [2, 3]),
        ):
            with self.subTest(frames=frames):
                _, ancestor = linked_cells(frames)
                self.assertEqual(ancestor[daughters].tolist(), [parent, parent])

    def test_division_bounds_retain_their_inclusivity(self):
        cases = (
            # Daughter sum below the mother: the inclusive lower bound is not
            # permission to accept a negative daughter-sum-minus-mother difference.
            ([[5.0], [2.0, 2.75]], 0.8, False),
            # Upper bound remains strict: difference == threshold * mother.
            ([[5.0], [3.0, 6.0]], 0.8, False),
            # The independent first-daughter bound also remains strict.
            ([[5.0], [4.5, 2.5]], 0.8, False),
            # A case strictly inside both upper bounds is still accepted.
            ([[5.0], [3.0, 5.9]], 0.8, True),
        )
        for frames, threshold, expected in cases:
            with self.subTest(frames=frames, threshold=threshold):
                _, ancestor = linked_cells(frames, threshold)
                actual = np.all(ancestor[1:] == 0)
                self.assertEqual(actual, expected)


class TimeRegressionTests(unittest.TestCase):
    def test_units_have_identical_physical_hour_results(self):
        results = {}
        for unit in ("s", "min", "h", "d"):
            cells = physical_cells(unit=unit).sample(frac=1.0, random_state=17)
            before = cells.copy(deep=True)
            with self.subTest(unit=unit):
                result = get_one_trajectory(cells, time_unit=unit)
                self.assertIsNotNone(result)
                (trajectory, segments, division_time, growth, divisions), _ = result
                np.testing.assert_allclose(trajectory["time_h"], [0, 0.5, 1, 1.5, 2])
                assert_frame_equal(cells, before)
                results[unit] = (
                    trajectory["time_h"].to_numpy(),
                    segments[0][0],
                    division_time,
                    growth,
                    divisions,
                )
        reference = results["s"]
        for unit, result in results.items():
            for actual, expected in zip(result, reference):
                if isinstance(expected, np.ndarray):
                    np.testing.assert_allclose(actual, expected, equal_nan=True)
                else:
                    self.assertEqual(actual, expected)

    def test_unknown_time_unit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "time unit|time_unit|unit"):
            get_one_trajectory(physical_cells(), time_unit="fortnight")

    def test_duration_and_time_window_boundaries_are_in_hours(self):
        cells = physical_cells()
        result = get_one_trajectory(
            cells, time_unit="s", start_time=0.5, stop_time=1.5
        )
        self.assertIsNotNone(result)
        trajectory = result[0][0]
        # Cropping does not reset the physical-time origin, and both boundaries
        # are inclusive. The selected window spans exactly one hour.
        np.testing.assert_allclose(trajectory["time_h"], [0.5, 1.0, 1.5])
        accepted = process_channel_worker(
            "0", "A", cells, 0.8, "s", "longest_lineage", 1.0,
            start_time=0.5, stop_time=1.5,
        )
        self.assertIsNotNone(accepted)
        self.assertAlmostEqual(
            accepted[1]["trajectory"]["time_h"].max()
            - accepted[1]["trajectory"]["time_h"].min(),
            1.0,
        )
        self.assertIsNone(
            process_channel_worker(
                "0", "A", cells, 0.8, "s", "longest_lineage", 1.000001,
                start_time=0.5, stop_time=1.5,
            )
        )


class StatisticsRegressionTests(unittest.TestCase):
    def test_unsorted_input_has_sorted_segments_and_known_fit_statistics(self):
        order = np.array([6, 0, 8, 2, 4, 1, 9, 3, 5, 7])
        time = order.astype(float)
        frame = pd.DataFrame(
            {
                "time_h": time,
                "spine_length": 2.0 * np.exp(0.3 * time),
                "fov_cell_index": order,
            }
        )
        before = frame.copy(deep=True)
        segments, division_time, growth = compute_traj_stats(frame, [4, 8])

        for segment_time, _ in segments:
            self.assertTrue(np.all(np.diff(segment_time) >= 0))
        np.testing.assert_array_equal(segments[0][0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(segments[1][0], [5, 6, 7, 8])
        np.testing.assert_array_equal(segments[2][0], [9])
        np.testing.assert_allclose(division_time, [[6.0], [4.0]])
        np.testing.assert_allclose(growth[0], [2.0, 6.5, np.nan], equal_nan=True)
        np.testing.assert_allclose(growth[1], [0.3, 0.3, np.nan], equal_nan=True)
        np.testing.assert_allclose(growth[2], [1.0, 1.0, np.nan], equal_nan=True)
        assert_frame_equal(frame, before)

    def test_sort_is_stable_and_source_frame_is_unmodified(self):
        frame = pd.DataFrame(
            {
                "time_h": ["10", "2", "2", "1"],
                "spine_length": [4.0, 2.0, 3.0, 1.0],
                "fov_cell_index": [10, 2, 12, 1],
                "row_tag": ["ten", "first-two", "second-two", "one"],
            },
            index=[8, 5, 3, 2],
        )
        before = frame.copy(deep=True)
        segments, _, _ = compute_traj_stats(frame, [])
        np.testing.assert_array_equal(segments[0][0], [1, 2, 2, 10])
        self.assertEqual(
            segments[0][1]["row_tag"].tolist(),
            ["one", "first-two", "second-two", "ten"],
        )
        self.assertEqual(segments[0][1].index.tolist(), [2, 5, 3, 8])
        assert_frame_equal(frame, before)

    def test_invalid_observations_warn_and_do_not_contaminate_fit(self):
        time = np.arange(12, dtype=float)
        length = 2.0 * np.exp(0.25 * time)
        length[[1, 2, 3, 4, 5]] = [0.0, -1.0, np.nan, np.inf, -np.inf]
        time[[6, 7, 8]] = [np.nan, np.inf, -np.inf]
        frame = pd.DataFrame(
            {"time_h": time, "spine_length": length, "fov_cell_index": np.arange(12)}
        )
        before = frame.copy(deep=True)
        with self.assertWarnsRegex(RuntimeWarning, "invalid"):
            _, _, growth = compute_traj_stats(frame, [])
        self.assertEqual(growth.shape, (3, 1))
        np.testing.assert_allclose(growth[:, 0], [5.5, 0.25, 1.0])
        assert_frame_equal(frame, before)

    def test_invalid_length_mother_keeps_boundary_and_invalid_time_cannot(self):
        time = np.arange(11, dtype=float)
        length = 2.0 * np.exp(0.25 * time)
        length[4] = 0.0  # An invalid fit value must not erase its cycle boundary.
        time[10] = np.nan
        frame = pd.DataFrame(
            {"time_h": time, "spine_length": length, "fov_cell_index": np.arange(11)}
        )
        before = frame.copy(deep=True)
        with self.assertWarnsRegex(RuntimeWarning, "invalid"):
            segments, division_time, growth = compute_traj_stats(frame, [4, 8, 10])
        self.assertEqual(
            [segment[1]["fov_cell_index"].tolist() for segment in segments],
            [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]],
        )
        self.assertEqual(segments[0][1]["spine_length"].iloc[-1], 0.0)
        np.testing.assert_allclose(division_time, [[6.0], [4.0]])
        np.testing.assert_allclose(growth[:, :2], [[1.5, 6.5], [0.25, 0.25], [1.0, 1.0]])
        assert_frame_equal(frame, before)

    def test_noisy_log_lengths_have_independently_known_r2(self):
        # For these five log lengths: sum((t-mean(t))^2)=10,
        # cross-product=3.2, total response sum of squares=1.228.
        # Thus OLS slope=.32 and R2=(.32**2*10)/1.228=256/307.
        frame = pd.DataFrame(
            {"time_h": np.arange(5, dtype=float),
             "spine_length": np.exp([0.0, 0.5, 0.4, 1.3, 1.2]),
             "fov_cell_index": np.arange(5)}
        )
        _, _, growth = compute_traj_stats(frame, [])
        np.testing.assert_allclose(growth[:, 0], [2.0, 0.32, 256.0 / 307.0])

    def test_no_time_variation_warns_instead_of_fitting(self):
        frame = pd.DataFrame(
            {"time_h": [1.0] * 4, "spine_length": [2.0, 3.0, 4.0, 5.0],
             "fov_cell_index": np.arange(4)}
        )
        with self.assertWarnsRegex(RuntimeWarning, "time variation"):
            _, _, growth = compute_traj_stats(frame, [])
        self.assertTrue(np.isnan(growth).all())

    def test_constant_lengths_have_zero_slope_and_undefined_r2(self):
        frame = pd.DataFrame(
            {
                "time_h": np.arange(4, dtype=float),
                "spine_length": np.full(4, 2.0),
                "fov_cell_index": np.arange(4),
            }
        )
        _, _, growth = compute_traj_stats(frame, [])
        np.testing.assert_allclose(growth[:2, 0], [1.5, 0.0])
        self.assertTrue(np.isnan(growth[2, 0]))

    def test_fewer_than_four_valid_points_produce_nan_statistics(self):
        frame = pd.DataFrame(
            {
                "time_h": [0.0, 1.0, 2.0],
                "spine_length": [1.0, 2.0, 4.0],
                "fov_cell_index": [0, 1, 2],
            }
        )
        _, _, growth = compute_traj_stats(frame, [])
        self.assertEqual(growth.shape, (3, 1))
        self.assertTrue(np.isnan(growth).all())


if __name__ == "__main__":
    unittest.main()
