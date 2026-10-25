"""Streamlit integration tests for division/exclusion conflict resolution."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from moma_lineage_analysis.moma_logic import compute_traj_stats
from moma_lineage_analysis.paths import PACKAGE_ROOT
from moma_lineage_analysis.trajectory_cleaning import (
    ModelConfig,
    TrajectoryNormalizer,
    build_model,
    save_checkpoint,
)
from streamlit.testing.v1 import AppTest

APP_FILE = str(PACKAGE_ROOT / "app" / "Moma_lineage_streamlit.py")


TRAJECTORY_KEY = "trajectory_0"


def _element(elements, label: str):
    return next(element for element in elements if element.label == label)


def _element_containing(elements, label_fragment: str):
    return next(
        element for element in elements if label_fragment in element.label
    )


def _trajectory_entry(
    *,
    next_length: float = 2.0,
    divisions: list[int] | None = None,
    exclusions: set[int] | None = None,
) -> dict:
    """Build a trajectory whose point 3 can resolve onto clean point 2."""

    dataframe = pd.DataFrame(
        {
            "fov_cell_index": np.arange(8),
            "time_h": np.arange(8, dtype=float),
            "spine_length": [2.0, 2.8, 4.0, 20.0, next_length, 2.4, 3.0, 4.0],
            "signal_mean": np.arange(8, dtype=float),
        }
    )
    raw_divisions = [3, 7] if divisions is None else list(divisions)
    segments, division_time, growth_rate = compute_traj_stats(
        dataframe,
        raw_divisions,
    )
    return {
        "trajectory": dataframe,
        "segments": segments,
        "division_time": division_time,
        "growth_rate": growth_rate,
        "div_indices": raw_divisions.copy(),
        "original_div_indices": raw_divisions.copy(),
        "excluded_indices": set() if exclusions is None else set(exclusions),
    }


def _app_with_entry(entry: dict) -> AppTest:
    app = AppTest.from_file(
        APP_FILE,
        default_timeout=30,
    ).run()
    if app.exception:
        raise AssertionError(app.exception)

    app.session_state["trajectory_dict"] = {TRAJECTORY_KEY: entry}
    app.session_state["selected_keys"] = {TRAJECTORY_KEY}
    app.session_state["current_idx"] = 0
    app.run()
    if app.exception:
        raise AssertionError(app.exception)
    return app


def _select_plot_points(app: AppTest, points: list[dict]) -> None:
    app.session_state["traj_plot"] = {
        "selection": {
            "points": points,
            "point_indices": list(range(len(points))),
            "box": [],
            "lasso": [],
        }
    }


def _checkpoint_signature(checkpoint: str, device: str = "cpu") -> tuple:
    path = Path(checkpoint).resolve()
    file_stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        str(path),
        file_stat.st_mtime_ns,
        file_stat.st_size,
        digest,
        device,
    )


def _save_tiny_checkpoint(checkpoint: str) -> None:
    """Create the same lightweight CNN checkpoint used by the AI panel tests."""

    model_config = ModelConfig(
        model_type="cnn",
        hidden_size=8,
        num_layers=1,
        kernel_size=3,
        dropout=0.0,
    )
    save_checkpoint(
        checkpoint,
        build_model(model_config),
        model_config=model_config,
        normalizer=TrajectoryNormalizer(),
        thresholds=(0.5, 0.5),
        metadata={"purpose": "streamlit auto-pilot regression test"},
    )


def _app_with_entries(
    entries: dict[str, dict],
    *,
    current_idx: int = 0,
) -> AppTest:
    app = AppTest.from_file(
        APP_FILE,
        default_timeout=30,
    ).run()
    if app.exception:
        raise AssertionError(app.exception)

    app.session_state["trajectory_dict"] = entries
    app.session_state["selected_keys"] = set(entries)
    app.session_state["current_idx"] = current_idx
    app.run()
    if app.exception:
        raise AssertionError(app.exception)
    return app


def _load_tiny_ai_model(app: AppTest, checkpoint: str) -> None:
    """Enable AI, load a CPU checkpoint, and make predictions deterministic."""

    _element(app.toggle, "Enable AI-assisted cleaning").set_value(True)
    app.run()
    _element(app.selectbox, "Model checkpoint").set_value("Custom path…")
    app.run()
    _element(app.text_input, "Custom checkpoint path").set_value(checkpoint)
    app.run()
    _element(app.selectbox, "Device").set_value("cpu")
    app.run()
    _element(app.button, "📦 Load / Reload Model").click()
    app.run()
    if app.exception:
        raise AssertionError(app.exception)

    # Sigmoid outputs are strictly positive, so zero thresholds make every
    # point a stable suggestion regardless of the checkpoint's random weights.
    _element(app.slider, "Division threshold").set_value(0.0)
    _element(app.slider, "Noise threshold").set_value(0.0)
    app.run()
    if app.exception:
        raise AssertionError(app.exception)


def _entry_with_invalid_model_rows() -> tuple[dict, set[int]]:
    """Return one trajectory with two hard-invalid model input rows."""

    entry = _trajectory_entry(divisions=[3, 7])
    trajectory = entry["trajectory"].copy()
    trajectory.loc[trajectory["fov_cell_index"] == 1, "time_h"] = np.nan
    trajectory.loc[
        trajectory["fov_cell_index"] == 5,
        "spine_length",
    ] = np.inf
    entry["trajectory"] = trajectory
    return entry, {1, 5}


def _viewer_status(app: AppTest) -> str:
    return next(
        str(element.value)
        for element in app.markdown
        if "Current:" in str(element.value)
        and "Selected:" in str(element.value)
    )


def _main_plot_traces(app: AppTest) -> list[dict]:
    chart = next(
        element
        for element in app.get("plotly_chart")
        if element.proto.id.endswith("-traj_plot")
    )
    return json.loads(chart.proto.spec)["data"]


def _decode_plotly_array(value) -> np.ndarray:
    """Decode plain or Plotly-6 typed-array JSON values for trace assertions."""

    if isinstance(value, dict) and "bdata" in value:
        decoded = np.frombuffer(
            base64.b64decode(value["bdata"]),
            dtype=np.dtype(value["dtype"]),
        )
        shape = value.get("shape")
        if shape:
            decoded = decoded.reshape(
                tuple(int(part.strip()) for part in shape.split(","))
            )
        return decoded
    return np.asarray(value, dtype=object)


def _trace_ids(trace: dict) -> list[int]:
    customdata = trace.get("customdata")
    if customdata is None:
        return []
    values = _decode_plotly_array(customdata)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim > 1:
        values = values[:, 0]
    result = []
    for value in values.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        result.append(int(value))
    return result


def _trace_values(trace: dict, field: str) -> np.ndarray:
    value = trace.get(field)
    if value is None:
        return np.asarray([], dtype=object)
    return _decode_plotly_array(value)


def _line_id_runs(traces: list[dict]) -> list[list[int]]:
    return [
        _trace_ids(trace)
        for trace in traces
        if "lines" in trace.get("mode", "")
    ]


def _applied_exclusion_traces(traces: list[dict]) -> list[dict]:
    """The canonical trace name is part of the cleaning-panel UI contract."""

    return [trace for trace in traces if trace.get("name") == "Applied exclusion"]


class StreamlitDivisionResolutionPanelTests(unittest.TestCase):
    def _assert_invalid_model_rows_are_synchronized(
        self,
        app: AppTest,
        trajectory_key: str,
        invalid_ids: set[int],
    ) -> None:
        entry = app.session_state["trajectory_dict"][trajectory_key]
        trajectory = entry["trajectory"]
        all_ids = set(trajectory["fov_cell_index"].tolist())
        valid_ids = all_ids - invalid_ids

        self.assertEqual(entry["excluded_indices"], invalid_ids)
        self.assertEqual(entry["effective_div_indices"], [3, 7])
        self.assertGreaterEqual(entry["_trajectory_stats_recompute_count"], 1)
        self.assertEqual(
            entry["_trajectory_stats_signature"][-1],
            tuple(sorted(invalid_ids)),
        )

        segment_frames = [frame for _, frame in entry["segments"]]
        segment_ids = [
            int(cell_id)
            for frame in segment_frames
            for cell_id in frame["fov_cell_index"].tolist()
        ]
        self.assertEqual(set(segment_ids), valid_ids)
        self.assertEqual(len(segment_ids), len(valid_ids))
        for frame in segment_frames:
            self.assertTrue(
                np.isfinite(
                    pd.to_numeric(frame["time_h"], errors="coerce")
                ).all()
            )
            self.assertTrue(
                np.isfinite(
                    pd.to_numeric(frame["spine_length"], errors="coerce")
                ).all()
            )

        prediction = app.session_state["ai_predictions"][trajectory_key]
        predicted_ids = set(prediction["fov_cell_index"].tolist())
        self.assertEqual(predicted_ids, all_ids)
        valid_prediction = prediction[
            prediction["fov_cell_index"].isin(valid_ids)
        ]
        self.assertTrue(valid_prediction["ai_input_valid"].all())
        self.assertEqual(
            set(valid_prediction["ai_prediction_source"]),
            {"model"},
        )
        self.assertTrue(
            np.isfinite(
                pd.to_numeric(valid_prediction["time_h"], errors="coerce")
            ).all()
        )
        self.assertTrue(
            np.isfinite(
                pd.to_numeric(
                    valid_prediction["spine_length"],
                    errors="coerce",
                )
            ).all()
        )
        invalid_prediction = prediction[
            prediction["fov_cell_index"].isin(invalid_ids)
        ].set_index("fov_cell_index")
        self.assertFalse(invalid_prediction["ai_input_valid"].any())
        self.assertEqual(
            set(invalid_prediction["ai_prediction_source"]),
            {"validation_rule"},
        )
        self.assertEqual(
            invalid_prediction.loc[1, "ai_input_invalid_reason"],
            "nonfinite_time",
        )
        self.assertEqual(
            invalid_prediction.loc[5, "ai_input_invalid_reason"],
            "nonfinite_spine_length",
        )
        self.assertTrue(
            (invalid_prediction["pred_division_probability"] == 0.0).all()
        )
        self.assertTrue(
            (invalid_prediction["pred_excluded_probability"] == 1.0).all()
        )
        self.assertFalse(invalid_prediction["pred_division_point"].any())
        self.assertTrue(invalid_prediction["pred_excluded_point"].all())

    def test_keep_drop_reuses_figure_and_statistics_caches(self) -> None:
        app = _app_with_entry(_trajectory_entry())
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        # The overlay style/notice block must render on every run regardless of
        # Keep/Drop state, so toggling the button cannot shift its position.
        initial_markdown_count = len(app.markdown)
        self.assertNotIn(
            ".st-key-toggle_trajectory_selection_btn",
            " ".join(element.value for element in app.markdown),
        )
        initial_stats_count = data["_trajectory_stats_recompute_count"]
        initial_cache = app.session_state["trajectory_figure_cache"]
        initial_signature = initial_cache["signature"]
        initial_figure = initial_cache["figure"]
        initial_build_count = initial_cache["build_count"]
        self.assertEqual(len(app.session_state["selected_keys"]), 1)
        self.assertIn("Selected:** 1", _viewer_status(app))

        _element(app.button, "❌ Drop Trajectory").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.markdown), initial_markdown_count)
        self.assertIn(
            ".st-key-toggle_trajectory_selection_btn",
            " ".join(element.value for element in app.markdown),
        )
        self.assertEqual(app.session_state["selected_keys"], set())
        self.assertEqual(len(app.session_state["selected_keys"]), 0)
        self.assertIn("Selected:** 0", _viewer_status(app))
        dropped_cache = app.session_state["trajectory_figure_cache"]
        self.assertEqual(dropped_cache["signature"], initial_signature)
        self.assertIs(dropped_cache["figure"], initial_figure)
        self.assertEqual(dropped_cache["build_count"], initial_build_count)
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_stats_count,
        )

        _element(app.button, "✅ Keep Trajectory").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.markdown), initial_markdown_count)
        self.assertNotIn(
            ".st-key-toggle_trajectory_selection_btn",
            " ".join(element.value for element in app.markdown),
        )
        self.assertEqual(app.session_state["selected_keys"], {TRAJECTORY_KEY})
        self.assertEqual(len(app.session_state["selected_keys"]), 1)
        self.assertIn("Selected:** 1", _viewer_status(app))
        kept_cache = app.session_state["trajectory_figure_cache"]
        self.assertEqual(kept_cache["signature"], initial_signature)
        self.assertIs(kept_cache["figure"], initial_figure)
        self.assertEqual(kept_cache["build_count"], initial_build_count)
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_stats_count,
        )

    def test_drop_fov_matches_exact_index_and_preserves_trajectories(self) -> None:
        entries = {
            key: _trajectory_entry()
            for key in (
                "FOV_0_CH_1", "FOV_00_CH_2", "FOV_1_CH_0",
                "FOV_01_CH_2", "FOV_10_CH_1", "trajectory_0",
            )
        }
        app = _app_with_entries(entries)
        initial_cache = app.session_state["trajectory_figure_cache"]
        initial_epoch = app.session_state["trajectory_plot_epoch"]
        self.assertEqual(app.sidebar.number_input(key="drop_fov_index").value, 0)

        for fov_index, remaining, count in (
            (0, {"FOV_1_CH_0", "FOV_01_CH_2", "FOV_10_CH_1", "trajectory_0"}, 2),
            (1, {"FOV_10_CH_1", "trajectory_0"}, 2),
        ):
            with self.subTest(fov_index=fov_index):
                app.sidebar.number_input(key="drop_fov_index").set_value(fov_index).run()
                app.sidebar.button(key="drop_fov_trajectories").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["selected_keys"], remaining)
                self.assertEqual(set(app.session_state["trajectory_dict"]), set(entries))
                self.assertIn(
                    f"Dropped {count} trajectories from FOV {fov_index}.",
                    [item.value for item in app.sidebar.success],
                )
                self.assertEqual(app.session_state["current_idx"], 0)
                self.assertEqual(app.session_state["trajectory_plot_epoch"], initial_epoch)
                self.assertIs(
                    app.session_state["trajectory_figure_cache"]["figure"],
                    initial_cache["figure"],
                )
                for entry in entries.values():
                    self.assertEqual(entry["div_indices"], [3, 7])
                    self.assertEqual(entry["excluded_indices"], set())
        self.assertFalse(_element(app.button, "✅ Keep Trajectory").disabled)
        _element(app.button, "✅ Keep Trajectory").click().run()
        self.assertFalse(app.exception)
        self.assertIn("FOV_0_CH_1", app.session_state["selected_keys"])

    def test_drop_fov_empty_missing_and_already_dropped_feedback(self) -> None:
        app = AppTest.from_file(APP_FILE, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.sidebar.number_input(key="drop_fov_index").disabled)
        self.assertTrue(app.sidebar.button(key="drop_fov_trajectories").disabled)

        app = _app_with_entries({"FOV_0_CH_1": _trajectory_entry()})
        app.sidebar.number_input(key="drop_fov_index").set_value(99).run()
        app.sidebar.button(key="drop_fov_trajectories").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_keys"], {"FOV_0_CH_1"})
        self.assertIn(
            "No loaded trajectories match FOV 99.",
            [item.value for item in app.sidebar.warning],
        )

        app.sidebar.number_input(key="drop_fov_index").set_value(0).run()
        app.sidebar.button(key="drop_fov_trajectories").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_keys"], set())
        self.assertIn("Selected:** 0", _viewer_status(app))
        app.sidebar.button(key="drop_fov_trajectories").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_keys"], set())
        self.assertIn(
            "All trajectories in FOV 0 are already dropped.",
            [item.value for item in app.sidebar.info],
        )

    def test_drop_fov_updates_selected_statistics_and_labeled_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            entries = {
                "FOV_0_CH_1": _trajectory_entry(),
                "FOV_0_CH_2": _trajectory_entry(),
                "FOV_2_CH_1": _trajectory_entry(),
            }
            app = _app_with_entries(entries)
            _element(app.sidebar.text_input, "Data Directory Path").set_value(
                temporary_directory
            ).run()
            _element(app.toggle, "Calculate selected statistics").set_value(True).run()
            _element(app.radio, "Statistic").set_value("Generation Time").run()

            def generation_sample_count():
                histogram = next(
                    element for element in app.get("plotly_chart")
                    if element.proto.id.endswith("-selected_generation_histogram")
                )
                trace = json.loads(histogram.proto.spec)["data"][0]
                return len(_trace_values(trace, "x"))

            all_samples = generation_sample_count()
            self.assertGreater(all_samples, 0)
            app.sidebar.button(key="drop_fov_trajectories").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["selected_keys"], {"FOV_2_CH_1"})
            self.assertEqual(generation_sample_count(), all_samples // 3)
            _element(app.sidebar.button, "📊 Export with Labels").click().run()
            self.assertFalse(app.exception)
            exported = pd.read_csv(Path(temporary_directory) / "selected_labeled.csv")
            self.assertEqual(set(exported["trajectory_key"]), {"FOV_2_CH_1"})
            self.assertEqual(len(exported), len(entries["FOV_2_CH_1"]["trajectory"]))

    def test_trajectory_jump_controls_sync_and_clear_plot_selection(self) -> None:
        entries = {
            f"trajectory_{index}": _trajectory_entry()
            for index in range(3)
        }
        app = _app_with_entries(entries)
        self.assertEqual(app.select_slider(key="trajectory_jump_slider").value, 1)
        self.assertEqual(app.number_input(key="trajectory_jump_number").value, 1)
        epoch = app.session_state["trajectory_plot_epoch"]
        _select_plot_points(app, [{"x": 3.0, "customdata": [3]}])

        app.select_slider(key="trajectory_jump_slider").set_value(3).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_idx"], 2)
        self.assertEqual(app.number_input(key="trajectory_jump_number").value, 3)
        self.assertIn("trajectory_2 (3/3)", _viewer_status(app))
        self.assertEqual(app.session_state["trajectory_plot_epoch"], epoch + 1)
        self.assertFalse(app.session_state["traj_plot"]["selection"]["points"])

        app.number_input(key="trajectory_jump_number").set_value(1).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_idx"], 0)
        self.assertEqual(app.select_slider(key="trajectory_jump_slider").value, 1)
        self.assertNotIn(
            "All trajectories have been reviewed",
            " ".join(element.value for element in app.markdown),
        )

        for label, number in (
            ("Next ➡️", 2), ("⬅️ Prev", 1), ("⬅️ Prev", 3), ("Next ➡️", 1)
        ):
            with self.subTest(label=label, number=number):
                _element(app.button, label).click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["current_idx"], number - 1)
                self.assertEqual(
                    app.select_slider(key="trajectory_jump_slider").value, number
                )
                self.assertEqual(
                    app.number_input(key="trajectory_jump_number").value, number
                )
        self.assertIn(
            "All trajectories have been reviewed",
            " ".join(element.value for element in app.markdown),
        )
        for entry in entries.values():
            self.assertEqual(entry["div_indices"], [3, 7])
            self.assertEqual(entry["excluded_indices"], set())

    def test_trajectory_jump_controls_reset_with_loaded_data(self) -> None:
        app = _app_with_entries({
            f"trajectory_{index}": _trajectory_entry()
            for index in range(4)
        })
        app.number_input(key="trajectory_jump_number").set_value(4).run()
        self.assertFalse(app.exception)

        # Process Data replaces the trajectories and resets current_idx to zero.
        for count in (4, 1, 0, 3):
            with self.subTest(trajectory_count=count):
                entries = {
                    f"replacement_{index}": _trajectory_entry()
                    for index in range(count)
                }
                app.session_state["trajectory_dict"] = entries
                app.session_state["selected_keys"] = set(entries)
                app.session_state["current_idx"] = 0
                app.run()
                self.assertFalse(app.exception)
                if not count:
                    self.assertFalse(app.select_slider)
                    self.assertIn("Load data to start.", [item.value for item in app.info])
                    continue
                slider = app.select_slider(key="trajectory_jump_slider")
                number_input = app.number_input(key="trajectory_jump_number")
                self.assertEqual(slider.value, 1)
                self.assertEqual(number_input.value, 1)
                self.assertEqual(number_input.max, count)
                self.assertEqual(slider.disabled, count == 1)
                self.assertEqual(number_input.disabled, count == 1)

    def test_trajectory_jumps_bypass_autopilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            _save_tiny_checkpoint(checkpoint)
            entries = {
                f"trajectory_{index}": _trajectory_entry(divisions=[])
                for index in range(3)
            }
            app = _app_with_entries(entries)
            _load_tiny_ai_model(app, checkpoint)
            _element(app.toggle, "Auto pilot on Next").set_value(True).run()

            app.select_slider(key="trajectory_jump_slider").set_value(3).run()
            app.number_input(key="trajectory_jump_number").set_value(2).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["current_idx"], 1)
            self.assertEqual(app.session_state["ai_predictions"], {})
            self.assertEqual(app.session_state["ai_autopilot_completed_count"], 0)
            self.assertNotIn("ai_autopilot_request", app.session_state)
            for entry in entries.values():
                self.assertEqual(entry["div_indices"], [])
                self.assertEqual(entry["excluded_indices"], set())

            _element(app.button, "Next ➡️").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["ai_autopilot_completed_count"], 1)
            self.assertEqual(app.select_slider(key="trajectory_jump_slider").value, 3)
            self.assertEqual(app.number_input(key="trajectory_jump_number").value, 3)

    def test_autopilot_next_predicts_and_applies_destination_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            _save_tiny_checkpoint(checkpoint)
            source = _trajectory_entry(divisions=[3, 7])
            destination = _trajectory_entry(divisions=[])
            app = _app_with_entries(
                {
                    "trajectory_0": source,
                    "trajectory_1": destination,
                }
            )
            _load_tiny_ai_model(app, checkpoint)

            autopilot = _element(app.toggle, "Auto pilot on Next")
            self.assertFalse(autopilot.disabled)
            autopilot.set_value(True)
            app.run()
            self.assertTrue(app.session_state["ai_autopilot_enabled"])

            _element(app.button, "Next ➡️").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["current_idx"], 1)
            self.assertEqual(
                app.session_state["ai_autopilot_completed_count"],
                1,
            )
            self.assertEqual(
                app.session_state["ai_autopilot_last_key"],
                "trajectory_1",
            )
            self.assertEqual(
                app.session_state["ai_autopilot_last_status"],
                "success",
            )
            self.assertIn("trajectory_1", app.session_state["ai_predictions"])
            expected_ids = set(range(8))
            applied_destination = app.session_state["trajectory_dict"][
                "trajectory_1"
            ]
            self.assertEqual(
                set(applied_destination["div_indices"]),
                expected_ids,
            )
            self.assertEqual(
                applied_destination["excluded_indices"],
                expected_ids,
            )
            self.assertEqual(source["div_indices"], [3, 7])
            self.assertEqual(source["excluded_indices"], set())

    def test_autopilot_prev_only_navigates_without_prediction_or_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            _save_tiny_checkpoint(checkpoint)
            previous = _trajectory_entry(divisions=[])
            current = _trajectory_entry(divisions=[3, 7])
            app = _app_with_entries(
                {
                    "trajectory_0": previous,
                    "trajectory_1": current,
                },
                current_idx=1,
            )
            _load_tiny_ai_model(app, checkpoint)
            _element(app.toggle, "Auto pilot on Next").set_value(True)
            app.run()
            self.assertTrue(app.session_state["ai_autopilot_enabled"])

            _element(app.button, "⬅️ Prev").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["current_idx"], 0)
            self.assertEqual(
                app.session_state["ai_autopilot_completed_count"],
                0,
            )
            self.assertIsNone(app.session_state["ai_autopilot_last_key"])
            self.assertIsNone(app.session_state["ai_autopilot_last_status"])
            self.assertEqual(app.session_state["ai_predictions"], {})
            self.assertEqual(previous["div_indices"], [])
            self.assertEqual(previous["excluded_indices"], set())
            self.assertEqual(current["div_indices"], [3, 7])
            self.assertEqual(current["excluded_indices"], set())

    def test_autopilot_invalid_or_unloaded_model_fails_without_mutation(
        self,
    ) -> None:
        for scenario in ("unloaded", "invalidated"):
            with self.subTest(scenario=scenario):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    checkpoint = os.path.join(
                        temporary_directory,
                        "tiny_model.pt",
                    )
                    _save_tiny_checkpoint(checkpoint)
                    source = _trajectory_entry(divisions=[3, 7])
                    destination = _trajectory_entry(divisions=[])
                    app = _app_with_entries(
                        {
                            "trajectory_0": source,
                            "trajectory_1": destination,
                        }
                    )

                    if scenario == "unloaded":
                        _element(
                            app.toggle,
                            "Enable AI-assisted cleaning",
                        ).set_value(True)
                        app.run()
                        _element(
                            app.selectbox,
                            "Model checkpoint",
                        ).set_value("Custom path…")
                        app.run()
                        _element(
                            app.text_input,
                            "Custom checkpoint path",
                        ).set_value(checkpoint)
                        app.run()
                        _element(app.selectbox, "Device").set_value("cpu")
                        app.run()
                        self.assertTrue(
                            _element(
                                app.toggle,
                                "Auto pilot on Next",
                            ).disabled
                        )
                        # Simulate a stale session that was armed before the
                        # model became unavailable. The callback must fail safe.
                        app.session_state["ai_autopilot_enabled"] = True
                    else:
                        _load_tiny_ai_model(app, checkpoint)
                        _element(
                            app.toggle,
                            "Auto pilot on Next",
                        ).set_value(True)
                        app.run()
                        os.remove(checkpoint)

                    _element(app.button, "Next ➡️").click()
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state["current_idx"], 0)
                    self.assertEqual(
                        app.session_state["ai_autopilot_completed_count"],
                        0,
                    )
                    self.assertEqual(
                        app.session_state["ai_autopilot_last_key"],
                        "trajectory_1",
                    )
                    self.assertEqual(
                        app.session_state["ai_autopilot_last_status"],
                        "error",
                    )
                    self.assertIn(
                        "Auto pilot failed for trajectory_1",
                        app.session_state["ai_model_error"],
                    )
                    self.assertNotIn(
                        "trajectory_1",
                        app.session_state["ai_predictions"],
                    )
                    self.assertEqual(source["div_indices"], [3, 7])
                    self.assertEqual(source["excluded_indices"], set())
                    self.assertEqual(destination["div_indices"], [])
                    self.assertEqual(destination["excluded_indices"], set())

    def test_ai_preview_preflights_invalid_rows_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            _save_tiny_checkpoint(checkpoint)
            destination, invalid_ids = _entry_with_invalid_model_rows()
            app = _app_with_entries(
                {
                    "trajectory_0": _trajectory_entry(divisions=[3, 7]),
                    "trajectory_1": destination,
                }
            )
            _load_tiny_ai_model(app, checkpoint)
            _element(app.selectbox, "Scope").set_value("All")
            app.run()

            _element(app.button, "✨ Run AI Preview").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["ai_model_error"])
            self.assertIn("trajectory_1", app.session_state["ai_predictions"])
            self._assert_invalid_model_rows_are_synchronized(
                app,
                "trajectory_1",
                invalid_ids,
            )

            # Threshold zero must not turn validation-rule placeholders into
            # exported division labels.
            _element(app.sidebar.text_input, "Data Directory Path").set_value(
                temporary_directory
            )
            app.run()
            _element(app.sidebar.button, "📊 Export with Labels").click()
            app.run()
            self.assertFalse(app.exception)
            exported = pd.read_csv(
                os.path.join(temporary_directory, "selected_labeled.csv")
            )
            invalid_export = exported[
                (exported["trajectory_key"] == "trajectory_1")
                & exported["fov_cell_index"].isin(invalid_ids)
            ]
            self.assertEqual(len(invalid_export), len(invalid_ids))
            self.assertEqual(
                set(invalid_export["division_point"].astype(str).str.lower()),
                {"false"},
            )
            self.assertEqual(
                set(invalid_export["excluded_point"].astype(str).str.lower()),
                {"true"},
            )
            # AI prediction/provenance columns are deliberately not exported;
            # deterministic preflight exclusions surface only as canonical
            # labels in the export schema.
            for dropped_column in (
                "pred_division_point",
                "pred_excluded_point",
                "pred_division_probability",
                "pred_excluded_probability",
                "ai_prediction_source",
                "ai_input_valid",
            ):
                self.assertNotIn(dropped_column, exported.columns)

            # A nonfinite-time point has no selectable x-coordinate, so the
            # dedicated validation control must still support a per-ID rollback.
            _element(app.button, "Next ➡️").click()
            app.run()
            self.assertEqual(app.session_state["current_idx"], 1)
            _element(
                app.multiselect,
                "Invalid input IDs to restore",
            ).set_value([1])
            app.run()
            _element(
                app.button,
                "↩️ Restore Chosen Invalid Inputs",
            ).click()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(
                app.session_state["trajectory_dict"]["trajectory_1"][
                    "excluded_indices"
                ],
                invalid_ids - {1},
            )

    def test_current_invalid_rows_do_not_crash_before_ai_preflight(self) -> None:
        entry, invalid_ids = _entry_with_invalid_model_rows()
        app = _app_with_entry(entry)
        self.assertFalse(app.exception)

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], set())
        segment_ids = {
            int(cell_id)
            for _, frame in data["segments"]
            for cell_id in frame["fov_cell_index"].tolist()
        }
        self.assertEqual(segment_ids, set(range(8)) - invalid_ids)
        self.assertEqual(data["effective_div_indices"], [3, 7])

    def test_autopilot_preflights_invalid_destination_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            _save_tiny_checkpoint(checkpoint)
            destination, invalid_ids = _entry_with_invalid_model_rows()
            app = _app_with_entries(
                {
                    "trajectory_0": _trajectory_entry(divisions=[3, 7]),
                    "trajectory_1": destination,
                }
            )
            _load_tiny_ai_model(app, checkpoint)
            # No ordinary random-model suggestions should be applied. Invalid
            # observations remain deterministic exclusions from preflight.
            _element(app.slider, "Division threshold").set_value(1.0)
            _element(app.slider, "Noise threshold").set_value(1.0)
            app.run()
            _element(app.toggle, "Auto pilot on Next").set_value(True)
            app.run()

            _element(app.button, "Next ➡️").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["current_idx"], 1)
            self.assertEqual(
                app.session_state["ai_autopilot_last_status"],
                "success",
            )
            self.assertEqual(
                app.session_state["ai_autopilot_last_key"],
                "trajectory_1",
            )
            self.assertEqual(
                app.session_state["ai_autopilot_completed_count"],
                1,
            )
            self._assert_invalid_model_rows_are_synchronized(
                app,
                "trajectory_1",
                invalid_ids,
            )

    def test_unchanged_trajectory_reuses_cached_derived_statistics(self) -> None:
        app = _app_with_entry(_trajectory_entry(divisions=[3, 7]))
        before = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        before_count = before["_trajectory_stats_recompute_count"]
        before_signature = before["_trajectory_stats_signature"]
        before_division_time = before["division_time"].copy()
        before_growth_rate = before["growth_rate"].copy()
        before_effective = list(before["effective_div_indices"])

        app.run()
        self.assertFalse(app.exception)
        after = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(
            after["_trajectory_stats_recompute_count"],
            before_count,
        )
        self.assertEqual(after["_trajectory_stats_signature"], before_signature)
        self.assertEqual(after["effective_div_indices"], before_effective)
        np.testing.assert_allclose(
            after["division_time"],
            before_division_time,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            after["growth_rate"],
            before_growth_rate,
            equal_nan=True,
        )

    def test_label_and_schema_changes_each_invalidate_statistics_cache(self) -> None:
        app = _app_with_entry(_trajectory_entry(divisions=[]))
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        initial_count = data["_trajectory_stats_recompute_count"]
        initial_signature = data["_trajectory_stats_signature"]

        data["div_indices"].append(2)
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        division_signature = data["_trajectory_stats_signature"]
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_count + 1,
        )
        self.assertNotEqual(division_signature, initial_signature)
        self.assertEqual(data["effective_div_indices"], [2])

        data["excluded_indices"].add(2)
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        excluded_signature = data["_trajectory_stats_signature"]
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_count + 2,
        )
        self.assertNotEqual(excluded_signature, division_signature)

        # A stale derived-state schema must miss even though the raw-input
        # signature itself is otherwise unchanged.
        data["_trajectory_stats_schema"] = -1
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_count + 3,
        )
        self.assertGreater(data["_trajectory_stats_schema"], 0)
        self.assertEqual(
            data["_trajectory_stats_signature"],
            excluded_signature,
        )

        # The public data-revision component used by
        # mark_trajectory_data_changed() also invalidates unchanged labels.
        previous_signature = data["_trajectory_stats_signature"]
        data["_trajectory_data_revision"] = int(
            data.get("_trajectory_data_revision", 0)
        ) + 1
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_count + 4,
        )
        self.assertNotEqual(
            data["_trajectory_stats_signature"],
            previous_signature,
        )

    def test_warm_navigation_does_not_recompute_selected_trajectories(self) -> None:
        app = AppTest.from_file(
            APP_FILE,
            default_timeout=30,
        ).run()
        self.assertFalse(app.exception)
        trajectory_count = 24
        trajectory_dict = {
            f"trajectory_{index}": _trajectory_entry(divisions=[3, 7])
            for index in range(trajectory_count)
        }
        app.session_state["trajectory_dict"] = trajectory_dict
        app.session_state["selected_keys"] = set(trajectory_dict)
        app.session_state["current_idx"] = 0
        app.run()
        self.assertFalse(app.exception)

        # Aggregate statistics are intentionally lazy. Warm every selected
        # trajectory through the user-facing control before measuring whether
        # navigation reuses the cached per-trajectory results.
        _element(app.toggle, "Calculate selected statistics").set_value(True)
        app.run()
        self.assertFalse(app.exception)

        warmed = app.session_state["trajectory_dict"]
        before_counts = {
            key: entry["_trajectory_stats_recompute_count"]
            for key, entry in warmed.items()
        }
        before_signatures = {
            key: entry["_trajectory_stats_signature"]
            for key, entry in warmed.items()
        }
        self.assertEqual(len(before_counts), trajectory_count)
        self.assertTrue(all(count == 1 for count in before_counts.values()))

        _element(app.button, "Next ➡️").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_idx"], 1)
        _element(app.button, "⬅️ Prev").click()
        app.run()
        self.assertFalse(app.exception)
        app.run()
        self.assertFalse(app.exception)

        after = app.session_state["trajectory_dict"]
        self.assertEqual(
            {
                key: entry["_trajectory_stats_recompute_count"]
                for key, entry in after.items()
            },
            before_counts,
        )
        self.assertEqual(
            {
                key: entry["_trajectory_stats_signature"]
                for key, entry in after.items()
            },
            before_signatures,
        )

    def test_noop_exclude_and_empty_reset_remain_cache_hits(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[], exclusions={3})
        )
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        initial_count = data["_trajectory_stats_recompute_count"]
        initial_signature = data["_trajectory_stats_signature"]

        _select_plot_points(app, [{"x": 3.0, "customdata": [3]}])
        _element(app.button, "🚫 Exclude Selected").click()
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], {3})
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            initial_count,
        )
        self.assertEqual(data["_trajectory_stats_signature"], initial_signature)

        # The first reset is a real edit and should produce exactly one miss.
        _element_containing(app.button, "Reset All Exclusions").click()
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], set())
        reset_count = data["_trajectory_stats_recompute_count"]
        reset_signature = data["_trajectory_stats_signature"]
        self.assertEqual(reset_count, initial_count + 1)

        # Repeating Reset All with an already empty set is a cache hit.
        _element_containing(app.button, "Reset All Exclusions").click()
        app.run()
        self.assertFalse(app.exception)
        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["_trajectory_stats_recompute_count"], reset_count)
        self.assertEqual(data["_trajectory_stats_signature"], reset_signature)

    def test_ai_enable_state_survives_navigation_and_plain_reruns(self) -> None:
        app = AppTest.from_file(
            APP_FILE,
            default_timeout=30,
        ).run()
        self.assertFalse(app.exception)
        app.session_state["trajectory_dict"] = {
            "trajectory_0": _trajectory_entry(divisions=[]),
            "trajectory_1": _trajectory_entry(divisions=[]),
        }
        app.session_state["selected_keys"] = {"trajectory_0", "trajectory_1"}
        app.session_state["current_idx"] = 0
        app.run()
        self.assertFalse(app.exception)

        _element(app.toggle, "Enable AI-assisted cleaning").set_value(True)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["ai_cleaning_enabled"])
        self.assertTrue(
            _element(app.toggle, "Enable AI-assisted cleaning").value
        )

        _element(app.button, "Next ➡️").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_idx"], 1)
        self.assertTrue(app.session_state["ai_cleaning_enabled"])
        self.assertTrue(
            _element(app.toggle, "Enable AI-assisted cleaning").value
        )

        _element(app.button, "⬅️ Prev").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_idx"], 0)
        self.assertTrue(app.session_state["ai_cleaning_enabled"])
        self.assertTrue(
            _element(app.toggle, "Enable AI-assisted cleaning").value
        )

        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["ai_cleaning_enabled"])
        self.assertTrue(
            _element(app.toggle, "Enable AI-assisted cleaning").value
        )

    def test_ai_configuration_controls_live_only_in_sidebar(self) -> None:
        app = _app_with_entry(_trajectory_entry(divisions=[]))
        _element(app.toggle, "Enable AI-assisted cleaning").set_value(True)
        app.run()
        self.assertFalse(app.exception)

        sidebar_text_inputs = [element.label for element in app.sidebar.text_input]
        sidebar_selectboxes = [element.label for element in app.sidebar.selectbox]
        sidebar_buttons = [element.label for element in app.sidebar.button]
        sidebar_toggles = [element.label for element in app.sidebar.toggle]
        sidebar_sliders = [element.label for element in app.sidebar.slider]
        sidebar_checkboxes = [element.label for element in app.sidebar.checkbox]
        sidebar_radios = [element.label for element in app.sidebar.radio]
        self.assertIn("Enable AI-assisted cleaning", sidebar_toggles)
        self.assertIn("Model checkpoint", sidebar_selectboxes)
        self.assertIn("Device", sidebar_selectboxes)
        self.assertIn("Scope", sidebar_selectboxes)
        self.assertIn("📦 Load / Reload Model", sidebar_buttons)
        self.assertIn("Division threshold", sidebar_sliders)
        self.assertIn("Noise threshold", sidebar_sliders)
        self.assertIn("Apply divisions", sidebar_checkboxes)
        self.assertIn("Apply exclusions", sidebar_checkboxes)
        self.assertIn("Apply policy", sidebar_radios)

        main_text_inputs = [element.label for element in app.main.text_input]
        main_selectboxes = [element.label for element in app.main.selectbox]
        main_buttons = [element.label for element in app.main.button]
        main_toggles = [element.label for element in app.main.toggle]
        main_sliders = [element.label for element in app.main.slider]
        main_checkboxes = [element.label for element in app.main.checkbox]
        main_radios = [element.label for element in app.main.radio]
        self.assertNotIn("Enable AI-assisted cleaning", main_toggles)
        self.assertNotIn("Model checkpoint", main_selectboxes)
        self.assertNotIn("Custom checkpoint path", main_text_inputs)
        self.assertNotIn("Device", main_selectboxes)
        self.assertNotIn("Scope", main_selectboxes)
        self.assertNotIn("📦 Load / Reload Model", main_buttons)
        self.assertNotIn("Division threshold", main_sliders)
        self.assertNotIn("Noise threshold", main_sliders)
        self.assertNotIn("Apply divisions", main_checkboxes)
        self.assertNotIn("Apply exclusions", main_checkboxes)
        self.assertNotIn("Apply policy", main_radios)

    def test_spine_fragments_keep_one_explicit_cycle_color(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[], exclusions={3})
        )

        traces = _main_plot_traces(app)
        spine_fragments = [
            trace
            for trace in traces
            if "lines" in trace.get("mode", "")
        ]
        self.assertEqual(
            [_trace_ids(trace) for trace in spine_fragments],
            [[0, 1, 2], [4, 5, 6, 7]],
        )
        line_colors = [
            trace.get("line", {}).get("color") for trace in spine_fragments
        ]
        marker_colors = [
            trace.get("marker", {}).get("color") for trace in spine_fragments
        ]
        self.assertTrue(all(isinstance(color, str) and color for color in line_colors))
        self.assertEqual(marker_colors, line_colors)
        self.assertEqual(len(set(line_colors)), 1)

    def test_nonpositive_exclusion_lengths_use_positive_marker_fallbacks(self) -> None:
        entry = _trajectory_entry(divisions=[], exclusions={2, 3})
        trajectory = entry["trajectory"]
        trajectory.loc[trajectory["fov_cell_index"] == 2, "spine_length"] = 0.0
        trajectory.loc[trajectory["fov_cell_index"] == 3, "spine_length"] = -1.0
        app = _app_with_entry(entry)

        traces = _main_plot_traces(app)
        applied = _applied_exclusion_traces(traces)
        self.assertEqual(len(applied), 1)
        marker = applied[0]
        self.assertEqual(_trace_ids(marker), [2, 3])

        display_y = np.asarray(_trace_values(marker, "y"), dtype=float)
        self.assertTrue(np.isfinite(display_y).all())
        self.assertTrue((display_y > 0).all())

        # Plot positioning may use a positive fallback on the logarithmic axis,
        # while marker customdata must retain the original scientific values.
        customdata = np.asarray(
            _trace_values(marker, "customdata"),
            dtype=object,
        )
        self.assertEqual(customdata[:, 0].astype(int).tolist(), [2, 3])
        self.assertEqual(customdata[:, 1].astype(float).tolist(), [0.0, -1.0])
        self.assertEqual(
            customdata[:, 2].tolist(),
            ["fallback position", "fallback position"],
        )

    def test_fluorescence_lines_are_split_at_excluded_observations(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[], exclusions={3})
        )

        _element(app.toggle, "Show Fluorescence").set_value(True)
        app.run()
        self.assertFalse(app.exception)

        traces = _main_plot_traces(app)
        fluorescence = [
            trace for trace in traces if trace.get("name") == "signal_mean"
        ]
        self.assertEqual(len(fluorescence), 2)
        self.assertTrue(
            all("lines" in trace.get("mode", "") for trace in fluorescence)
        )
        self.assertEqual(
            [_trace_ids(trace) for trace in fluorescence],
            [[0, 1, 2], [4, 5, 6, 7]],
        )
        line_colors = [
            trace.get("line", {}).get("color") for trace in fluorescence
        ]
        marker_colors = [
            trace.get("marker", {}).get("color") for trace in fluorescence
        ]
        self.assertTrue(all(isinstance(color, str) and color for color in line_colors))
        self.assertEqual(marker_colors, line_colors)
        self.assertEqual(len(set(line_colors)), 1)

    def test_nonpositive_pending_ai_and_review_markers_use_fallback_y(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            model_config = ModelConfig(
                model_type="cnn",
                hidden_size=8,
                num_layers=1,
                kernel_size=3,
                dropout=0.0,
            )
            save_checkpoint(
                checkpoint,
                build_model(model_config),
                model_config=model_config,
                normalizer=TrajectoryNormalizer(),
                thresholds=(0.5, 0.5),
                metadata={"purpose": "nonpositive overlay fallback test"},
            )

            entry = _trajectory_entry(
                next_length=3.0,
                divisions=[3, 5],
                exclusions={3},
            )
            trajectory = entry["trajectory"]
            trajectory.loc[
                trajectory["fov_cell_index"] == 6,
                "spine_length",
            ] = 0.0
            trajectory.loc[
                trajectory["fov_cell_index"] == 3,
                "spine_length",
            ] = -1.0

            prediction = trajectory.loc[
                :, ["fov_cell_index", "time_h", "spine_length"]
            ].copy()
            prediction.insert(0, "trajectory_key", TRAJECTORY_KEY)
            prediction["pred_division_probability"] = 0.1
            prediction["pred_excluded_probability"] = 0.1
            prediction.loc[
                prediction["fov_cell_index"] == 6,
                "pred_excluded_probability",
            ] = 0.9
            prediction["pred_division_point"] = False
            prediction["pred_excluded_point"] = (
                prediction["pred_excluded_probability"] >= 0.5
            )

            signature = _checkpoint_signature(checkpoint)
            app = _app_with_entry(entry)
            app.session_state["ai_cleaning_enabled"] = True
            app.session_state["ai_checkpoint_path"] = checkpoint
            app.session_state["ai_model_tag"] = "Custom path…"
            app.session_state["ai_checkpoint_custom_path"] = checkpoint
            app.session_state["ai_device"] = "cpu"
            app.session_state["ai_model_signature"] = signature
            app.session_state["ai_prediction_signature"] = signature
            app.session_state["ai_prediction_checkpoint"] = signature[0]
            app.session_state["ai_predictions"] = {TRAJECTORY_KEY: prediction}
            app.session_state["ai_division_threshold"] = 0.5
            app.session_state["ai_excluded_threshold"] = 0.5
            app.run()
            self.assertFalse(app.exception)

            data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
            decision = data["division_resolution"].loc[
                data["division_resolution"]["source_index"] == 3
            ].iloc[0]
            self.assertEqual(decision["source_index"], 3)
            self.assertEqual(decision["division_resolution"], "needs_review")

            traces = _main_plot_traces(app)
            pending = [
                trace for trace in traces if trace.get("name") == "AI exclusion"
            ]
            review = [
                trace
                for trace in traces
                if trace.get("name") == "Division needs review"
            ]
            self.assertEqual(len(pending), 1)
            self.assertEqual(len(review), 1)
            self.assertEqual(_trace_ids(pending[0]), [6])
            self.assertEqual(_trace_ids(review[0]), [3])

            pending_y = np.asarray(_trace_values(pending[0], "y"), dtype=float)
            review_y = np.asarray(_trace_values(review[0], "y"), dtype=float)
            self.assertTrue(np.isfinite(pending_y).all())
            self.assertTrue(np.isfinite(review_y).all())
            self.assertTrue((pending_y > 0).all())
            self.assertTrue((review_y > 0).all())

            pending_custom = np.asarray(
                _trace_values(pending[0], "customdata"),
                dtype=object,
            )
            review_custom = np.asarray(
                _trace_values(review[0], "customdata"),
                dtype=object,
            )
            self.assertEqual(float(pending_custom[0, 2]), 0.0)
            self.assertEqual(float(review_custom[0, 2]), -1.0)

    def test_noop_selective_restore_preserves_ai_undo_snapshot(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[], exclusions={3})
        )
        undo_snapshot = {
            TRAJECTORY_KEY: {
                "div_indices": {7},
                "excluded_indices": set(),
            }
        }
        app.session_state["ai_undo_snapshot"] = undo_snapshot
        before = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        before_count = before["_trajectory_stats_recompute_count"]
        before_signature = before["_trajectory_stats_signature"]

        # ID 2 is valid and selected, but it is not excluded. The restore
        # action is therefore a no-op and must not invalidate AI undo history.
        _select_plot_points(app, [{"x": 2.0, "customdata": [2]}])
        _element_containing(app.button, "Restore Selected Exclusions").click()
        app.run()
        self.assertFalse(app.exception)

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], {3})
        self.assertEqual(app.session_state["ai_undo_snapshot"], undo_snapshot)
        self.assertEqual(
            data["_trajectory_stats_recompute_count"],
            before_count,
        )
        self.assertEqual(data["_trajectory_stats_signature"], before_signature)

    def test_manual_exclusion_has_one_marker_and_breaks_the_line(self) -> None:
        app = _app_with_entry(_trajectory_entry(divisions=[]))

        _select_plot_points(app, [{"x": 3.0, "customdata": 3}])
        _element(app.button, "🚫 Exclude Selected").click()
        app.run()
        self.assertFalse(app.exception)

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], {3})
        traces = _main_plot_traces(app)

        # Excluded ID 3 is a selectable marker, but it is not part of either
        # line trace. Splitting the traces prevents Plotly from drawing a false
        # direct connection from point 2 to point 4.
        applied = _applied_exclusion_traces(traces)
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0].get("mode"), "markers")
        self.assertEqual(_trace_ids(applied[0]), [3])
        self.assertEqual(_line_id_runs(traces), [[0, 1, 2], [4, 5, 6, 7]])

    def test_ai_applied_exclusion_uses_only_the_canonical_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "tiny_model.pt")
            model_config = ModelConfig(
                model_type="cnn",
                hidden_size=8,
                num_layers=1,
                kernel_size=3,
                dropout=0.0,
            )
            save_checkpoint(
                checkpoint,
                build_model(model_config),
                model_config=model_config,
                normalizer=TrajectoryNormalizer(),
                thresholds=(0.5, 0.5),
                metadata={"purpose": "applied exclusion marker test"},
            )

            entry = _trajectory_entry(divisions=[])
            trajectory = entry["trajectory"]
            prediction = trajectory.loc[
                :, ["fov_cell_index", "time_h", "spine_length"]
            ].copy()
            prediction.insert(0, "trajectory_key", TRAJECTORY_KEY)
            prediction["pred_division_probability"] = 0.1
            prediction["pred_excluded_probability"] = 0.1
            prediction.loc[
                prediction["fov_cell_index"] == 3,
                "pred_excluded_probability",
            ] = 0.9
            prediction["pred_division_point"] = False
            prediction["pred_excluded_point"] = (
                prediction["pred_excluded_probability"] >= 0.5
            )

            signature = _checkpoint_signature(checkpoint)
            app = _app_with_entry(entry)
            app.session_state["ai_cleaning_enabled"] = True
            app.session_state["ai_checkpoint_path"] = checkpoint
            app.session_state["ai_model_tag"] = "Custom path…"
            app.session_state["ai_checkpoint_custom_path"] = checkpoint
            app.session_state["ai_device"] = "cpu"
            app.session_state["ai_model_signature"] = signature
            app.session_state["ai_prediction_signature"] = signature
            app.session_state["ai_prediction_checkpoint"] = signature[0]
            app.session_state["ai_predictions"] = {TRAJECTORY_KEY: prediction}
            app.session_state["ai_division_threshold"] = 0.5
            app.session_state["ai_excluded_threshold"] = 0.5
            app.session_state["ai_apply_divisions"] = False
            app.session_state["ai_apply_exclusions"] = True
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(
                _element(app.button, "✅ Apply AI Suggestions").disabled
            )

            _element(app.button, "✅ Apply AI Suggestions").click()
            app.run()
            self.assertFalse(app.exception)

            data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
            self.assertEqual(data["excluded_indices"], {3})
            traces = _main_plot_traces(app)
            applied = _applied_exclusion_traces(traces)
            self.assertEqual(len(applied), 1)
            self.assertEqual(_trace_ids(applied[0]), [3])

            # Once applied, the point must not also remain as a red AI preview
            # marker. There is exactly one exclusion-marker occurrence for ID 3.
            preview = [
                trace for trace in traces if trace.get("name") == "AI exclusion"
            ]
            self.assertTrue(all(3 not in _trace_ids(trace) for trace in preview))
            exclusion_marker_occurrences = sum(
                _trace_ids(trace).count(3)
                for trace in traces
                if trace.get("name") in {"Applied exclusion", "AI exclusion"}
            )
            self.assertEqual(exclusion_marker_occurrences, 1)
            self.assertEqual(_line_id_runs(traces), [[0, 1, 2], [4, 5, 6, 7]])

            # Resetting applied exclusions is a manual label edit, but it must
            # not collapse or disable the enabled AI panel/model selection.
            _element_containing(app.button, "Reset All Exclusions").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(
                _element(app.toggle, "Enable AI-assisted cleaning").value
            )
            self.assertEqual(
                app.session_state["trajectory_dict"][TRAJECTORY_KEY][
                    "excluded_indices"
                ],
                set(),
            )
            self.assertIsNotNone(
                _element(app.button, "📦 Load / Reload Model")
            )
            reset_preview = [
                trace
                for trace in _main_plot_traces(app)
                if trace.get("name") == "AI exclusion"
            ]
            self.assertEqual(
                sum(_trace_ids(trace).count(3) for trace in reset_preview),
                1,
            )

    def test_restore_selected_exclusions_leaves_unselected_ids_excluded(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[], exclusions={2, 4})
        )
        initial_traces = _main_plot_traces(app)
        initial_applied = _applied_exclusion_traces(initial_traces)
        self.assertEqual(len(initial_applied), 1)
        self.assertEqual(set(_trace_ids(initial_applied[0])), {2, 4})

        _select_plot_points(app, [{"x": 2.0, "customdata": [2]}])
        _element_containing(app.button, "Restore Selected Exclusions").click()
        app.run()
        self.assertFalse(app.exception)

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], {4})
        traces = _main_plot_traces(app)
        applied = _applied_exclusion_traces(traces)
        self.assertEqual(len(applied), 1)
        self.assertEqual(_trace_ids(applied[0]), [4])
        self.assertEqual(_line_id_runs(traces), [[0, 1, 2, 3], [5, 6, 7]])

    def test_reset_all_exclusions_clears_every_excluded_id(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[], exclusions={2, 4})
        )

        _element_containing(app.button, "Reset All Exclusions").click()
        app.run()
        self.assertFalse(app.exception)

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["excluded_indices"], set())
        traces = _main_plot_traces(app)
        self.assertEqual(_applied_exclusion_traces(traces), [])
        self.assertEqual(_line_id_runs(traces), [list(range(8))])

    def test_exclusion_shifts_effective_boundary_and_export_is_auditable(self) -> None:
        app = _app_with_entry(_trajectory_entry())

        # Excluding raw division 3 leaves its annotation intact, but the clear
        # 2.0 / 4.0 length drop moves the effective boundary to clean point 2.
        _select_plot_points(app, [{"x": 3.0, "customdata": 3}])
        _element(app.button, "🚫 Exclude Selected").click()
        app.run()
        self.assertFalse(app.exception)

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["div_indices"], [3, 7])
        self.assertEqual(data["excluded_indices"], {3})
        self.assertEqual(data["effective_div_indices"], [2, 7])

        decisions = data["division_resolution"].set_index("source_index")
        self.assertEqual(decisions.loc[3, "effective_index"], 2)
        self.assertEqual(
            decisions.loc[3, "division_resolution"],
            "shifted_previous",
        )
        self.assertAlmostEqual(decisions.loc[3, "length_ratio"], 0.5)
        self.assertEqual(decisions.loc[7, "effective_index"], 7)
        self.assertEqual(decisions.loc[7, "division_resolution"], "unchanged")

        visible = data["trajectory"].loc[
            ~data["trajectory"]["fov_cell_index"].isin({3})
        ]
        expected_segments, expected_division_time, _ = compute_traj_stats(
            visible,
            [2, 7],
        )
        np.testing.assert_allclose(
            data["division_time"],
            expected_division_time,
        )
        self.assertEqual(
            [segment["fov_cell_index"].tolist() for _, segment in data["segments"]],
            [segment["fov_cell_index"].tolist() for _, segment in expected_segments],
        )
        self.assertAlmostEqual(float(data["division_time"][1, 0]), 5.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            _element(app.sidebar.text_input, "Data Directory Path").set_value(
                temporary_directory
            )
            app.run()
            _element(app.sidebar.button, "📊 Export with Labels").click()
            app.run()
            self.assertFalse(app.exception)

            exported = pd.read_csv(
                os.path.join(temporary_directory, "selected_labeled.csv")
            ).sort_values("time_h")

        raw_source = exported.loc[exported["fov_cell_index"] == 3].iloc[0]
        effective_target = exported.loc[exported["fov_cell_index"] == 2].iloc[0]
        self.assertTrue(bool(raw_source["division_point"]))
        self.assertTrue(bool(raw_source["excluded_point"]))
        self.assertFalse(bool(raw_source["effective_division_point"]))
        self.assertFalse(bool(effective_target["division_point"]))
        self.assertFalse(bool(effective_target["excluded_point"]))
        self.assertTrue(bool(effective_target["effective_division_point"]))
        self.assertEqual(
            exported["cell_cycle_index"].tolist(),
            [0, 0, 0, 1, 1, 1, 1, 1],
        )
        # The resolution audit stays in session state (asserted above); the
        # export schema carries canonical labels only.
        for dropped_column in (
            "division_resolution",
            "division_effective_target_index",
            "division_previous_valid_index",
            "division_next_valid_index",
            "division_length_ratio",
        ):
            self.assertNotIn(dropped_column, exported.columns)

    def test_ratio_at_threshold_needs_review_and_has_no_effective_boundary(self) -> None:
        # The rule is deliberately strict: 3.0 / 4.0 == 0.75 is not a clear
        # enough drop to move the division automatically.
        app = _app_with_entry(
            _trajectory_entry(
                next_length=3.0,
                divisions=[3],
                exclusions={3},
            )
        )

        data = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(data["div_indices"], [3])
        self.assertEqual(data["excluded_indices"], {3})
        self.assertEqual(data["effective_div_indices"], [])
        self.assertEqual(data["division_resolution_summary"]["needs_review"], 1)
        decision = data["division_resolution"].iloc[0]
        self.assertEqual(decision["division_resolution"], "needs_review")
        self.assertTrue(pd.isna(decision["effective_index"]))
        self.assertAlmostEqual(float(decision["length_ratio"]), 0.75)
        self.assertEqual(len(data["segments"]), 1)
        self.assertEqual(data["division_time"].size, 0)

    def test_merge_cycles_preserves_latest_raw_boundary(self) -> None:
        app = _app_with_entry(_trajectory_entry(divisions=[3, 7]))
        _select_plot_points(
            app,
            [
                {"x": 3.0, "customdata": [3]},
                {"x": 7.0, "customdata": [7]},
            ],
        )
        _element(app.button, "🔗 Merge Cycles").click()
        app.run()
        self.assertFalse(app.exception)

        after = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(after["div_indices"], [7])
        self.assertEqual(after["effective_div_indices"], [7])
        self.assertEqual(
            after["division_resolution"]["source_index"].tolist(),
            [7],
        )

    def test_merge_cycles_preserves_latest_coalesced_boundary(self) -> None:
        # Sources 3 and 4 form one excluded division run and both resolve onto
        # effective boundary 2. Boundary 1 is selected earlier. Merging must
        # remove source 1 while retaining every raw source represented by the
        # latest selected effective boundary.
        app = _app_with_entry(
            _trajectory_entry(
                divisions=[1, 3, 4, 7],
                exclusions={3, 4},
            )
        )
        before = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(before["effective_div_indices"], [1, 2, 7])
        mapped = before["division_resolution"].set_index("source_index")
        self.assertEqual(mapped.loc[3, "effective_index"], 2)
        self.assertEqual(mapped.loc[4, "effective_index"], 2)

        _select_plot_points(
            app,
            [
                {"x": 1.0, "customdata": [1]},
                {"x": 2.0, "customdata": [2]},
            ],
        )
        _element(app.button, "🔗 Merge Cycles").click()
        app.run()
        self.assertFalse(app.exception)

        after = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(after["div_indices"], [3, 4, 7])
        self.assertEqual(after["excluded_indices"], {3, 4})
        self.assertEqual(after["effective_div_indices"], [2, 7])
        retained = after["division_resolution"].set_index("source_index")
        self.assertEqual(retained.loc[3, "effective_index"], 2)
        self.assertEqual(retained.loc[4, "effective_index"], 2)
        self.assertNotIn(1, retained.index)

    def test_reset_exclusions_restores_raw_source_as_effective(self) -> None:
        app = _app_with_entry(
            _trajectory_entry(divisions=[3, 7], exclusions={3})
        )
        shifted = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(shifted["effective_div_indices"], [2, 7])
        self.assertEqual(
            shifted["division_resolution"].set_index("source_index").loc[
                3, "division_resolution"
            ],
            "shifted_previous",
        )

        _element_containing(app.button, "Reset All Exclusions").click()
        app.run()
        self.assertFalse(app.exception)

        restored = app.session_state["trajectory_dict"][TRAJECTORY_KEY]
        self.assertEqual(restored["div_indices"], [3, 7])
        self.assertEqual(restored["excluded_indices"], set())
        self.assertEqual(restored["effective_div_indices"], [3, 7])
        decisions = restored["division_resolution"].set_index("source_index")
        self.assertEqual(decisions.loc[3, "effective_index"], 3)
        self.assertEqual(decisions.loc[3, "division_resolution"], "unchanged")


if __name__ == "__main__":
    unittest.main()
