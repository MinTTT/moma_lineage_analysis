"""Image-review dialog integration against temporary pipeline-format exports."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock as _OriginalMagicMock

import numpy as np
import pandas as pd
import tifffile
from streamlit.components.v2.component_manager import BidiComponentManager
from streamlit.runtime import Runtime
from streamlit.testing.v1 import AppTest
import streamlit.testing.v1.app_test as app_test_module

from .test_streamlit_division_resolution_panel import (
    APP_FILE,
    _app_with_entries,
    _element,
    _select_plot_points,
    _trajectory_entry,
)


_ORIGINAL_APP_TEST_MAGIC_MOCK = app_test_module.MagicMock


def _magic_mock_with_component_registry(*args, **kwargs):
    """Give AppTest's mocked Runtime a real v2 component registry."""

    mock = _ORIGINAL_APP_TEST_MAGIC_MOCK(*args, **kwargs)
    if kwargs.get("spec") is Runtime:
        mock.bidi_component_registry = BidiComponentManager()
    return mock


def _gallery_payload(app: AppTest) -> dict:
    components = app.get("bidi_component")
    assert len(components) == 1, "Expected exactly one compact image gallery"
    proto = components[0].proto
    assert proto.WhichOneof("data") == "json"
    return json.loads(proto.json)


def _gallery_frames(app: AppTest) -> list[dict]:
    return _gallery_payload(app)["frames"]


def _review_entry(directory: Path, count: int = 8):
    entry = _trajectory_entry(divisions=[3, 7], exclusions={2})
    frame = entry["trajectory"].copy()
    frame["fov_cell_index"] = np.arange(100, 108)
    frame["time_point"] = [f"t{index}.tiff" for index in range(8)]
    frame["time_s"] = np.arange(8, dtype=float) * 3600
    frame["channels"] = 7
    frame["cell_index"] = 2
    entry["trajectory"] = frame
    entry["div_indices"] = [103, 107]
    entry["original_div_indices"] = [103, 107]
    entry["excluded_indices"] = {102}
    source = directory / "fov_000_cells_df.csv"
    frame.to_csv(source, index=False)
    entry["source_csv_path"] = str(source)
    images, masks = {}, {}
    for index in range(count):
        name = f"t{index}.tiff"
        image = np.arange(160 * 30, dtype=np.uint16).reshape(160, 30) + index * 20
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[10:50, 5:20] = 1
        mask[55:110, 5:20] = 2
        images[name] = {"phase": image}
        if index != 1:
            images[name]["green"] = image // 2
        masks[name] = mask
    parameters = {
        name: [
            {"cell_index": 1, "mesh": [[5.0, 10.0, 20.0, 10.0], [5.0, 25.0, 20.0, 25.0], [5.0, 45.0, 20.0, 45.0]]},
            {"cell_index": 2, "mesh": [[5.0, 55.0, 20.0, 55.0], [5.0, 80.0, 20.0, 80.0], [5.0, 105.0, 20.0, 105.0]]},
        ]
        for name in images
    }
    export = {
        "info_dict": {"cell_detection_opticalChannel_name": "phase", "algorithm": "DT"},
        "cells_sideChannel_image_dict": {
            "fov_channels_image_dict": {7: images},
            "fov_cells_mask_dict": {7: masks},
            "fov_cells_pars_dict": {7: parameters},
        },
    }
    with (directory / "fov_000_exported_data.pkl").open("wb") as handle:
        pickle.dump(export, handle)
    return entry


class StreamlitImageReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_test_module.MagicMock = _magic_mock_with_component_registry

    @classmethod
    def tearDownClass(cls):
        app_test_module.MagicMock = _ORIGINAL_APP_TEST_MAGIC_MOCK

    def test_review_requires_selection_and_original_metadata(self):
        app = _app_with_entries({"FOV_0_CH_7": _trajectory_entry()})
        slider = app.select_slider(key="trajectory_jump_slider")
        self.assertGreater(slider.proto.max, slider.proto.min)
        self.assertTrue(slider.disabled)
        app.button(key="review_images").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Select points on the plot first.", [item.value for item in app.info])
        self.assertNotIn("image_review_snapshot", app.session_state)
        _select_plot_points(app, [{"customdata": [2]}])
        app.button(key="review_images").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)
        self.assertNotIn("image_review_snapshot", app.session_state)

    def test_popup_orders_frames_preserves_selection_and_reuses_plot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = _review_entry(Path(temporary_directory))
            app = _app_with_entries({"FOV_0_CH_7": entry})
            cache = app.session_state["trajectory_figure_cache"]
            epoch = app.session_state["trajectory_plot_epoch"]
            labels = (entry["div_indices"].copy(), entry["excluded_indices"].copy())
            _select_plot_points(app, [
                {"customdata": [107]}, {"customdata": [102]},
                {"customdata": 100}, {"customdata": [102]},
            ])
            selection = app.session_state["traj_plot"]
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["image_review_open"])
            initial_images = _gallery_frames(app)
            snapshot = app.session_state["image_review_snapshot"]
            self.assertEqual([frame.time_point for frame in snapshot.frames], ["t0.tiff", "t2.tiff", "t7.tiff"])
            self.assertTrue(snapshot.frames[1].observations[0].excluded)
            self.assertEqual(app.selectbox(key="image_review_optical_channel").value, "phase")
            self.assertEqual(len(_gallery_frames(app)), 3)
            self.assertEqual(app.session_state["trajectory_plot_epoch"], epoch)
            self.assertEqual(app.session_state["traj_plot"], selection)
            self.assertEqual(app.session_state["trajectory_figure_cache"]["build_count"], cache["build_count"])
            app.toggle(key="image_review_fill").set_value(True).run()
            self.assertNotEqual(
                _gallery_frames(app),
                initial_images,
            )
            app.toggle(key="image_review_outline").set_value(False).run()
            app.selectbox(key="image_review_cells").set_value("All cells").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["image_review_snapshot"], snapshot)
            self.assertEqual(app.session_state["trajectory_figure_cache"]["build_count"], cache["build_count"])
            self.assertEqual(entry["div_indices"], labels[0])
            self.assertEqual(entry["excluded_indices"], labels[1])

            # A later live selection must not replace the popup's snapshot.
            _select_plot_points(app, [{"customdata": [101]}])
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["image_review_snapshot"], snapshot)
            app.button(key="close_image_review").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["image_review_open"])
            self.assertNotIn("image_review_snapshot", app.session_state)
            self.assertEqual(app.session_state["trajectory_plot_epoch"], epoch)
            self.assertEqual(app.session_state["trajectory_figure_cache"]["build_count"], cache["build_count"])
            _select_plot_points(app, [{"customdata": [101]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.toggle(key="image_review_fill").value)
            self.assertFalse(app.toggle(key="image_review_outline").value)
            self.assertEqual(app.selectbox(key="image_review_cells").value, "All cells")


    def test_compact_gallery_click_metadata_and_dt_overlay(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = _review_entry(Path(temporary_directory))
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [100]}, {"customdata": [101]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)

            payload = _gallery_payload(app)
            frames = payload["frames"]
            self.assertEqual(len(frames), 2)
            self.assertEqual([frame["width"] for frame in frames], [90, 90])
            self.assertEqual([frame["height"] for frame in frames], [480, 480])
            self.assertFalse(any(item.value.startswith("1. t0.tiff") for item in app.caption))
            self.assertIn("Image: t1.tiff", frames[1]["info"])
            self.assertIn("Time: 1.000 h", frames[1]["info"])
            self.assertIn("Side channel: 7", frames[1]["info"])
            self.assertIn("Optical channel: phase", frames[1]["info"])
            self.assertIn("Cell: 2", frames[1]["info"])

            # AppTest cannot execute the frontend click callback; preserve the
            # state the callback stores and verify the next gallery render uses it.
            app.session_state["image_review_selected_frame"] = "1"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(_gallery_payload(app)["selected"], "1")

            before_dt = [frame["src"] for frame in _gallery_frames(app)]
            app.toggle(key="image_review_dt").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertNotEqual([frame["src"] for frame in _gallery_frames(app)], before_dt)
            self.assertTrue(any("DT overlay" in item.value for item in app.caption))

            app.button(key="close_image_review").click().run()
            _select_plot_points(app, [{"customdata": [102]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.toggle(key="image_review_dt").value)


    def test_export_selected_image_and_mask_tiffs_to_source_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            entry = _review_entry(directory)
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [100]}, {"customdata": [101]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="export_review_image").disabled)

            app.session_state["image_review_selected_frame"] = "1"
            app.toggle(key="image_review_dt").set_value(True).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="export_review_image").disabled)
            app.button(key="export_review_image").click().run()
            self.assertFalse(app.exception)

            outputs = [
                path for path in directory.glob("fov_000_t1_phase_side7_*.tif")
                if not path.stem.endswith("_masks")
            ]
            self.assertEqual(len(outputs), 1)
            output = outputs[0]
            tag = output.stem.rsplit("_", 1)[1]
            self.assertEqual(len(tag), 8)
            mask_output = output.with_name(output.stem + "_masks.tif")
            self.assertTrue(mask_output.is_file())
            with (directory / "fov_000_exported_data.pkl").open("rb") as handle:
                export = pickle.load(handle)
            expected_image = export["cells_sideChannel_image_dict"]["fov_channels_image_dict"][7]["t1.tiff"]["phase"]
            expected_mask = export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"][7]["t1.tiff"]
            pixels = tifffile.imread(output)
            # DT was enabled for the display, but export must be the untouched
            # original 2D source image with no mask/spine/rib pixels burned in.
            self.assertEqual(pixels.dtype, expected_image.dtype)
            np.testing.assert_array_equal(pixels, expected_image)
            mask_pixels = tifffile.imread(mask_output)
            self.assertEqual(mask_pixels.dtype, expected_mask.dtype)
            np.testing.assert_array_equal(mask_pixels, expected_mask)

            app.button(key="export_review_image").click().run()
            self.assertFalse(app.exception)
            all_images = [
                path for path in directory.glob("fov_000_t1_phase_side7_*.tif")
                if not path.stem.endswith("_masks")
            ]
            all_masks = list(directory.glob("fov_000_t1_phase_side7_*_masks.tif"))
            self.assertEqual(len(all_images), 2)
            self.assertEqual(len(all_masks), 2)
            self.assertEqual(len({path.stem.rsplit("_", 1)[1] for path in all_images}), 2)

    def test_layer_change_is_an_actual_dialog_fragment_run(self):
        from ._image_review_fragment_check import run_dialog_fragment

        with tempfile.TemporaryDirectory() as temporary_directory:
            app = _app_with_entries({"FOV_0_CH_7": _review_entry(Path(temporary_directory))})
            _select_plot_points(app, [{"customdata": [102]}, {"customdata": [100]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            before = app.session_state["trajectory_figure_cache"]["build_count"]
            result = run_dialog_fragment(app)
            self.assertFalse(app.exception)
            self.assertTrue(result["fragment_success"])
            self.assertIn("bidi_component", result["delta_types"])
            self.assertNotIn("plotly_chart", result["delta_types"])
            self.assertEqual(app.session_state["trajectory_figure_cache"]["build_count"], before)
            self.assertTrue(app.session_state["image_review_fill"])

    def test_popup_x_and_escape_dismiss_trigger_closes_review(self):
        from ._image_review_fragment_check import dismiss_dialog

        with tempfile.TemporaryDirectory() as temporary_directory:
            app = _app_with_entries({"FOV_0_CH_7": _review_entry(Path(temporary_directory))})
            _select_plot_points(app, [{"customdata": [102]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            result = dismiss_dialog(app)
            self.assertFalse(app.exception)
            self.assertTrue(result["dismiss_trigger_sent"])
            self.assertFalse(app.session_state["image_review_open"])
            self.assertNotIn("image_review_snapshot", app.session_state)

    def test_navigating_to_another_trajectory_closes_stale_review(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = _review_entry(Path(temporary_directory))
            app = _app_with_entries({"FOV_0_CH_7": entry, "FOV_1_CH_8": _trajectory_entry()})
            _select_plot_points(app, [{"customdata": [100]}])
            app.button(key="review_images").click().run()
            self.assertTrue(app.session_state["image_review_open"])
            # Simulate an external cursor change; a stale popup must not follow it.
            app.session_state["current_idx"] = 1
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["image_review_open"])
            self.assertNotIn("image_review_snapshot", app.session_state)


    def test_page_size_uses_measured_gallery_width(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = _review_entry(Path(temporary_directory))
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [index]} for index in range(100, 108)])
            app.session_state["image_review_viewport_width"] = 500
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(_gallery_frames(app)), 5)
            self.assertEqual(_gallery_payload(app)["reportedWidth"], 500)

            app.button(key="image_review_later").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(_gallery_frames(app)), 3)
            self.assertEqual(app.session_state["image_review_page"], 2)

            # A wider block repacks all eight frames and clamps the page number.
            app.session_state["image_review_viewport_width"] = 1100
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(_gallery_frames(app)), 8)
            self.assertEqual(app.session_state["image_review_page"], 1)
            self.assertTrue(app.button(key="image_review_later").disabled)

    def test_popup_paging_and_sparse_optical_channels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = _review_entry(Path(temporary_directory))
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [index]} for index in range(100, 108)])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(_gallery_frames(app)), 6)
            app.button(key="image_review_later").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["image_review_page"], 2)
            self.assertEqual(len(_gallery_frames(app)), 2)
            self.assertTrue(app.button(key="image_review_later").disabled)
            app.button(key="image_review_earlier").click().run()
            app.selectbox(key="image_review_optical_channel").set_value("green").run()
            self.assertFalse(app.exception)
            frames = _gallery_frames(app)
            self.assertEqual(len(frames), 6)
            self.assertEqual(sum(frame["src"] is not None for frame in frames), 5)
            self.assertEqual(sum(frame["src"] is None for frame in frames), 1)
            self.assertTrue(any("green" in item.value for item in app.warning))

    def test_missing_mask_preserves_image_and_reports_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            entry = _review_entry(directory)
            path = directory / "fov_000_exported_data.pkl"
            with path.open("rb") as handle:
                export = pickle.load(handle)
            export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"] = {}
            with path.open("wb") as handle:
                pickle.dump(export, handle)
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [100]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(_gallery_frames(app)), 1)
            self.assertTrue(any("mask" in item.value.lower() for item in app.warning))

    def test_missing_pickle_has_closeable_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = _review_entry(Path(temporary_directory))
            expected_pickle = Path(temporary_directory) / "fov_000_exported_data.pkl"
            expected_pickle.unlink()
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [100]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Image review is unavailable" in item.value for item in app.warning))
            app.button(key="close_image_review").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["image_review_open"])

    def test_popup_refreshes_cached_pickle_after_file_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            entry = _review_entry(directory)
            app = _app_with_entries({"FOV_0_CH_7": entry})
            _select_plot_points(app, [{"customdata": [100]}])
            app.button(key="review_images").click().run()
            self.assertFalse(app.exception)
            initial_image = _gallery_frames(app)[0]["src"]
            path = directory / "fov_000_exported_data.pkl"
            old_stat = path.stat()
            with path.open("rb") as handle:
                export = pickle.load(handle)
            export["cells_sideChannel_image_dict"]["fov_channels_image_dict"][7]["t0.tiff"]["phase"] = np.zeros((160, 30), dtype=np.uint16)
            with path.open("wb") as handle:
                pickle.dump(export, handle)
            os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1000000000))
            app.run()
            self.assertFalse(app.exception)
            self.assertNotEqual(_gallery_frames(app)[0]["src"], initial_image)

    def test_process_data_keeps_source_path_outside_measurements(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "fov_012_cells_df.csv"
            rows = [
                {"time_s": float(index), "time_point": f"t{index}.tiff", "channels": channel,
                 "cell_index": 1, "spine_length": 2.0 + index * 0.5}
                for channel in (3, 8) for index in range(4)
            ]
            pd.DataFrame(rows).to_csv(source, index=False)
            app = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(app.sidebar.text_input, "Data Directory Path").set_value(temporary_directory).run()
            _element(app.sidebar.number_input, "Min Duration (h)").set_value(0.0).run()
            _element(app.sidebar.slider, "Number of Cores").set_value(1).run()
            # Simulate a stale review from previously loaded data.
            app.session_state["image_review_open"] = True
            app.session_state["image_review_snapshot"] = "old"
            _element(app.sidebar.button, "Process Data").click().run()
            self.assertFalse(app.exception)
            entries = app.session_state["trajectory_dict"]
            self.assertEqual(len(entries), 2)
            for entry in entries.values():
                self.assertEqual(entry["source_csv_path"], str(source.resolve()))
                self.assertNotIn("source_csv_path", entry["trajectory"].columns)
            self.assertFalse(app.session_state["image_review_open"])
            self.assertNotIn("image_review_snapshot", app.session_state)
