"""End-to-end curation regression using the supplied two-FOV pipeline export.

All database/CSV/TIFF writes happen in a temporary copy. The original fixtures
are fingerprinted before and after the test to guard against accidental edits.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile
from streamlit.testing.v1 import AppTest
import streamlit.testing.v1.app_test as app_test_module

from moma_lineage_analysis.curation_store import DB_FILENAME, list_datasets, load_dataset
from moma_lineage_analysis.image_review import (
    get_review_frame, load_review_pickle,
)
from moma_lineage_analysis.image_review_dt import get_dt_meshes
from .test_streamlit_division_resolution_panel import (
    APP_FILE, _element, _select_plot_points,
)
from .test_streamlit_image_review import (
    _gallery_frames, _magic_mock_with_component_registry,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "exported_example"
SOURCE_NAMES = (
    "fov_0_cells_df.csv", "fov_0_exported_data.pkl",
    "fov_1_cells_df.csv", "fov_1_exported_data.pkl",
)


def _fingerprints(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir()) if path.is_file()
    }


def _assert_clean(test, app):
    test.assertFalse(app.exception, [item.value for item in app.exception])
    test.assertFalse(app.error, [item.value for item in app.error])
    test.assertFalse(app.session_state.filtered_state.get("curation_error"))


@unittest.skipUnless(
    all((EXAMPLE_DIR / name).is_file() for name in SOURCE_NAMES),
    "The supplied two-FOV exported_example fixture is unavailable",
)
class ExportedExampleTests(unittest.TestCase):
    def test_real_export_autosave_recovery_sql_and_original_tiff_pair(self):
        before = _fingerprints(EXAMPLE_DIR)
        try:
            with tempfile.TemporaryDirectory() as temporary, patch.object(
                app_test_module, "MagicMock", _magic_mock_with_component_registry,
            ):
                directory = Path(temporary)
                for name in SOURCE_NAMES:
                    shutil.copy2(EXAMPLE_DIR / name, directory / name)

                app = AppTest.from_file(APP_FILE, default_timeout=40).run()
                _element(app.sidebar.text_input, "Data Directory Path").set_value(temporary).run()
                _element(app.sidebar.number_input, "Min Duration (h)").set_value(0.0).run()
                _element(app.sidebar.slider, "Number of Cores").set_value(1).run()
                _element(app.sidebar.button, "Process Data").click().run()
                _assert_clean(self, app)
                entries = app.session_state["trajectory_dict"]
                self.assertEqual(len(entries), 21)
                self.assertEqual(sum(len(entry["trajectory"]) for entry in entries.values()), 441)
                self.assertEqual(len(app.session_state["selected_keys"]), 21)
                self.assertEqual(app.session_state["curation_dataset_id"], 1)
                self.assertTrue((directory / DB_FILENAME).is_file())
                for entry in entries.values():
                    self.assertEqual(entry["_review_status"], "unreviewed")
                    frame = entry["trajectory"]
                    self.assertEqual(len(frame), 21)
                    self.assertAlmostEqual(float(frame.time_h.min()), 0.0)
                    self.assertGreater(float(frame.time_h.max()), 1.0)
                    self.assertLess(float(frame.time_h.max()), 1.01)

                # Image review on both FOVs, including original TIFF export with
                # all display overlays enabled. No output may contain overlays.
                image_results = []
                for key in ("FOV_0_CH_1", "FOV_1_CH_1"):
                    keys = list(app.session_state["trajectory_dict"])
                    app.number_input(key="trajectory_jump_number").set_value(keys.index(key) + 1).run()
                    _assert_clean(self, app)
                    entry = app.session_state["trajectory_dict"][key]
                    indices = entry["trajectory"]["fov_cell_index"].tolist()
                    _select_plot_points(app, [{"customdata":[int(index)]} for index in indices])
                    app.session_state["image_review_viewport_width"] = 1100
                    app.button(key="review_images").click().run()
                    _assert_clean(self, app)
                    self.assertGreater(len(_gallery_frames(app)), 6)
                    app.session_state["image_review_selected_frame"] = "0"
                    app.toggle(key="image_review_fill").set_value(True).run()
                    app.toggle(key="image_review_dt").set_value(True).run()
                    _assert_clean(self, app)
                    self.assertFalse(app.button(key="export_review_image").disabled)
                    snapshot = app.session_state["image_review_snapshot"]
                    export = load_review_pickle(snapshot.pickle_path)
                    raw_image, raw_mask, messages = get_review_frame(export, snapshot.frames[0], "phase")
                    self.assertFalse(messages)
                    meshes, messages = get_dt_meshes(export, snapshot.frames[0])
                    self.assertFalse(messages)
                    self.assertTrue(meshes)
                    app.button(key="export_review_image").click().run()
                    _assert_clean(self, app)
                    fov = key.split("_CH_")[0].lower()
                    image_paths = [
                        path for path in directory.glob(f"{fov}_t0_phase_side1_*.tif")
                        if not path.stem.endswith("_masks")
                    ]
                    self.assertEqual(len(image_paths), 1)
                    image_path = image_paths[0]
                    self.assertEqual(len(image_path.stem.rsplit("_",1)[1]), 8)
                    mask_path = image_path.with_name(image_path.stem + "_masks.tif")
                    saved_image, saved_mask = tifffile.imread(image_path), tifffile.imread(mask_path)
                    self.assertEqual(saved_image.dtype, raw_image.dtype)
                    self.assertEqual(saved_mask.dtype, raw_mask.dtype)
                    np.testing.assert_array_equal(saved_image, raw_image)
                    np.testing.assert_array_equal(saved_mask, raw_mask)
                    image_results.append({"key":key, "shape":list(raw_image.shape),
                                          "dtype":str(raw_image.dtype),
                                          "page_images":len(_gallery_frames(app)),
                                          "dt_ribs":len(meshes[0].ribs)})
                    app.button(key="close_image_review").click().run()
                    _assert_clean(self, app)

                # Save a real manual edit and explicit review confirmation.
                app.number_input(key="trajectory_jump_number").set_value(1).run()
                entry = app.session_state["trajectory_dict"]["FOV_0_CH_1"]
                exclude_id = next(
                    int(index) for index in entry["trajectory"]["fov_cell_index"].iloc[2:-2]
                    if index not in entry["div_indices"]
                )
                _select_plot_points(app, [{"customdata":[exclude_id]}])
                _element(app.button, "🚫 Exclude Selected").click().run()
                _assert_clean(self, app)
                _element(app.button, "Next ➡️").click().run()
                _assert_clean(self, app)
                self.assertEqual(app.session_state["current_idx"], 1)
                app.sidebar.number_input(key="drop_fov_index").set_value(1).run()
                app.button(key="drop_fov_trajectories").click().run()
                _assert_clean(self, app)
                self.assertEqual(len(app.session_state["selected_keys"]), 5)

                _element(app.sidebar.button, "📊 Export with Labels").click().run()
                _assert_clean(self, app)
                expected = pd.read_csv(directory / "selected_labeled.csv")
                for column in ("division_point", "excluded_point", "effective_division_point"):
                    expected[column] = expected[column].astype(int)
                with closing(sqlite3.connect(directory / DB_FILENAME)) as connection:
                    self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                    self.assertFalse(connection.execute("PRAGMA foreign_key_check").fetchall())
                    actual = pd.read_sql_query("SELECT * FROM selected_labeled", connection)
                    self.assertEqual(connection.execute("SELECT count(*) FROM all_labeled").fetchone()[0], 441)
                    self.assertEqual(connection.execute("SELECT count(*) FROM reviewed_selected_labeled").fetchone()[0], 21)
                self.assertEqual(len(actual), 105)
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

                # Fresh session recovery must preserve all measurements and
                # annotations, including the 16 dropped trajectories.
                saved = load_dataset(directory / DB_FILENAME, 1)
                fresh = AppTest.from_file(APP_FILE, default_timeout=40).run()
                _element(fresh.sidebar.text_input, "Data Directory Path").set_value(temporary).run()
                fresh.button(key="resume_curation").click().run()
                _assert_clean(self, fresh)
                self.assertEqual(fresh.session_state["current_idx"], 1)
                self.assertEqual(fresh.session_state["selected_keys"], app.session_state["selected_keys"])
                restored = fresh.session_state["trajectory_dict"]
                self.assertEqual(len(restored), 21)
                for key, entry in app.session_state["trajectory_dict"].items():
                    pd.testing.assert_frame_equal(
                        restored[key]["trajectory"].reset_index(drop=True),
                        entry["trajectory"].reset_index(drop=True),
                    )
                    self.assertEqual(set(restored[key]["div_indices"]), set(entry["div_indices"]))
                    self.assertEqual(restored[key]["original_div_indices"], entry["original_div_indices"])
                    self.assertEqual(restored[key]["excluded_indices"], entry["excluded_indices"])
                    self.assertEqual(restored[key]["_review_status"], entry["_review_status"])
                self.assertEqual(restored["FOV_0_CH_1"]["excluded_indices"], {exclude_id})
                self.assertEqual(restored["FOV_0_CH_1"]["_review_status"], "reviewed")
                _element(fresh.sidebar.button, "Process Data").click().run()
                _assert_clean(self, fresh)
                self.assertEqual(fresh.session_state["curation_dataset_id"], 1)
                self.assertEqual(len(list_datasets(directory / DB_FILENAME)), 1)
                self.assertEqual(fresh.session_state["trajectory_dict"]["FOV_0_CH_1"]["excluded_indices"], {exclude_id})

                print(json.dumps({
                    "input_rows":1622, "fovs":2, "trajectories":21, "tracked_observations":441,
                    "selected_observations":len(actual), "reviewed_observations":21,
                    "database_revision":saved["revision"], "sql_csv_columns":len(actual.columns),
                    "image_reviews":image_results, "fresh_session_recovery":"pass",
                    "sql_csv_parity":"pass", "source_fixture_unchanged":"pass",
                }, sort_keys=True))
        finally:
            self.assertEqual(_fingerprints(EXAMPLE_DIR), before)
