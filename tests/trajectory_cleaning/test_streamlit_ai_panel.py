"""Streamlit smoke test for the AI-assisted trajectory cleaning panel."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from moma_lineage_analysis.moma_logic import compute_traj_stats
from moma_lineage_analysis.paths import PACKAGE_MODELS_RUN_DIR, PACKAGE_ROOT
from moma_lineage_analysis.trajectory_cleaning import (
    ModelConfig,
    TrajectoryNormalizer,
    build_model,
    save_checkpoint,
)
from streamlit.testing.v1 import AppTest

APP_FILE = str(PACKAGE_ROOT / "app" / "Moma_lineage_streamlit.py")


def _synthetic_trajectory_entry() -> dict:
    dataframe = pd.DataFrame(
        {
            "time_h": np.arange(10, dtype=float) / 15.0,
            "spine_length": [2.0, 2.3, 2.7, 3.1, 3.6, 2.0, 2.4, 2.8, 3.2, 3.7],
            "fov_cell_index": np.arange(10),
            "signal_mean": np.arange(10, dtype=float),
        }
    )
    divisions = [4, 9]
    segments, division_time, growth_rate = compute_traj_stats(dataframe, divisions)
    return {
        "trajectory": dataframe,
        "segments": segments,
        "division_time": division_time,
        "growth_rate": growth_rate,
        "div_indices": divisions.copy(),
        "original_div_indices": divisions.copy(),
        "excluded_indices": set(),
    }


def _element(elements, label: str):
    return next(element for element in elements if element.label == label)


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


class StreamlitAIPanelTests(unittest.TestCase):
    def test_checkpoint_preview_apply_and_undo(self) -> None:
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
                metadata={"purpose": "streamlit smoke test"},
            )

            app = AppTest.from_file(
                APP_FILE,
                default_timeout=30,
            ).run()
            self.assertFalse(app.exception)

            app.session_state["trajectory_dict"] = {
                "trajectory_0": _synthetic_trajectory_entry()
            }
            app.session_state["selected_keys"] = {"trajectory_0"}
            app.session_state["current_idx"] = 0
            app.run()
            self.assertFalse(app.exception)

            _element(app.toggle, "Enable AI-assisted cleaning").set_value(True)
            app.run()
            self.assertEqual(
                _element(app.selectbox, "Model checkpoint").value,
                "R010 CNN (merged corpus · production)",
            )
            self.assertEqual(
                app.session_state["ai_checkpoint_path"],
                str(PACKAGE_MODELS_RUN_DIR / "best_model.pt"),
            )
            self.assertAlmostEqual(
                _element(app.slider, "Division threshold").value,
                0.5,
            )
            self.assertAlmostEqual(
                _element(app.slider, "Noise threshold").value,
                0.5,
            )
            self.assertTrue(_element(app.checkbox, "Apply divisions").value)
            self.assertTrue(_element(app.checkbox, "Apply exclusions").value)
            self.assertTrue(
                _element(app.button, "✨ Run AI Preview").disabled,
                "Preview must remain disabled without a loaded checkpoint",
            )

            _element(app.selectbox, "Model checkpoint").set_value("Custom path…")
            app.run()
            _element(app.text_input, "Custom checkpoint path").set_value(checkpoint)
            app.run()
            _element(app.selectbox, "Device").set_value("cpu")
            app.run()
            _element(app.button, "📦 Load / Reload Model").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(_element(app.button, "✨ Run AI Preview").disabled)

            _element(app.button, "✨ Run AI Preview").click()
            app.run()
            self.assertIn("trajectory_0", app.session_state["ai_predictions"])

            # AI controls are conditionally rendered. Their durable values must
            # survive an off/on cycle so the earlier plot overlay cannot read
            # missing threshold keys on the enabling rerun.
            _element(app.slider, "Division threshold").set_value(0.37)
            _element(app.slider, "Noise threshold").set_value(0.63)
            app.run()
            _element(app.toggle, "Enable AI-assisted cleaning").set_value(False)
            app.run()
            _element(app.toggle, "Enable AI-assisted cleaning").set_value(True)
            app.run()
            self.assertFalse(app.exception)
            self.assertAlmostEqual(
                _element(app.slider, "Division threshold").value, 0.37
            )
            self.assertAlmostEqual(
                _element(app.slider, "Noise threshold").value, 0.63
            )

            # A changed checkpoint signature must make an old preview inapplicable.
            checkpoint_stat = os.stat(checkpoint)
            os.utime(
                checkpoint,
                ns=(checkpoint_stat.st_atime_ns, checkpoint_stat.st_mtime_ns + 1_000_000),
            )
            app.run()
            self.assertTrue(
                _element(app.button, "✅ Apply AI Suggestions").disabled
            )
            _element(app.button, "📦 Load / Reload Model").click()
            app.run()
            _element(app.button, "✨ Run AI Preview").click()
            app.run()

            _element(app.slider, "Division threshold").set_value(0.0)
            _element(app.slider, "Noise threshold").set_value(0.0)
            app.run()
            self.assertFalse(_element(app.button, "✅ Apply AI Suggestions").disabled)

            _element(app.button, "✅ Apply AI Suggestions").click()
            app.run()
            applied = app.session_state["trajectory_dict"]["trajectory_0"]
            self.assertEqual(applied["div_indices"], list(range(10)))
            self.assertEqual(applied["excluded_indices"], set(range(10)))
            self.assertFalse(_element(app.button, "↩️ Undo AI Changes").disabled)

            # Overlapping division/exclusion labels remain explicit, while only
            # non-excluded divisions define exported cell-cycle boundaries.
            _element(app.sidebar.text_input, "Data Directory Path").set_value(
                temporary_directory
            )
            app.run()
            _element(app.sidebar.button, "📊 Export with Labels").click()
            app.run()
            exported = pd.read_csv(
                os.path.join(temporary_directory, "selected_labeled.csv")
            )
            self.assertTrue(exported["division_point"].all())
            self.assertTrue(exported["excluded_point"].all())
            self.assertFalse(exported["effective_division_point"].any())
            self.assertEqual(exported["cell_cycle_index"].unique().tolist(), [0])
            # AI predictions and checkpoint provenance are deliberately not
            # part of the export schema (canonical labels only).
            for dropped_column in (
                "pred_division_probability",
                "pred_excluded_probability",
                "pred_division_point",
                "pred_excluded_point",
                "ai_checkpoint_sha256",
                "ai_model_checkpoint",
            ):
                self.assertNotIn(dropped_column, exported.columns)

            # Changing the model selection clears stale previews but must not
            # destroy the ability to undo labels that were already applied.
            _element(app.selectbox, "Device").set_value("auto")
            app.run()
            self.assertIsNotNone(app.session_state["ai_undo_snapshot"])
            self.assertFalse(_element(app.button, "↩️ Undo AI Changes").disabled)

            _element(app.button, "↩️ Undo AI Changes").click()
            app.run()
            restored = app.session_state["trajectory_dict"]["trajectory_0"]
            self.assertEqual(restored["div_indices"], [4, 9])
            self.assertEqual(restored["excluded_indices"], set())
            self.assertFalse(app.exception)

    def test_ai_merge_then_manual_exclusion_and_division(self) -> None:
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
                metadata={"purpose": "manual-after-AI regression test"},
            )

            app = AppTest.from_file(
                APP_FILE,
                default_timeout=30,
            ).run()
            trajectory_entry = _synthetic_trajectory_entry()
            trajectory = trajectory_entry["trajectory"]
            prediction = trajectory.loc[
                :, ["fov_cell_index", "time_h", "spine_length"]
            ].copy()
            prediction.insert(0, "trajectory_key", "trajectory_0")
            prediction["pred_division_probability"] = 0.1
            prediction["pred_excluded_probability"] = 0.1
            prediction.loc[
                prediction["fov_cell_index"] == 1,
                "pred_division_probability",
            ] = 0.9
            prediction.loc[
                prediction["fov_cell_index"] == 3,
                "pred_excluded_probability",
            ] = 0.9
            prediction["pred_division_point"] = (
                prediction["pred_division_probability"] >= 0.5
            )
            prediction["pred_excluded_point"] = (
                prediction["pred_excluded_probability"] >= 0.5
            )
            signature = _checkpoint_signature(checkpoint)

            app.session_state["trajectory_dict"] = {
                "trajectory_0": trajectory_entry
            }
            app.session_state["selected_keys"] = {"trajectory_0"}
            app.session_state["current_idx"] = 0
            app.session_state["ai_cleaning_enabled"] = True
            app.session_state["ai_checkpoint_path"] = checkpoint
            app.session_state["ai_model_tag"] = "Custom path…"
            app.session_state["ai_checkpoint_custom_path"] = checkpoint
            app.session_state["ai_device"] = "cpu"
            app.session_state["ai_model_signature"] = signature
            app.session_state["ai_prediction_signature"] = signature
            app.session_state["ai_prediction_checkpoint"] = signature[0]
            app.session_state["ai_predictions"] = {"trajectory_0": prediction}
            # Reproduce a migrated/conditionally-cleaned session in which a
            # current preview exists but its threshold widget keys do not.
            del app.session_state["ai_division_threshold"]
            del app.session_state["ai_excluded_threshold"]
            app.run()

            self.assertFalse(app.exception)
            self.assertEqual(_element(app.radio, "Apply policy").value, "Merge")
            self.assertFalse(_element(app.button, "✅ Apply AI Suggestions").disabled)
            _element(app.button, "✅ Apply AI Suggestions").click()
            app.run()

            merged = app.session_state["trajectory_dict"]["trajectory_0"]
            self.assertEqual(set(merged["div_indices"]), {1, 4, 9})
            self.assertEqual(merged["excluded_indices"], {3})
            self.assertFalse(_element(app.button, "🚫 Exclude Selected").disabled)
            self.assertFalse(_element(app.button, "➕ Mark Division").disabled)

            app.session_state["traj_plot"] = {
                "selection": {
                    "points": [{"x": 0.0, "customdata": 0}],
                    "point_indices": [0],
                    "box": [],
                    "lasso": [],
                }
            }
            _element(app.button, "🚫 Exclude Selected").click()
            app.run()
            manually_excluded = app.session_state["trajectory_dict"]["trajectory_0"]
            self.assertEqual(manually_excluded["excluded_indices"], {0, 3})
            self.assertIsNone(app.session_state["ai_undo_snapshot"])

            app.session_state["traj_plot"] = {
                "selection": {
                    "points": [{"x": 2.0 / 15.0, "customdata": 2}],
                    "point_indices": [2],
                    "box": [],
                    "lasso": [],
                }
            }
            _element(app.button, "➕ Mark Division").click()
            app.run()
            manually_annotated = app.session_state["trajectory_dict"]["trajectory_0"]
            self.assertEqual(set(manually_annotated["div_indices"]), {1, 2, 4, 9})
            self.assertEqual(manually_annotated["excluded_indices"], {0, 3})
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
