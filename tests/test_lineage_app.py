"""End-to-end AppTest coverage for fresh data-processing defaults."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


APP_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "moma_lineage_analysis"
    / "app"
    / "Moma_lineage_streamlit.py"
)


def element(elements, label):
    return next(item for item in elements if item.label == label)


def shuffled_division_csv(path):
    rows = []
    frames = {
        "A": [[5.0], [2.0, 3.0], [2.5, 3.5]],  # exact-halving root division
        "B": [[4.0], [4.5], [5.0]],  # one non-self child: no division
    }
    for channel, channel_frames in frames.items():
        for frame, lengths in enumerate(channel_frames):
            for cell_index, length in enumerate(lengths):
                rows.append(
                    {
                        "time_s": 7200.0 + frame * 1800.0,
                        "time_point": f"t{frame}.tif",
                        "channels": channel,
                        "cell_index": cell_index,
                        "spine_length": length,
                    }
                )
    shuffled = (
        pd.DataFrame(rows).sample(frac=1.0, random_state=17).reset_index(drop=True)
    )
    shuffled.to_csv(path, index=False)
    return shuffled


class LineageAppRegressionTests(unittest.TestCase):
    def test_fresh_seconds_default_and_shuffled_division_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "FOV_7_cells_df.csv"
            source = shuffled_division_csv(csv_path)
            app = AppTest.from_file(str(APP_FILE), default_timeout=60).run()
            self.assertFalse(app.exception, app.exception)
            self.assertEqual(element(app.selectbox, "Time Unit").value, "s")

            element(app.text_input, "Data Directory Path").set_value(directory)
            element(app.number_input, "Min Duration (h)").set_value(0.0)
            element(app.slider, "Min Spine Length (µm)").set_value(0.0)
            element(app.slider, "Number of Cores").set_value(1)
            element(app.button, "Process Data").click()
            app.run()
            self.assertFalse(app.exception, app.exception)

            trajectories = app.session_state["trajectory_dict"]
            self.assertEqual(set(trajectories), {"FOV_7_CH_A", "FOV_7_CH_B"})
            divided = trajectories["FOV_7_CH_A"]
            trajectory = divided["trajectory"].sort_values("time_h")
            np.testing.assert_allclose(trajectory["time_h"], [0.0, 0.5, 1.0])
            self.assertAlmostEqual(
                trajectory["time_h"].max() - trajectory["time_h"].min(), 1.0
            )

            channel = source[source["channels"] == "A"]
            root_id = int(channel.index[channel["time_point"] == "t0.tif"][0])
            daughter_ids = set(
                channel.index[channel["time_point"] == "t1.tif"].astype(int)
            )
            self.assertEqual(divided["div_indices"], [root_id])
            rows = [trajectory.iloc[index] for index in range(3)]
            self.assertEqual(int(rows[0]["fov_cell_index"]), root_id)
            self.assertEqual(int(rows[0]["ancestor_index"]), root_id)
            self.assertIn(int(rows[1]["fov_cell_index"]), daughter_ids)
            self.assertEqual(int(rows[1]["ancestor_index"]), root_id)
            self.assertEqual(
                int(rows[2]["ancestor_index"]), int(rows[1]["fov_cell_index"])
            )

            control = trajectories["FOV_7_CH_B"]["trajectory"].sort_values("time_h")
            self.assertEqual(trajectories["FOV_7_CH_B"]["div_indices"], [])
            np.testing.assert_allclose(control["time_h"], [0.0, 0.5, 1.0])
            np.testing.assert_array_equal(
                control["ancestor_index"].astype(int),
                control["fov_cell_index"].shift(
                    1, fill_value=control["fov_cell_index"].iloc[0]
                ),
            )


if __name__ == "__main__":
    unittest.main()
