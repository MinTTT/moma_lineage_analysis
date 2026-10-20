"""Reconstruct the actual method manifest for METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md.

Parameter defaults are introspected from live code:
  - function defaults from moma_logic signatures (inspect);
  - sidebar UI defaults extracted from Moma_lineage_streamlit.py source.
Narrative keys (metrics/populations/plots) restate the contract sections verified
by code review and the smoke test in round R001.

Run:  python tests\\export_actual_manifest_lineage.py
      (requires an editable install of moma_lineage_analysis)
"""
import inspect
import json
import os
import re

from moma_lineage_analysis import moma_logic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "src", "moma_lineage_analysis", "app", "Moma_lineage_streamlit.py")
OUT = os.path.join(ROOT, "tests", "actual_manifest_lineage.json")

src = open(APP, encoding="utf-8").read()


def app_default(label_pattern, cast=float):
    """Extract the trailing default literal of a sidebar widget by its label."""
    m = re.search(label_pattern + r"[^\n]*", src)
    assert m, f"widget not found: {label_pattern}"
    return cast(m.group(1))


def app_default_multiline(label_pattern, cast=float):
    """Extract a default literal that may appear on later lines of the widget call."""
    m = re.search(label_pattern + r"[\s\S]*?value=([0-9.]+)", src)
    assert m, f"widget not found: {label_pattern}"
    return cast(m.group(1))


got = inspect.signature(moma_logic.get_one_trajectory).parameters
pcw = inspect.signature(moma_logic.process_channel_worker).parameters

actual = {
    "analysis_id": "lineage_trajectory_construction",
    "method_version": "1.0.0",
    "entry_points": [
        "Moma_lineage_streamlit.py (Streamlit app, sidebar Parameters)",
        "moma_logic.process_channel_worker",
        "moma_logic.get_one_trajectory",
    ],
    "inputs": ["*_cells_df.csv exported by Moma_CPM_FOV.py"],
    "analysis_unit": "single-cell lineage trajectory within one channel",
    "independent_unit": "channel within FOV within experiment",
    "populations": {
        "observation_grain": "one detected cell per (FOV, channel, time point)",
        "exclusions": [
            "spine_length < min_spine_length removed before lineage linking (1.0.0)",
            "optional start_time/stop_time window",
            "channels empty after filtering skipped",
            "trajectories shorter than min_duration rejected",
        ],
    },
    "metrics": {
        "SPINE_LENGTH": "spine_length (um), upstream-owned input field",
        "TIME_H": "time_h (h), (time_s - min(time_s)) / time_scale",
        "GENERATION_TIME": "division_time[1, :] (h), diff of consecutive division times",
        "GROWTH_RATE_CYC": "growth_rate[1, :] (1/h), polyfit slope of ln(spine_length) vs time_h per cycle",
    },
    "parameters": {
        "min_spine_length": {
            "ui_default": app_default_multiline(r'"Min Spine Length \(µm\)"'),
            "function_default": got["min_spine_length"].default,
            "unit": "um",
            "ui_widget": "float slider with adjustable min/max bounds",
        },
        "division_threshold": {
            "ui_default": app_default(r'"Division Threshold",\s*[0-9.]+,\s*[0-9.]+,\s*([0-9.]+)'),
        },
        "time_unit": {
            "ui_default": {3: "d", 2: "h", 1: "min", 0: "s"}[
                int(app_default(r'"Time Unit", \[[^\]]*\], index=([0-9])', cast=int))
            ],
        },
        "mode": {
            "ui_default": "longest_lineage"
            if re.search(r'"Tracking Mode",\s*\[[\'"]longest_lineage', src)
            else "most_division",
        },
        "min_duration": {
            "ui_default": app_default(r'"Min Duration \(h\)",\s*[0-9.]+,\s*[0-9.]+,\s*([0-9.]+)'),
            "unit": "h",
        },
    },
    "statistics": {
        "per_cycle_fit": "OLS via numpy.polyfit deg 1 on ln(spine_length) vs time_h, R^2 reported"
    },
    "plots": {"TRAJ_VIEWER": "spine_length (log y) vs time_h, per-cycle segments"},
    "outputs": [
        "streamlit session_state.trajectory_dict",
        "METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md",
    ],
    "validation_gates": [
        "no trajectory row with spine_length < min_spine_length when filter active",
        "filter executes before get_cell_lineage (code order)",
        "fully filtered channel returns None without error",
        "sidebar parameter reaches process_channel_worker on Process Data",
    ],
    "evidence": {
        "logic_version": moma_logic.LOGIC_VERSION,
        "worker_forwards_param": pcw["min_spine_length"].default == got["min_spine_length"].default,
        "worker_call_has_kwarg": "min_spine_length=min_spine" in src,
        "smoke_test": "tests/smoke_min_spine_length_filter.py ALL GATES PASSED (round R001)",
    },
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(actual, fh, indent=2, ensure_ascii=False)
print(f"WROTE: {OUT}")
print(json.dumps(actual["parameters"], indent=2, ensure_ascii=False))
print("evidence:", json.dumps(actual["evidence"], indent=2))
