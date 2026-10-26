"""Smoke test for the min_spine_length pre-division filter (moma_logic LOGIC_VERSION 2.3).

Introduced as evidence for round R001 of METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md;
retained as a regression command for method 1.1.0, round R002. A run is required
to establish a current outcome; the historical R001 result is not a fresh pass.
Acceptance gates checked here:
  1. With min_spine_length > 0, no constructed-trajectory row has spine_length < threshold.
  3. A channel whose rows all fall below the threshold returns None without error.
  (Gate 2 - filter precedes get_cell_lineage - is a code-order inspection in
  moma_logic.get_one_trajectory; division-count comparison below is behavioral evidence.)

Run:  python tests\smoke_min_spine_length_filter.py
      (requires an editable install of moma_lineage_analysis)
"""
import os

import numpy as np
import pandas as pd

from moma_lineage_analysis import moma_logic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "tests", "data", "cells_statistics", "Exp1", "fov_0_cells_df.csv")
THRESH = 1.5  # um; above the UI default (1.0) so the test is conservative

print(f"LOGIC_VERSION = {moma_logic.LOGIC_VERSION}")
df = pd.read_csv(CSV)
df["fov_cell_index"] = np.arange(len(df))

base, filt, chan_used, cells_used = None, None, None, None
for chan in df["channels"].unique()[:10]:
    cells = df[df["channels"] == chan].copy()
    base = moma_logic.process_channel_worker(
        "0", chan, cells, 1.2, "s", "longest_lineage", 0.0, min_spine_length=0.0
    )
    filt = moma_logic.process_channel_worker(
        "0", chan, cells, 1.2, "s", "longest_lineage", 0.0, min_spine_length=THRESH
    )
    if base is not None:
        chan_used, cells_used = chan, cells
        break

assert chan_used is not None, "no usable channel found in first 10 channels"
n_short = int((cells_used["spine_length"] < THRESH).sum())
print(
    f"channel={chan_used}: rows={len(cells_used)}, "
    f"spine_length range=({cells_used['spine_length'].min():.3f}, "
    f"{cells_used['spine_length'].max():.3f}) um, rows below {THRESH} um: {n_short}"
)

# Gate 3: all rows below an absurd threshold -> None, no exception
none_ret = moma_logic.process_channel_worker(
    "0", chan_used, cells_used, 1.2, "s", "longest_lineage", 0.0, min_spine_length=1e6
)
assert none_ret is None, "GATE 3 FAILED: fully filtered channel should return None"
print("GATE 3 PASS: fully filtered channel returns None without error")

# Gate 1: filtered trajectory contains no short cells
assert filt is not None, f"filtered channel {chan_used} unexpectedly returned None"
min_len = float(filt[1]["trajectory"]["spine_length"].min())
assert min_len >= THRESH, f"GATE 1 FAILED: trajectory has spine_length {min_len:.3f} < {THRESH}"
print(f"GATE 1 PASS: filtered trajectory min spine_length = {min_len:.3f} >= {THRESH}")

# Behavioral evidence for gate 2: division outcomes respond to the pre-division filter
d_base = len(base[1]["div_indices"])
d_filt = len(filt[1]["div_indices"])
short_in_base = int((base[1]["trajectory"]["spine_length"] < THRESH).sum())
print(
    f"divisions: unfiltered={d_base}, filtered({THRESH} um)={d_filt}; "
    f"unfiltered trajectory rows below {THRESH} um: {short_in_base}"
)

# Synthetic mechanism check: a short debris cell at the channel top at the frame
# before a real division causes a FALSE division (mother -> debris + mother) and
# masks the true division. The pre-division filter must restore the true division.
# Frames are 300 s apart; mother divides between t3 (L=3.0) and t4 (1.6 + 1.5).
rows = []
spec = [
    (1, [(0, 2.0)]),
    (2, [(0, 2.5)]),
    (3, [(0, 0.4), (1, 3.0)]),  # debris (idx 0, 0.4 um) above mother (idx 1)
    (4, [(0, 1.6), (1, 1.5)]),
    (5, [(0, 1.8), (1, 1.7)]),
    (6, [(0, 2.0), (1, 1.9)]),
]
for frame, cells_spec in spec:
    for cidx, length in cells_spec:
        rows.append(
            dict(time_s=frame * 300.0, time_point=f"t{frame}.tif",
                 channels="syn", cell_index=cidx, spine_length=length)
        )
syn = pd.DataFrame(rows)
syn["fov_cell_index"] = np.arange(len(syn))

syn_base = moma_logic.process_channel_worker(
    "syn", "syn", syn, 1.2, "s", "longest_lineage", 0.0, min_spine_length=0.0
)
syn_filt = moma_logic.process_channel_worker(
    "syn", "syn", syn, 1.2, "s", "longest_lineage", 0.0, min_spine_length=1.0
)
assert syn_base is not None and syn_filt is not None

def div_hours(ret):
    traj = ret[1]["trajectory"]
    divs = traj[traj["fov_cell_index"].isin(ret[1]["div_indices"])]["time_h"].values
    return sorted(divs.tolist())

t_base, t_filt = div_hours(syn_base), div_hours(syn_filt)
print(f"synthetic division times (h): unfiltered={t_base}, filtered(1.0 um)={t_filt}")
# time_h subtracts min(time_s): frame n -> (n-1)*300/3600 h. True division mother is
# at t3 -> 2*300/3600 = 1/6 h; debris-induced false division is at t2 -> 1/12 h.
assert t_filt == [2 * 300 / 3600], (
    f"GATE 2 FAILED: filtered run should detect the true division at t=1/6 h, got {t_filt}"
)
assert t_base == [1 * 300 / 3600], (
    f"GATE 2 FAILED: unfiltered run should show the debris-induced false division at t=1/12 h, got {t_base}"
)
print("GATE 2 PASS: without the filter, debris causes a false division at t2 (1/12 h) "
      "and masks the true division; with the filter the true division at t3 (1/6 h) is found")

print("ALL GATES PASSED")
