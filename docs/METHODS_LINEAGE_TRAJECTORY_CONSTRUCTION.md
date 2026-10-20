# Methods: Lineage Trajectory Construction

| Contract field | Value |
|---|---|
| Research task | Lineage Trajectory Construction |
| Analysis ID | `lineage_trajectory_construction` |
| Method version | `1.0.0` |
| Status | Active — round `R001` reconciled |
| Created | 2026-10-06 |
| Last reconciled | 2026-10-06 |
| Project root | `G:/python_code/Moma_cellPose` |
| Contract owner | Research team |
| Prior method version | None (previously undocumented legacy code) |

This file is the durable scientific contract for this research task. The active Methods prose, controlled registries, executable code, actual run manifest, and outputs must be reconciled after every analysis round.

## Layer 1 — Active Methods

### Research question and scope

Reconstruct single-cell (mother-cell) lineage trajectories inside each microfluidic side-groove channel from segmented per-frame cell measurements: link cell detections across consecutive time points, detect division events, select one representative lineage path per channel, and compute per-cell-cycle elongation statistics used by downstream curation and population analyses. In scope: `moma_logic.py` (`get_one_trajectory`, `get_cell_lineage`, `tracing_lineage`, `compute_traj_stats`, `process_channel_worker`) and the Streamlit app `Moma_lineage_streamlit.py` that parameterizes and runs them. Out of scope: upstream segmentation (`Moma_CPM_FOV.py`) and downstream curation/statistics on selected trajectories.

### Study design and provenance

Inputs are `*_cells_df.csv` files exported per FOV by the upstream `Moma_CPM_FOV.py` pipeline. Each row is one detected cell in one channel at one time point, with fields `time_s`, `time_point`, `channels`, `cell_index` (positional index within the channel, dead end at top), `spine_length`, `area`, `width`, bounding box, and per-channel fluorescence summaries. At load, the app assigns `fov_cell_index` as the row position within the FOV table, which is the unique cell-observation key for lineage linking.

### Population, analysis unit, and independent replicate

- Observation grain: one detected cell per (FOV, channel, time point).
- Analysis unit: one single-cell lineage trajectory within one channel.
- Independent replicate: channel within FOV within experiment (weighting decisions belong to downstream analysis contracts).
- Inclusion/exclusion rules, in execution order inside `get_one_trajectory`:
  1. **Pre-division spine-length filter (introduced at 1.0.0):** remove every observation with `spine_length < min_spine_length` before any lineage linking or division detection. `min_spine_length = 0` disables the filter.
  2. Optional time window: keep `start_time <= time_h <= stop_time` when set.
  3. Channels with no remaining rows are skipped (`None` returned).
  4. Post-hoc trajectory gate in `process_channel_worker`: trajectory duration must be `>= min_duration` hours.

### Preprocessing and quality control

Per channel: apply the exclusion rules above; convert time to hours via `time_h = (time_s - min(time_s)) / time_scale` with `time_scale` from the `time_unit` parameter (`s`=3600, `min`=60, `h`=1, `d`=1/24). Lineage linking (`get_cell_lineage`) walks consecutive time points and pairs cells by their order along the channel (`cell_index` sorted), assuming the mother stays at the dead-end (top) position. Division detection compares the mother length $L_0$ at $t_0$ with the two top candidate daughters $L_a, L_b$ at $t_1$; a division is accepted when

$$
0 < (L_a + L_b) - L_0 < \theta_{div}\, L_0 \quad \text{and} \quad L_a < 0.9\, L_0
$$

with $\theta_{div}$ = `division_threshold`. Accepted divisions assign both daughters the mother's `fov_cell_index` as `ancestor_index`; otherwise the single next cell inherits it. `tracing_lineage` then walks ancestors backward from every terminal cell and records lineage length, division count, member cell list, and division-point cell list. One trajectory per channel is selected by `mode` (`most_division` or `longest_lineage`). Because the pre-division filter removes rows frame by frame, it changes per-frame cell counts and therefore can change which pairings/divisions the positional heuristic makes for the surviving cells; this is the intended debris-removal behavior, not a bug, but it makes filtered and unfiltered trajectories non-identical.

### Variables and calculations

Per selected trajectory, `compute_traj_stats` splits `time_h` at division events into cell-cycle segments. Per segment with at least 4 non-NaN length observations, the elongation rate is the ordinary least-squares slope

$$
\lambda_{cyc} = \frac{d\,\ln(L)}{d\,t_h}
$$

fitted by `numpy.polyfit(..., deg=1)`, with $R^2$ reported alongside; segments with fewer than 4 points yield NaN. Generation time is the interval between consecutive division-event times.

### Statistical analysis

No inferential statistics at this layer; per-cycle OLS fits only. Downstream filtering of fits (e.g. $R^2$ gates) and group comparisons are governed by downstream analysis contracts.

### Figures and tables

The Streamlit viewer plots `spine_length` (log-scale y) against `time_h` per trajectory, split into cycle segments at effective division points. Exported tables (`selected_plotly.csv` / `selected_labeled.csv`) are produced by the curation layer, not this contract.

### Validation and acceptance criteria

1. With `min_spine_length > 0`, no row of any constructed trajectory has `spine_length < min_spine_length` (smoke-test assertion).
2. The filter executes before `get_cell_lineage` in `get_one_trajectory` (code-order check).
3. A channel whose rows are all filtered out returns `None` without error.
4. The sidebar parameter is wired to `process_channel_worker` so that "Process Data" reconstructs all trajectories with the active threshold.

### Interpretation boundary and limitations

`spine_length` is a segmentation-derived measurement owned by the upstream pipeline. Division events come from a positional/length heuristic, not direct septum observation; removing debris cells before linking changes the pairing context for surviving cells. Very small genuine newborn cells near the channel outlet can fall below a high `min_spine_length` and be dropped. Trajectories built with different `min_spine_length` values are not strictly comparable.

## Layer 2 — Controlled Registry and Constraints

### Terminology registry

| Metric ID | Canonical term | Exact field | Symbol | Unit | Data grain | Formula or algorithm | Interpretation | Limitation | Plot label | Deprecated aliases | Status / introduced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `SPINE_LENGTH` | cell spine length | `spine_length` | $L$ | µm | one cell observation | skeletonized midline length from upstream `Cell` fit | cell body length along its curved axis | debris/partial segmentations yield abnormally small values | Spine Length (µm) | None | Active / 1.0.0 |
| `TIME_H` | elapsed time in hours | `time_h` | $t_h$ | h | one cell observation | `(time_s - min(time_s)) / time_scale` | time relative to first frame of the channel | `time_scale` depends on `time_unit` parameter | Time (h) | None | Active / 1.0.0 |
| `GENERATION_TIME` | generation (cell-cycle) duration | `division_time[1, :]` | $T_g$ | h | one cell cycle | difference of consecutive division-event times | interdivision interval | division timing from positional heuristic | Generation Time (h) | None | Active / 1.0.0 |
| `GROWTH_RATE_CYC` | per-cycle elongation rate | `growth_rate[1, :]` | $\lambda_{cyc}$ | 1/h | one cell cycle | OLS slope of ln(`spine_length`) vs `time_h` per cycle segment | exponential elongation rate | NaN when < 4 valid points; assumes exponential growth within cycle | Growth Rate (1/h) | None | Active / 1.0.0 |

### Parameter registry

| Parameter ID | Code name | Active value | Unit | Role | Selection basis | Sensitivity requirement | Version introduced |
|---|---|---:|---|---|---|---|---|
| `MIN_SPINE_LENGTH` | `min_spine_length` | 1.0 (sidebar default); 0.0 = disabled (function default) | µm | Pre-division exclusion of observations with smaller spine length | User decision, round `R001`; slider bounds adjustable in UI | Changing it shifts pairings/division calls; filtered vs unfiltered results not comparable | 1.0.0 |
| `DIVISION_THRESHOLD` | `division_threshold` | 1.2 (sidebar default) | dimensionless (fraction of $L_0$) | Division acceptance tolerance | Legacy default | Sensitivity check recommended before trusting division counts | 1.0.0 (documented) |
| `TIME_UNIT` | `time_unit` | `d` (sidebar default) | — | Scaling of `time_s` to hours | Acquisition metadata | Wrong unit rescales all rates | 1.0.0 (documented) |
| `TRACKING_MODE` | `mode` | `longest_lineage` (sidebar default) | — | Trajectory selection criterion | Legacy default | Changes which lineage path is returned | 1.0.0 (documented) |
| `MIN_DURATION` | `min_duration` | 10.0 (sidebar default) | h | Post-hoc trajectory acceptance gate | Legacy default | Drops short trajectories | 1.0.0 (documented) |

### Data structure contract

| Object or table | Source | Grain | Unique key | Required fields | Inclusion/exclusion behavior | Version or fingerprint |
|---|---|---|---|---|---|---|
| `*_cells_df.csv` | Upstream `Moma_CPM_FOV.py` export | one row per (FOV, channel, time point, detected cell) | `fov_cell_index` (assigned at load = row position) | `time_s`, `time_point`, `channels`, `cell_index`, `spine_length` | Rows with `spine_length < min_spine_length` excluded before linking; optional time window; empty channel skipped | Not yet fingerprinted |
| `trajectory_dict` entry | `process_channel_worker` | one entry per accepted channel trajectory | dict key `FOV_{name}_CH_{channel}` | `trajectory`, `segments`, `division_time`, `growth_rate`, `div_indices` | Trajectories shorter than `min_duration` rejected | Schema per `moma_logic` LOGIC_VERSION 2.2 |

### Global analysis constraints

- Do not silently redefine, rename, clip, normalize, interpolate, exclude, or aggregate a registered metric.
- Treat repeated observations from one independent unit as dependent.
- Export the exact data used for each plot or table.
- Preserve non-comparable historical outputs under versioned filenames.
- Record all exploratory choices as exploratory or retrospective.
- Delimit display equations with `$$` on separate lines; never use `\[` and `\]`.

### Plot contract

| Plot ID | Metrics | Population | Aggregation | Uncertainty | Axes / transformation | Style / colors | Exported point table |
|---|---|---|---|---|---|---|---|
| `TRAJ_VIEWER` | `SPINE_LENGTH` vs `TIME_H` | one trajectory per view | none (raw points per segment) | none | x = Time (h), y = Spine Length (µm), log y-axis | per-cycle segment colors | not exported at this layer |

### Output registry

| Output ID | Path | Content | Method version | Generated by | Validation status |
|---|---|---|---|---|---|
| `METHODS_CONTRACT` | `METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md` | Living Methods contract | 1.0.0 | scientific-analysis-loop | Validated 2026-10-06 |
| `SMOKE_TEST` | `tests/smoke_min_spine_length_filter.py` | Acceptance-gate smoke test (gates 1–3 + synthetic mechanism check) | 1.0.0 | round `R001` | Passed (ALL GATES PASSED) |
| `ACTUAL_MANIFEST` | `tests/actual_manifest_lineage.json` | Actual run manifest reconstructed from live code | 1.0.0 | `tests/export_actual_manifest_lineage.py` | Reconciled (25 match, 0 mismatch) |
| `ALIGNMENT_REPORT` | `tests/lineage_alignment_report.csv` | Declared-vs-actual audit | 1.0.0 | `audit_method_alignment.py` | match=25, undeclared_actual=4 (evidence keys) |

### Declared method manifest

The JSON below is the machine-readable intended contract used for code-method reconciliation.

<!-- METHOD_MANIFEST_START -->
```json
{
  "analysis_id": "lineage_trajectory_construction",
  "method_version": "1.0.0",
  "entry_points": [
    "Moma_lineage_streamlit.py (Streamlit app, sidebar Parameters)",
    "moma_logic.process_channel_worker",
    "moma_logic.get_one_trajectory"
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
      "trajectories shorter than min_duration rejected"
    ]
  },
  "metrics": {
    "SPINE_LENGTH": "spine_length (um), upstream-owned input field",
    "TIME_H": "time_h (h), (time_s - min(time_s)) / time_scale",
    "GENERATION_TIME": "division_time[1, :] (h), diff of consecutive division times",
    "GROWTH_RATE_CYC": "growth_rate[1, :] (1/h), polyfit slope of ln(spine_length) vs time_h per cycle"
  },
  "parameters": {
    "min_spine_length": {"ui_default": 1.0, "function_default": 0.0, "unit": "um", "ui_widget": "float slider with adjustable min/max bounds"},
    "division_threshold": {"ui_default": 1.2},
    "time_unit": {"ui_default": "d"},
    "mode": {"ui_default": "longest_lineage"},
    "min_duration": {"ui_default": 10.0, "unit": "h"}
  },
  "statistics": {"per_cycle_fit": "OLS via numpy.polyfit deg 1 on ln(spine_length) vs time_h, R^2 reported"},
  "plots": {"TRAJ_VIEWER": "spine_length (log y) vs time_h, per-cycle segments"},
  "outputs": ["streamlit session_state.trajectory_dict", "METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md"],
  "validation_gates": [
    "no trajectory row with spine_length < min_spine_length when filter active",
    "filter executes before get_cell_lineage (code order)",
    "fully filtered channel returns None without error",
    "sidebar parameter reaches process_channel_worker on Process Data"
  ]
}
```
<!-- METHOD_MANIFEST_END -->

## Layer 3 — Analysis Rounds and Reconciliation

### Analysis-round registry

| Round | Date | Type | Question or change | Intended method version | Status | Primary artifacts |
|---|---|---|---|---|---|---|
| `R001` | 2026-10-06 | Intentional method change + retrospective documentation | Add user-controlled pre-division minimum spine-length exclusion (sidebar float slider with adjustable bounds, default 1.0 µm) to the Streamlit lineage pipeline; document the previously undocumented construction method | 1.0.0 | Closed | `moma_logic.py` (LOGIC_VERSION 2.2), `Moma_lineage_streamlit.py`, `tests/smoke_min_spine_length_filter.py`, `tests/actual_manifest_lineage.json`, `tests/lineage_alignment_report.csv`, this contract |

### Current round plan

Request: filter abnormal cells (spine length below a user-set threshold) before division-point finding; expose the threshold in the app's left parameter panel as a float slider whose min/max bounds are set via two input widgets; slider default 1.0 µm. Planned edits: (1) `moma_logic.get_one_trajectory` gains `min_spine_length=0.0` and filters rows right after the defensive copy, before `get_cell_lineage`; (2) `moma_logic.process_channel_worker` gains a trailing `min_spine_length=0.0` keyword and forwards it; `LOGIC_VERSION` bumped to 2.2; (3) `Moma_lineage_streamlit.py` sidebar gains a "Cell Filter" section (two bound inputs + slider) and passes the value into the worker call. Validation: smoke test of `process_channel_worker` on a real channel CSV with threshold 0 vs 1.5 µm (acceptance criteria 1–3), code-order inspection (criterion 2), and wiring inspection for criterion 4; then `validate_methods_file.py`.

### Declared-versus-actual reconciliation

Commands: `python tests\smoke_min_spine_length_filter.py`; `python tests\export_actual_manifest_lineage.py`; `python audit_method_alignment.py --methods METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md --actual-manifest tests\actual_manifest_lineage.json --output tests\lineage_alignment_report.csv --allow-differences`; `python validate_methods_file.py METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md` (all with the `Moma_cellpose` conda interpreter, 2026-10-06).

| Manifest path or method step | Declared | Actual evidence | Status | Resolution | Affected outputs |
|---|---|---|---|---|---|
| Methods contract initialized | Three-layer contract | File created from governed template | Match | Populated in round `R001` | `METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md` |
| `parameters.min_spine_length` (filter before `get_cell_lineage`; UI slider default 1.0 µm; function default 0.0) | Filter rows with `spine_length < min_spine_length` before lineage linking | `get_one_trajectory` signature + filter at `moma_logic.py` lines 113–122 (before `get_cell_lineage` call at line 137); introspected defaults ui=1.0 / function=0.0 | Match | Implemented as declared | `moma_logic.py`, `Moma_lineage_streamlit.py` |
| Validation gate 1 (no short cells in trajectory) | No trajectory row below threshold | Real channel 3 (`fov_0_cells_df.csv`, 821 rows): filtered trajectory min spine_length 3.074 ≥ 1.5 µm | Match | Gate passed in smoke test | `tests/smoke_min_spine_length_filter.py` |
| Validation gate 2 (filter precedes division finding) | Code order + behavioral effect | Code order inspected; synthetic debris case: unfiltered run produced a FALSE division at t2 (1/12 h) and masked the true division; filtered run detected the true division at t3 (1/6 h) | Match | Gate passed; mechanism confirmed | `tests/smoke_min_spine_length_filter.py` |
| Validation gate 3 (fully filtered channel) | Returns `None` without error | `min_spine_length=1e6` → `None`, no exception | Match | Gate passed in smoke test | `tests/smoke_min_spine_length_filter.py` |
| Validation gate 4 (sidebar wiring) | Slider value reaches worker on Process Data | `worker_call_has_kwarg: min_spine_length=min_spine` present at the `process_channel_worker` call; `py_compile` OK on both files | Match | Wired as declared; manual UI run pending with user | `Moma_lineage_streamlit.py` |
| `evidence.*` keys (4) in actual manifest | Not declared | `undeclared_actual` in alignment report | Intentional extra | Round documentation evidence, no metric impact; retained | `tests/actual_manifest_lineage.json` |

Smoke-test side observation (no action): on real `fov_0` channel 13 (14 sub-1.5 µm rows), division count was unchanged (27 → 27) because the short cells were not on the selected lineage path; filter is behaviorally inert when debris does not interfere, as intended.

### Method change log

| Method version | Date | Change | Reason | Comparability impact | Affected code and outputs |
|---|---|---|---|---|---|
| 1.0.0 | 2026-10-06 | First documented contract; introduces `min_spine_length` pre-division exclusion (UI default 1.0 µm) | User request: remove abnormal short cells before division-point finding | Trajectories built with the filter active are not strictly comparable to legacy unfiltered runs (per-frame counts and division calls can differ) | `moma_logic.py`, `Moma_lineage_streamlit.py` |

### Unresolved items and next checks

- Manual UI check pending: launch `streamlit run Moma_lineage_streamlit.py`, confirm the "Cell Filter (pre-division)" section renders (slider + Slider Min/Max inputs), and that changing bounds clamps the slider value correctly.
- No dataset fingerprint recorded for input CSVs yet.
- Sensitivity of division counts to `min_spine_length` (e.g. 0 vs 0.6 vs 1.0 µm) not yet quantified.
