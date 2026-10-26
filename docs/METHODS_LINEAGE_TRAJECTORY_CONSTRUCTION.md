# Methods: Lineage Trajectory Construction

| Contract field | Value |
|---|---|
| Research task | Lineage Trajectory Construction |
| Analysis ID | `lineage_trajectory_construction` |
| Method version | `1.1.0` |
| Status | Active — round `R002` reconciled |
| Created | 2026-10-06 |
| Last reconciled | 2026-10-26 (R002 full-suite and manifest reconciliation passed) |
| Project root | `G:/python_code/moma_lineage_analysis` |
| Contract owner | Research team |
| Prior method version | `1.0.0` (R001; previously undocumented legacy code) |

This file is the durable scientific contract for this research task. The active Methods prose, controlled registries, executable code, actual run manifest, and outputs must be reconciled after every analysis round.

## Layer 1 — Active Methods

### Research question and scope

Reconstruct single-cell (mother-cell) lineage trajectories inside each microfluidic side-groove channel from segmented per-frame cell measurements: link cell detections across consecutive time points, detect division events, select one representative lineage path per channel, and compute per-cell-cycle elongation statistics used by downstream curation and population analyses. In scope: `moma_logic.py` (`get_one_trajectory`, `get_cell_lineage`, `tracing_lineage`, `compute_traj_stats`, `process_channel_worker`) and the Streamlit app `Moma_lineage_streamlit.py` that parameterizes and runs them. Out of scope: upstream segmentation (`Moma_CPM_FOV.py`) and downstream curation/statistics on selected trajectories.

### Study design and provenance

Inputs are `*_cells_df.csv` files exported per FOV by the upstream `Moma_CPM_FOV.py` pipeline. Each row is one detected cell in one channel at one time point, with fields `time_s`, `time_point`, `channels`, `cell_index` (positional index within the channel, dead end at top), `spine_length`, `area`, `width`, bounding box, and per-channel fluorescence summaries. At load, the app assigns `fov_cell_index` as the row position within the FOV table. It identifies one observation within that loaded FOV, not a persistent cell identity; `ancestor_index` supplies links between observations.

Upstream unit evidence, inspected 2026-10-26 in `G:/python_code/Moma_cellPose`: `cell_coord_utility.py:1156–1157` multiplies the fitted spine arc length by `umppx`, `:1163` multiplies area by `umppx**2`, and `:1143` calibrates width. The DT implementation in `cell_morphology_compat.py:179–186` likewise sets `spine_length = _length_px * umppx`, area in µm², and widths in µm. `Moma_CPM_FOV.py:439–468` exports these attributes unchanged and puts `time_points_seconds` into `time_s`; `moma_dt_pipeline_utils.py:95–134` does the same for DT exports. Thus spine length and width already use µm, area uses µm², and acquisition time uses seconds. R002 corrects documentation of these units without rescaling numerical lengths or changing the minimum-length filter.

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

Per channel: apply the exclusion rules above; convert time to hours via `time_h = (time_s - min(time_s)) / time_scale` with `time_scale` from the `time_unit` parameter (`s`=3600, `min`=60, `h`=1, `d`=1/24). Fresh app sessions and the function default use `s`, matching upstream acquisition time; explicitly selected `d` remains supported. Any unsupported `time_unit` raises `ValueError` before filtering or reconstruction. Lineage linking (`get_cell_lineage`) walks consecutive time points and pairs cells by their order along the channel (`cell_index` sorted), assuming the mother stays at the dead-end (top) position. Division detection compares the mother length $L_0$ at $t_0$ with the two top candidate daughters $L_a, L_b$ at $t_1$; a division is accepted when

$$
0 \le (L_a + L_b) - L_0 < \theta_{div}\, L_0 \quad \text{and} \quad L_a < 0.9\, L_0
$$

with $\theta_{div}$ = `division_threshold`. The inclusive lower bound accepts exact conservation, including two exact half-length daughters; the upper tolerance and first-daughter 0.9 bound remain strict. Accepted divisions assign both daughters the mother's `fov_cell_index` as `ancestor_index`; otherwise the single next cell inherits it. `tracing_lineage` then walks ancestors backward from every terminal cell and records lineage length, division count, member cell list, and division-point cell list. A mother is a division point when it has more than one non-self child. The root self-edge is excluded from that child count, and the division check runs before stopping at a root, so a founder division is counted while a root with one continuation is not. One trajectory per channel is selected by `mode` (`most_division` or `longest_lineage`). Because the pre-division filter removes rows frame by frame, it changes per-frame cell counts and therefore can change which pairings/divisions the positional heuristic makes for the surviving cells; this is the intended debris-removal behavior, not a bug, but it makes filtered and unfiltered trajectories non-identical.

### Variables and calculations

`get_one_trajectory` returns a stable ascending-`time_h` trajectory. `compute_traj_stats` works on a defensive copy, converts `time_h` and `spine_length` to numeric, and stably sorts by time. Rows with nonfinite or nonnumeric time cannot belong to time segments and are omitted from segmentation with a `RuntimeWarning`. Rows with finite time but nonfinite, nonnumeric, zero, or negative length remain in their segments and are excluded only from fitting, also with a `RuntimeWarning`; values are not imputed.

Each segment ends on its division-mother observation; the next observation starts the next segment. Per segment with at least 4 observations having finite time and finite positive length, the elongation rate is the ordinary least-squares slope

$$
\lambda_{cyc} = \frac{d\,\ln(L)}{d\,t_h}
$$

fitted by `numpy.polyfit(..., deg=1, full=True)` after subtracting the first valid fit time. The fit must have rank 2 and finite coefficients. For nonconstant log length, $R^2 = 1 - \sum_i (\ln L_i - \widehat{\ln L_i})^2 / \sum_i (\ln L_i - \overline{\ln L})^2$ is computed explicitly from residuals. The reported fit time is the midpoint of the first and last valid fit times. Segments with fewer than 4 valid observations yield NaNs. With at least 4 valid observations but no time variation, the function emits a `RuntimeWarning` and returns NaNs; constant positive length over varying times has slope 0 and undefined (NaN) $R^2$. A numerical fitting failure, deficient rank, or nonfinite coefficients emits a `RuntimeWarning` and returns NaNs for that fit; only expected numerical exceptions are handled, with no broad exception swallow. Generation time is the interval between consecutive finite division-event times.

### Statistical analysis

No inferential statistics at this layer; per-cycle OLS fits only. Downstream filtering of fits (e.g. $R^2$ gates) and group comparisons are governed by downstream analysis contracts.

### Figures and tables

The Streamlit viewer plots `spine_length` (log-scale y) against `time_h` per trajectory, split into cycle segments at effective division points. Exported tables (`selected_plotly.csv` / `selected_labeled.csv`) are produced by the curation layer, not this contract.

### Validation and acceptance criteria

1. With `min_spine_length > 0`, no row of any constructed trajectory has `spine_length < min_spine_length` (smoke-test assertion).
2. The filter executes before `get_cell_lineage` in `get_one_trajectory` (code-order check).
3. A channel whose rows are all filtered out returns `None` without error.
4. The sidebar parameter is wired to `process_channel_worker` so that "Process Data" reconstructs all trajectories with the active threshold.
5. A founder with two non-self children contributes one division; a root self-edge plus one continuation contributes none.
6. Exact daughter-length conservation is accepted, including exact halving; a negative length difference, equality to the upper tolerance, or equality to the first-daughter 0.9 bound is rejected.
7. Seconds-valued source times convert to hours with the fresh UI default `s`; explicit `min`, `h`, and `d` conversions remain supported and unsupported units raise `ValueError`.
8. Shuffled trajectories and numeric time strings produce sorted, reproducible cycle membership and known exponential-growth slopes without mutating the input.
9. Nonfinite-time rows are excluded from segments with a `RuntimeWarning`; invalid/nonpositive lengths remain in segments but are excluded from fits with a `RuntimeWarning`. Fewer than 4 valid observations produce NaNs; repeated identical times warn and produce NaNs; constant lengths produce zero slope and NaN $R^2$.
10. Fits use centered time, rank checks, explicit residual $R^2$, and reported numerical failures. App statistics cache schema 2 invalidates stale cached statistics.

Regression commands (conda interpreter; R002 validation completed 2026-10-26):

```powershell
$py = 'G:\miniconda3\envs\Moma_cellpose\python.exe'
& $py -m pytest tests\test_moma_logic.py tests\test_lineage_app.py -q
& $py tests\smoke_min_spine_length_filter.py
& $py -m pytest tests\trajectory_cleaning\test_contract_integrity.py -q
```

The exporter records defaults, versions, and regression command references; running it does not execute regression tests or establish a fresh pass. R001 outcomes below remain historical.

### Interpretation boundary and limitations

`spine_length` is a segmentation-derived measurement owned by the upstream pipeline. Division events come from a positional/length heuristic, not direct septum observation; removing debris cells before linking changes the pairing context for surviving cells. Very small genuine newborn cells near the channel outlet can fall below a high `min_spine_length` and be dropped. Trajectories built with different `min_spine_length` values are not strictly comparable.

Method 1.1.0 can change founder division counts, selected paths (`most_division`), cycle boundaries, generation times, and fits relative to 1.0.0, particularly for exact-halving events. Correcting the fresh default from `d` to `s` affects duration gates and rates for seconds-valued inputs: processing seconds as days inflated elapsed hours by 86,400 and reduced rates by that factor. Preserve the historical outputs and parameters; use **Process Data** to reconstruct with logic 2.3 and the correct source unit before comparing or exporting new analyses. App statistics schema 2 refreshes cached statistics, but it does not reconstruct lineage links or migrate old annotations. Existing manual/AI annotations are not migrated onto newly constructed paths and require review after reconstruction. This round retains the positional tracker and does not train, promote, or change frozen datasets or model checkpoints.

## Layer 2 — Controlled Registry and Constraints

### Terminology registry

| Metric ID | Canonical term | Exact field | Symbol | Unit | Data grain | Formula or algorithm | Interpretation | Limitation | Plot label | Deprecated aliases | Status / introduced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `SPINE_LENGTH` | cell spine length | `spine_length` | $L$ | µm | one cell observation | upstream `Cell` fitted spine arc length or DT spine length, calibrated by `umppx` before export | cell body length along its curved axis | debris/partial segmentations yield abnormally small values | Spine Length (µm) | None | Active / 1.0.0 |
| `TIME_H` | elapsed time in hours | `time_h` | $t_h$ | h | one cell observation | `(time_s - min(time_s)) / time_scale` | time relative to first frame of the channel | `time_scale` depends on `time_unit` parameter | Time (h) | None | Active / 1.0.0 |
| `GENERATION_TIME` | generation (cell-cycle) duration | `division_time[1, :]` | $T_g$ | h | one cell cycle | difference of consecutive division-event times | interdivision interval | division timing from positional heuristic | Generation Time (h) | None | Active / 1.0.0 |
| `GROWTH_RATE_CYC` | per-cycle elongation rate | `growth_rate[1, :]` | $\lambda_{cyc}$ | 1/h | one cell cycle | OLS slope of ln(`spine_length`) vs `time_h` per cycle segment | exponential elongation rate | NaN when < 4 finite positive-length observations or no time variation; constant length has zero slope and NaN R²; assumes exponential growth within cycle | Growth Rate (1/h) | None | Active / 1.0.0 |

### Parameter registry

| Parameter ID | Code name | Active value | Unit | Role | Selection basis | Sensitivity requirement | Version introduced |
|---|---|---:|---|---|---|---|---|
| `MIN_SPINE_LENGTH` | `min_spine_length` | 1.0 (sidebar default); 0.0 = disabled (function default) | µm | Pre-division exclusion of observations with smaller spine length | User decision, round `R001`; slider bounds adjustable in UI | Changing it shifts pairings/division calls; filtered vs unfiltered results not comparable | 1.0.0 |
| `DIVISION_THRESHOLD` | `division_threshold` | 1.2 (sidebar default) | dimensionless (fraction of $L_0$) | Division acceptance tolerance | Legacy default | Sensitivity check recommended before trusting division counts | 1.0.0 (documented) |
| `TIME_UNIT` | `time_unit` | `s` (fresh sidebar and function default); explicit `min`, `h`, `d` supported | — | Scaling of source time to hours; unsupported units raise `ValueError` | Upstream `time_points_seconds` export; corrected default in 1.1.0 | Wrong unit rescales durations, duration filtering, and rates; reconstruct prior seconds-valued inputs processed as `d` | 1.0.0 (documented); default corrected 1.1.0 |
| `TRACKING_MODE` | `mode` | `longest_lineage` (sidebar default) | — | Trajectory selection criterion | Legacy default | Changes which lineage path is returned | 1.0.0 (documented) |
| `MIN_DURATION` | `min_duration` | 10.0 (sidebar default) | h | Post-hoc trajectory acceptance gate | Legacy default | Drops short trajectories | 1.0.0 (documented) |

### Data structure contract

| Object or table | Source | Grain | Unique key | Required fields | Inclusion/exclusion behavior | Version or fingerprint |
|---|---|---|---|---|---|---|
| `*_cells_df.csv` | Upstream `Moma_CPM_FOV.py` export | one row per (FOV, channel, time point, detected cell) | `fov_cell_index` (assigned at load = row position) | `time_s`, `time_point`, `channels`, `cell_index`, `spine_length` | Rows with `spine_length < min_spine_length` excluded before linking; optional time window; empty channel skipped | Not yet fingerprinted |
| `trajectory_dict` entry | `process_channel_worker` | one entry per accepted channel trajectory | dict key `FOV_{name}_CH_{channel}` | `trajectory`, `segments`, `division_time`, `growth_rate`, `div_indices` | Trajectories shorter than `min_duration` rejected | Schema per `moma_logic` LOGIC_VERSION 2.3; app statistics cache schema 2 |

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

Current R002 artifacts:

| Output ID | Path | Content | Method version | Generated by | Validation status |
|---|---|---|---|---|---|
| `METHODS_CONTRACT_R002` | `docs/METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md` | Current construction contract, units evidence, historical impact, regression gates | 1.1.0 | round `R002` | Reconciled 2026-10-26; full suite and manifest alignment passed |
| `REGRESSION_R002` | `tests/test_moma_logic.py`, `tests/test_lineage_app.py`, `tests/smoke_min_spine_length_filter.py` | Deterministic logic/UI regressions and retained filter smoke | 1.1.0 | round `R002` | Passed in full suite: 177 tests, 57 subtests; 3 skips; retained filter smoke passed |
| `ACTUAL_MANIFEST_R002` | `tests/actual_manifest_lineage.json` | Introspected defaults/versions and reviewed rule narrative | 1.1.0 | `tests/export_actual_manifest_lineage.py` | Regenerated 2026-10-26 with conda Python; all 18 non-evidence top-level fields match declared JSON by standard-library assertion; no test outcomes inferred |

Historical R001 output registry (statuses as recorded on 2026-10-06; the actual-manifest path now holds R002, with its R001 evidence retained under `evidence.historical_r001`):

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
  "method_version": "1.1.0",
  "entry_points": [
    "Moma_lineage_streamlit.py (Streamlit app, sidebar Parameters)",
    "moma_logic.process_channel_worker",
    "moma_logic.get_one_trajectory"
  ],
  "inputs": [
    "*_cells_df.csv exported by Moma_CPM_FOV.py"
  ],
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
    "SPINE_LENGTH": "spine_length (um), calibrated upstream before CSV export",
    "TIME_H": "time_h (h), (time_s - min(time_s)) / time_scale",
    "GENERATION_TIME": "division_time[1, :] (h), diff of consecutive division times",
    "GROWTH_RATE_CYC": "growth_rate[1, :] (1/h), centered-time OLS slope of ln(spine_length) per cycle using finite positive lengths"
  },
  "parameters": {
    "min_spine_length": {
      "ui_default": 1,
      "function_default": 0,
      "unit": "um",
      "ui_widget": "float slider with adjustable min/max bounds"
    },
    "division_threshold": {
      "ui_default": 1.2
    },
    "time_unit": {
      "ui_default": "s",
      "function_default": "s",
      "supported": [
        "s",
        "min",
        "h",
        "d"
      ],
      "invalid": "ValueError"
    },
    "mode": {
      "ui_default": "longest_lineage"
    },
    "min_duration": {
      "ui_default": 10,
      "unit": "h"
    }
  },
  "statistics": {
    "per_cycle_fit": "OLS via numpy.polyfit deg 1 full=True on ln(spine_length) vs time minus first valid time; rank 2 and finite coefficients required",
    "ordering": "defensive numeric-time/length copy, stable ascending time; input unchanged",
    "segments": "split immediately after division-mother rows; nonfinite-time rows omitted with RuntimeWarning; invalid lengths retained",
    "fit_observations": "at least 4 finite times with finite positive lengths; invalid observations warn with RuntimeWarning",
    "degenerate": "fewer than 4 valid observations -> NaNs; no time variation -> RuntimeWarning and NaNs; constant length -> slope 0 and NaN R^2",
    "r_squared": "1 - sum(residual^2) / sum((log_length - mean(log_length))^2), explicit residual calculation",
    "numerical_failures": "expected numerical failures, deficient rank or nonfinite coefficients -> RuntimeWarning and NaNs; no broad exception swallow"
  },
  "plots": {
    "TRAJ_VIEWER": "spine_length (log y) vs time_h, per-cycle segments"
  },
  "outputs": [
    "streamlit session_state.trajectory_dict",
    "METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md"
  ],
  "validation_gates": [
    "no trajectory row with spine_length < min_spine_length when filter active",
    "filter executes before get_cell_lineage (code order)",
    "fully filtered channel returns None without error",
    "sidebar parameter reaches process_channel_worker on Process Data",
    "founder with two non-self children counted once; root plus one continuation not a division",
    "exact conservation accepted; negative difference and equality to either strict upper bound rejected",
    "fresh UI seconds default; all supported time units convert correctly; invalid time_unit raises ValueError",
    "sorted numeric-time copies preserve input and recover known exponential slopes",
    "nonfinite times omitted from segments; invalid lengths retained but excluded from fits with RuntimeWarning",
    "insufficient observations, repeated times, constant lengths and numerical fit failures follow declared NaN/warning policy",
    "app statistics schema 2 invalidates stale statistics; sorted trajectory returned"
  ],
  "round_id": "R002",
  "round_date": "2026-10-26",
  "logic_version": 2.3,
  "statistics_schema_version": 2,
  "rules": {
    "division": "0 <= (L_a + L_b) - L_0 < division_threshold * L_0 and L_a < 0.9 * L_0",
    "tracing": "count more than one non-self child before the root self-edge stop",
    "trajectory_order": "get_one_trajectory returns a stable ascending-time_h trajectory",
    "upstream_units": "length and width already um; area already um^2; time_s seconds; no length rescaling",
    "reconstruction": "Process Data required for new lineage rules; old annotations are not migrated"
  }
}
```
<!-- METHOD_MANIFEST_END -->

## Layer 3 — Analysis Rounds and Reconciliation

### Analysis-round registry

| Round | Date | Type | Question or change | Intended method version | Status | Primary artifacts |
|---|---|---|---|---|---|---|
| `R001` | 2026-10-06 | Intentional method change + retrospective documentation | Add user-controlled pre-division minimum spine-length exclusion (sidebar float slider with adjustable bounds, default 1.0 µm) to the Streamlit lineage pipeline; document the previously undocumented construction method | 1.0.0 | Closed | `moma_logic.py` (LOGIC_VERSION 2.2), `Moma_lineage_streamlit.py`, `tests/smoke_min_spine_length_filter.py`, `tests/actual_manifest_lineage.json`, `tests/lineage_alignment_report.csv`, this contract |
| `R002` | 2026-10-26 | Correctness fixes + units documentation | Founder-division counting, exact-halving acceptance, seconds default, validated units, sorted finite fits, explicit warning/degeneracy policy; app statistics schema 2 | 1.1.0 | Closed — regression suite, smoke and manifest alignment passed | `moma_logic.py` (LOGIC_VERSION 2.3), `Moma_lineage_streamlit.py`, `tests/test_moma_logic.py`, `tests/test_lineage_app.py`, current actual manifest and this contract |

### Historical R001 plan (retained)

Request: filter abnormal cells (spine length below a user-set threshold) before division-point finding; expose the threshold in the app's left parameter panel as a float slider whose min/max bounds are set via two input widgets; slider default 1.0 µm. Planned edits: (1) `moma_logic.get_one_trajectory` gains `min_spine_length=0.0` and filters rows right after the defensive copy, before `get_cell_lineage`; (2) `moma_logic.process_channel_worker` gains a trailing `min_spine_length=0.0` keyword and forwards it; `LOGIC_VERSION` bumped to 2.2; (3) `Moma_lineage_streamlit.py` sidebar gains a "Cell Filter" section (two bound inputs + slider) and passes the value into the worker call. Validation: smoke test of `process_channel_worker` on a real channel CSV with threshold 0 vs 1.5 µm (acceptance criteria 1–3), code-order inspection (criterion 2), and wiring inspection for criterion 4; then `validate_methods_file.py`.

### Historical R001 declared-versus-actual reconciliation (retained)

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

### Current R002 plan and reconciliation

R002 (2026-10-26) implements the approved correctness and units fixes with exact-halving acceptance first: non-self child counting before the founder stop; inclusive lower division bound; `s` fresh default and strict unit validation; stable numeric-time cycle fits with finite-observation validation, explicit numerical warnings, and sorted trajectory returns. `LOGIC_VERSION` is 2.3 and `TRAJECTORY_STATISTICS_SCHEMA_VERSION` is 2. Units corrections to the README and cleaning-model `SPINE_LENGTH` registry are documentation only; cleaning method 0.7.0 and its R001–R013 history remain unchanged.

Manifest regeneration and reconciliation commands (conda interpreter; exporter and equality assertion completed with exit 0 on 2026-10-26; these do not execute regression tests):

```powershell
$py = 'G:\miniconda3\envs\Moma_cellpose\python.exe'
& $py tests\export_actual_manifest_lineage.py
$lineage_check = @'
import json, pathlib, re
root = pathlib.Path(".")
text = (root / "docs/METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md").read_text(encoding="utf-8")
declared = json.loads(re.search(r"<!-- METHOD_MANIFEST_START -->\s*```json\s*([\s\S]*?)\s*```", text).group(1))
actual = json.loads((root / "tests/actual_manifest_lineage.json").read_text(encoding="utf-8"))
actual.pop("evidence")
assert actual == declared
print("MATCH: all non-evidence manifest fields")
'@
& $py -c $lineage_check
```

| Method step | Intended behavior | Evidence / regression reference | Current status | Impact |
|---|---|---|---|---|
| Upstream units | Length/width µm; area µm²; `time_s` seconds | Calibration and unchanged-attribute export source locators in Study design and provenance | Source inspected 2026-10-26 | Documentation correction only; no length rescaling or filter change |
| Defaults and versions | `s`, logic 2.3, statistics schema 2 | `tests/export_actual_manifest_lineage.py`; regenerated `tests/actual_manifest_lineage.json` | Regenerated 2026-10-26; all 18 non-evidence top-level fields match declared JSON by assertion; exporter does not run tests | Correct hours and stale-statistics invalidation |
| Founder and division boundaries | Ignore self-edge; count founder division; accept exact conservation and retain strict other bounds | Regression command: `pytest tests/test_moma_logic.py -q` | Passed in full suite on 2026-10-26 | Division counts, selected paths and cycle boundaries can change |
| Sorted, valid fits | Stable numeric-time copy; finite positive fit observations; warnings and explicit degeneracy policy | Regression command: `pytest tests/test_moma_logic.py -q` | Passed in full suite on 2026-10-26 | Cycle statistics may differ from historical fits |
| App default/cache and retained filter | Fresh `s`; cache schema 2; minimum-length behavior retained | Regression commands: `pytest tests/test_lineage_app.py -q`; `python tests/smoke_min_spine_length_filter.py` | AppTest and filter smoke passed on 2026-10-26 | Reconstruct links; old annotations are not migrated |

### R002 validation results

Validated 2026-10-26 using the editable-install conda interpreter, with CUDA hidden for the full suite. The final command `python -m pytest tests -q --junitxml=runs/lineage_correctness_20261026/pytest_full.xml` passed: **177 tests and 57 subtests**, **3 skips**, no failures/errors, in **58.96 s**. The 16 warnings are existing third-party Pydantic deprecations. The skips are the manual training experiment and two checks whose legacy run artifacts are absent. These passes include the contract-integrity suite, exact half-length root and non-root regressions, strict division bounds, shuffled physical-unit equivalence, numeric time strings, independent noisy-fit R², invalid-length boundary preservation, invalid times, degenerate fits, and the end-to-end Process Data AppTest.

The retained filter smoke reports **ALL GATES PASSED**. For the bundled two-channel fixture at threshold 1.2, seconds, and longest-lineage selection, channel 3 gains its omitted founder division (**5 → 6**), channel 13 remains **27**, and both selected paths and durations remain unchanged. This is a before/after characterization, not a measurement of biological accuracy. The declared and actual method manifests match after excluding `evidence`; the production checkpoint SHA-256 remains `6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`.

Evidence: `runs/lineage_correctness_20261026/validation.json`, `pytest_full.xml`, and `fixture_comparison.json` in the same run directory. Source-comparison snapshot limitations are recorded in `snapshot_metadata.json`.

### Method change log

| Method version | Date | Change | Reason | Comparability impact | Affected code and outputs |
|---|---|---|---|---|---|
| 1.0.0 | 2026-10-06 | First documented contract; introduces `min_spine_length` pre-division exclusion (UI default 1.0 µm) | User request: remove abnormal short cells before division-point finding | Trajectories built with the filter active are not strictly comparable to legacy unfiltered runs (per-frame counts and division calls can differ) | `moma_logic.py`, `Moma_lineage_streamlit.py` |
| 1.1.0 | 2026-10-26 | Count non-self children before founder stop; accept exact daughter-length conservation; default fresh UI to seconds and reject unknown units; return sorted trajectories; validate and center OLS fits with explicit warnings/R²; statistics schema 2 | Approved correctness/units fixes; deterministic regressions registered | Founder/exact-halving division calls, path selection, cycle membership and fits can differ; wrong-unit historical runs require reconstruction; old annotations are not migrated | `moma_logic.py`, `Moma_lineage_streamlit.py`, regression tests, current manifest; README/cleaning units corrected without numerical rescaling or cleaning-method bump |

### Unresolved items and next checks

- R002 validation is complete; see the evidence below. Broader biological tracking accuracy still requires annotated full-channel ground truth.
- Manual UI check pending: launch `streamlit run src/moma_lineage_analysis/app/Moma_lineage_streamlit.py`, confirm the "Cell Filter (pre-division)" section renders (slider + Slider Min/Max inputs), and that changing bounds clamps the slider value correctly.
- The bundled regression CSV fingerprint is recorded in `runs/lineage_correctness_20261026/fixture_comparison.json`; fingerprints for user analysis inputs are not yet recorded automatically.
- Sensitivity of division counts to `min_spine_length` (e.g. 0 vs 0.6 vs 1.0 µm) not yet quantified.
