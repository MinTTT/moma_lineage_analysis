# moma-lineage-analysis

Standalone toolkit for **mother-machine (microfluidics) cell lineage analysis**:
lineage construction from per-frame cell measurements, interactive curation in a
Streamlit app, and AI-assisted trajectory cleaning (division / noise detection).

Transferred from the `Moma_cellPose` workspace; governed methods contracts live
in `docs/`.

## Layout

```
src/moma_lineage_analysis/
├── app/Moma_lineage_streamlit.py   # Streamlit curation app (entry point)
├── app/cli.py                      # `moma-lineage` console launcher
├── moma_logic.py                   # lineage linking, division detection, per-cycle stats
├── paths.py                        # package-anchored model directory
├── trajectory_cleaning/            # PyTorch sequence models + Streamlit AI API
│   └── cli.py                      # `moma-trajectory-cleaning` train/predict CLI
└── models/trajectory_cleaning_run/ # released checkpoints (R001/R002/R003 + promoted R010 CNN)
datasets/                           # frozen verified training corpora (see datasets/README.md)
tests/                              # unit + smoke + AppTest UI suites (incl. data fixture)
docs/                               # METHODS_* contracts
```

## Install (editable)

```powershell
pip install -e G:\python_code\moma_lineage_analysis
# with test runner:
pip install -e G:\python_code\moma_lineage_analysis[dev]
```

## Run the app

```powershell
moma-lineage                                   # console script
# or
python -m moma_lineage_analysis.app
# or
streamlit run src\moma_lineage_analysis\app\Moma_lineage_streamlit.py
```

Then set **Data Directory Path** in the sidebar to a folder of `*_cells_df.csv`
files (exported by the upstream `Moma_CPM_FOV.py` pipeline) and click
**Process Data**.

To drop an entire field of view, use **Drop trajectories by FOV** in the left
sidebar. Enter its **FOV index (0-based)** (for example, `0` matches
`FOV_0_CH_16` and every other channel in FOV 0), then click
**Drop all trajectories in FOV**. The sidebar shows how many loaded trajectories
match that FOV. Dropped trajectories are excluded from selected
exports and statistics and remain available for inspection; use **Keep
Trajectory** to keep one again.

## Review images for selected observations

1. Keep the upstream `<fov>_exported_data.pkl` beside its matching
   `<fov>_cells_df.csv` in **Data Directory Path**, then click **Process Data**.
   Image review uses these local pipeline exports; only use pickles from a
   source you trust.
2. Lasso or box-select observations in the trajectory plot, including excluded
   red X markers when needed, and click **🖼️ Review Images** in **Data Cleaning**.
3. A large popup shows full saved side-channel images in a compact horizontal
   strip with small fixed gaps and no per-image captions. The page size adapts
   to the measured block width, image aspect ratios, and selected image height,
   so more frames fit when space allows. Click an image to show its FOV, image
   name, time point, side/optical channel, and cell labels above the strip.
   **Earlier frames**, **Later frames**, and the page selector navigate longer
   selections.
4. Choose an **Optical channel**, toggle **Mask fill** and **Mask outlines**,
   and choose **Selected cells** or **All cells**. The selected trajectory cells
   are highlighted. Enable **DT spine and ribs** to draw the saved DT mesh:
   yellow ribs and a magenta spine through their midpoints. The image height
   control changes the display scale while preserving aspect ratio and the
   orientation saved by the pipeline.

The popup captures the selection when opened. Layer and page changes affect
only image review; close it to return to curation. The clicked-image information
line also includes exclusion/division annotations. Missing fluorescence frames,
masks, or saved DT meshes are reported without substituting other cells or
reconstructing legacy geometry. Images use a shared display contrast across
each page; the original image data are preserved.

The app pairs the pickle using the original CSV's exact filename and joins rows
by `channels`, `time_point`, and `cell_index`. Mask boundaries come from the
segmentation labels, supporting both legacy and DT exports. Image review reads
the pickle on demand and caches one FOV per browser session. Sessions loaded
before this feature need **Process Data** again to capture source-file
information. The image viewer requires Streamlit 1.55 or later.

## Labeled CSV export columns

The app writes curated trajectories with curation labels from two places, both
using the same schema (one row per cell observation, rows sorted by `time_h`):

- **Export with Labels** (sidebar): all *selected* (kept) trajectories
  concatenated into `<data dir>/selected_labeled.csv`.
- **Export Current Trajectory** (analysis panel): only the trajectory on screen,
  saved as `<data dir>/<trajectory_key>_selected_labeled.csv`.

File names contain `selected` on purpose: the **Process Data** loader skips any
CSV whose name contains `selected`, so exports are never re-ingested as raw
input.

### Measurement columns (carried through from the input `*_cells_df.csv`)

| Column | Meaning |
|---|---|
| `time_s` | Acquisition time of the frame, in seconds (from the upstream pipeline). |
| `time_h` | Time in hours since the first frame of this FOV/channel: `(time_s - min time_s) / 3600`. |
| `time_point` | Frame index (0-based). |
| `channels` | Channel name the cell was segmented in. |
| `cell_index` | Per-frame segmentation id (unique only within one frame). |
| `fov_cell_index` | Cell identity within the FOV, linked across frames; the label columns below reference this id. |
| `ancestor_index` | Parent `fov_cell_index` from tracking; used to trace the lineage back to its founder cell. |
| `spine_length` | Cell length along the growth channel (pixels); the main growth/division signal. |
| `area` | Cell area (pixels²). |
| `umppx` | Micrometers per pixel (scale calibration). |
| `width`, `width_area` | Cell width estimates. |
| `x_min`, `x_max`, `y_min`, `y_max` | Bounding box of the cell mask (pixels). |
| *(one per fluorescence channel)* | Per-cell fluorescence intensity as exported by the upstream pipeline; any column not listed above is treated as a fluorescence channel. |

### Label columns (added by the app)

| Column | Meaning |
|---|---|
| `trajectory_key` | Trajectory identifier (e.g. `FOV_0_CH_16`); constant within one trajectory and matches the key shown in the app. |
| `division_point` | `True` on the **mother frame immediately before a division** (the last frame before the daughter's length drop), exactly as marked by the curator or the AI. Raw label — never rewritten by conflict resolution. |
| `excluded_point` | `True` for noise/artifact frames the curator excluded from analysis. Excluded rows are **kept in the file and flagged, never deleted**. |
| `effective_division_point` | Division boundary actually used for per-cycle statistics, after division/exclusion conflict resolution: if the raw `division_point` frame is excluded, the boundary moves to the previous valid measurement (or is dropped/coalesced when unresolvable). May differ from `division_point`; the raw label stays untouched. |
| `cell_cycle_index` | 0-based cell-cycle number within the trajectory; increments on the row immediately after each `effective_division_point`. |

To rebuild statistics downstream, filter `~excluded_point`, split by
`cell_cycle_index` (boundaries at `effective_division_point`), and group by
`trajectory_key`. These files are also the input format for the training-data
pipeline (see `scripts/qc_raw_train_data.py` and `AGENTS.md` workflow A).

## Tests

```powershell
python -m pytest tests -q
python tests\smoke_min_spine_length_filter.py
```

## Train a trajectory-cleaning model

The current retraining corpus is the merged 4-source corpus, stored as SQLite
dataset lineage `verified_merged_20261017@v1` (373,440 rows / 1,149
trajectories; provenance:
`datasets/verified_merged_20261017/merge_manifest.json`). The frozen R004 corpus
(`verified_20260916_no_longruns@v1`) is the historical production trainset; all
corpora are stored once in `datasets/store/lineage.sqlite3` and catalogued in
`datasets/manifest.json`. Training reads the store directly and records the
dataset lineage IDs it used. The legacy corpus CSVs were retired in phase 3
(2026-10-17) and are restorable byte-identically from the store:
`python scripts\dataset_store.py export --dataset <id> --version 1 --output <path>`
(see `datasets/store/retired_csv_manifest_20261017.json`), after which the CSV
path also works:

```powershell
# Store-direct training (lineage ids recorded in metrics.json)
moma-trajectory-cleaning train `
  --dataset-lineage verified_merged_20261017@v1 `
  --store datasets\store\lineage.sqlite3 `
  --require-human-verified --source-column source_dataset `
  --output-dir runs\my_run

# Legacy CSV path (same frames; --dataset-manifest required with the gate)
moma-trajectory-cleaning train `
  --input datasets\verified_merged_20261017\verified_training_data.csv `
  --dataset-manifest datasets\verified_merged_20261017\merge_manifest.json `
  --require-human-verified --source-column source_dataset `
  --output-dir runs\my_run

moma-trajectory-cleaning predict --input my_trajectories.csv `
  --checkpoint runs\my_run\best_model.pt --output predictions.csv
```

The deployed production checkpoint is the **R010 CNN**
(`src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt`,
sha256 `6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`,
promoted 2026-10-17); the previous R008 CNN is backed up
under `promotion_backups/20261017T124420.980675Z`.

## Model checkpoints

The app's AI panel resolves checkpoints from the package-anchored
`models/trajectory_cleaning_run/` directory (see
`moma_lineage_analysis.paths.PACKAGE_MODELS_RUN_DIR`); no working-directory
dependence. Override the default checkpoint with the `MOMA_AI_MODEL_PATH`
environment variable.
