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

Fresh sessions default **Time Unit** to `s`, matching upstream `time_s` exports.
Choose `min`, `h`, or `d` explicitly only when the input values use that unit;
unsupported units raise `ValueError`. Lineage method 1.1.0 (R002, logic 2.3)
corrects founder-division counting, accepts exact daughter-length conservation,
and uses sorted, finite observations for per-cycle fits. Reconstruct with
**Process Data** to use these rules; existing annotations are not migrated.
Recheck division labels after reconstruction because selected paths and cycle
boundaries can change. Historical runs using `d` for seconds-valued inputs have
incorrect durations and rates and require reconstruction with `s`.

To drop an entire field of view, use **Drop trajectories by FOV** in the left
sidebar. Enter its **FOV index (0-based)** (for example, `0` matches
`FOV_0_CH_16` and every other channel in FOV 0), then click
**Drop all trajectories in FOV**. The sidebar shows how many loaded trajectories
match that FOV. Dropped trajectories are excluded from selected
exports and statistics and remain available for inspection; use **Keep
Trajectory** to keep one again.

## Autosave and resume curation

**Process Data** creates `moma_curation.sqlite3` in the input directory after
reconstruction. It stores all reconstructed measurements, initial division
labels, current labels, keep/drop membership, review status, and the current
cursor. Image pickle arrays and AI preview caches stay outside this database.
This local working database is separate from the frozen training-data store.

- Each manual label edit, reset, keep/drop action, FOV drop, AI application,
  and AI undo commits immediately in a SQLite transaction.
- **Next** confirms the current trajectory as reviewed, saves it, and opens
  the next one. The right-arrow shortcut uses the same action. **Prev** and
  trajectory-number jumps navigate without confirming review.
- AI auto pilot skips reviewed destinations, including after wrapping back to
  the first trajectory. AI Merge/Replace, input preflight, and AI undo preserve
  reviewed trajectories; only unreviewed trajectories in the selected scope
  are eligible for AI changes.
- Editing a reviewed trajectory changes its status back to **in progress**.
- After a crash or restart, enter the same directory, choose a saved dataset
  version, and click **Resume saved review**. Measurements, labels, keep/drop
  choices, processing settings, and the last trajectory are restored.
- Processing identical CSVs with identical settings resumes the saved version.
  Changed source contents, settings, or app logic create a new version while
  preserving older versions. Incompatible sources/logic block resume to avoid
  incorrect image joins and reconstruction assumptions.
- **Saved** displays the committed revision. An **Autosave failed** error means
  edits remain in memory; navigation and auto pilot pause until saving succeeds.
  Revision checks reject silent overwrites from a second browser session. On
  conflict, **Save recovery copy & reload database** first writes unsaved local
  edits into a separate `moma_curation_recovery_*.sqlite3` file, then reloads
  the latest revision. Both versions are preserved.
- For a session loaded before autosave was introduced, click **Save current
  session to SQLite**. This creates a new version, preserving its unsaved
  labels without overwriting an older review. Later loads use **Process Data**.

Three active-dataset SQL views have the same measurement and label columns as
labeled CSV exports. Boolean labels use 0/1. `selected_labeled` contains kept
 trajectories; `reviewed_selected_labeled` contains reviewed, kept trajectories;
`all_labeled` also includes dropped trajectories. Source/review bookkeeping
stays in separate tables and never becomes measurement columns.

```python
import sqlite3
from contextlib import closing
import pandas as pd

with closing(sqlite3.connect("moma_curation.sqlite3")) as connection:
    df = pd.read_sql_query(
        "SELECT * FROM reviewed_selected_labeled ORDER BY trajectory_key, time_h",
        connection,
    )
```

Views follow the dataset version most recently opened/saved by the app. Local
disks use WAL journaling. Windows mapped network drives and UNC shares use
rollback (`DELETE`) journaling, which avoids WAL shared-memory requirements
across computers. Both modes retain full synchronization, foreign-key checks,
short transactions, and revision conflict checks. Network storage must support
SQLite file locking and stay reachable and writable during a save. A network
or permission failure leaves edits in memory and pauses navigation; reconnect
and click **Retry autosave** without reprocessing or discarding the session.

While the app runs, `-wal`, `-shm`, or `-journal` companion files may hold live
or recovery state; do not delete or omit them from a raw copy. Close the app
before copying the database, or use SQLite's backup API for a live backup.

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
5. Click an image, then use **Export selected image + mask (TIFF)** to save the
   original source image and its original 2D label mask into the source data
   folder. The image contains no mask, outline, spine, or rib overlays; those
   remain display-only. Pixel values and dtypes are preserved. The pair shares
   a random eight-character filename tag; the mask filename ends in
   `_masks.tif`. Repeated exports create new tags and never overwrite existing
   files.

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
| `fov_cell_index` | Unique observation ID within the loaded FOV table (assigned from row position), not a persistent cell identity across frames; the label columns below reference this ID. |
| `ancestor_index` | Parent `fov_cell_index` from tracking; used to trace the lineage back to its founder cell. |
| `spine_length` | Cell spine length (µm), already calibrated upstream; the main growth/division signal. |
| `area` | Cell area (µm²), already calibrated upstream. |
| `umppx` | Micrometers per pixel (scale calibration). |
| `width`, `width_area` | Cell width estimates (µm), already calibrated upstream. |
| `x_min`, `x_max`, `y_min`, `y_max` | Bounding box of the cell mask (pixels). |
| *(one per fluorescence channel)* | Per-cell fluorescence intensity as exported by the upstream pipeline; any column not listed above is treated as a fluorescence channel. |

Length, width, and area calibration happens upstream using `umppx`; do not
multiply these exported measurements by it again. Source-code evidence for both
legacy and DT exports is recorded in
[the lineage methods contract](docs/METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md).

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
