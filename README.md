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
