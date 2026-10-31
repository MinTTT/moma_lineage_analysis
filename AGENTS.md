# AGENTS.md - working agreement for LLM agents

This file tells an AI agent (and any new human collaborator) what this project
is, how the code is organised, what the deep-learning models are, and the
exact workflows for auditing new training data, training models, and promoting
(updating) the deployed model. Follow it; it is the operating contract for
agents working in this repository.

Project root: `G:\python_code\moma_lineage_analysis` (Windows). The workspace
is NOT a git repository; there is no `git log` to consult. Provenance lives in
the documents and hash manifests listed in section 7.

---

## 1. Project aim

Mother-machine (microfluidic) cell-lineage analysis toolkit:

1. Build single-cell lineage trajectories from per-frame
   `*_cells_df.csv` measurements exported by the upstream `Moma_CPM_FOV.py`
   Cellpose pipeline (link cells across frames, detect divisions, compute
   per-cycle statistics).
2. Curate trajectories interactively in a Streamlit app (label division points,
   mark noise/exclusion points, merge cycles, review AI suggestions).
3. Train and deploy PyTorch sequence models that predict, from only `time_h`
   and `spine_length`, two independent multi-label targets per observation:
   `division_point` (mother frame immediately before a division) and
   `excluded_point` (noise / artifact, exclude from analysis).

The scientific goal is to reproduce human trajectory-cleaning decisions for
offline analysis - not unattended deletion. All models are human-reviewed
previews only.

---

## 2. Repository map (app structure)

```text
moma_lineage_analysis/
├── src/moma_lineage_analysis/
│   ├── app/
│   │   ├── Moma_lineage_streamlit.py  # Streamlit curation app (UI, AI panel, exports)
│   │   ├── cli.py                     # `moma-lineage` console launcher
│   │   └── __main__.py                # `python -m moma_lineage_analysis.app`
│   ├── moma_logic.py                  # lineage tracing, division detection, per-cycle stats (LOGIC_VERSION 2.3)
│   ├── sciplot.py                     # plotting helpers
│   ├── paths.py                       # PACKAGE_MODELS_RUN_DIR (package-anchored checkpoints)
│   ├── trajectory_cleaning/
│   │   ├── cli.py                     # `moma-trajectory-cleaning train|predict`
│   │   ├── data.py                    # data contract, normalizer, TrajectoryDataset, splits
│   │   ├── models.py                  # CNNLabeler, ResNetLabeler, LSTMLabeler
│   │   ├── training.py                # losses, train loop, threshold calibration, evaluate_model, load_checkpoint
│   │   ├── streamlit_api.py           # AI preview API + division/exclusion conflict resolver
│   │   ├── prepare_verified_training_data.py  # build attested corpora from verified CSVs
│   │   ├── dataset_store.py           # local SQLite dataset store (schema/import/export/verify)
│   │   ├── promote_best_model.py      # validation-only selection + promoted artifact copy
│   │   ├── plot_training_results.py   # loss/F1/label-balance figures for a run dir
│   │   └── MODEL_REPORT.md            # model report (architectures, round results, comparisons)
│   ├── trainset_review/               # `moma-trainset-review` review/edit/training-ready app
│   │   ├── streamlit_app.py           # focused training-data review UI
│   │   ├── store.py                   # guarded SQLite review backend
│   │   └── cli.py                     # `moma-trainset-review` console launcher
│   └── models/trajectory_cleaning_run/  # released + promoted checkpoints (see section 3.5)
├── scripts/
│   ├── qc_raw_train_data.py           # technical QC of raw *_selected_labeled.csv exports
│   ├── stage_verified_labels.py       # canonical label staging (R002 policy / effective mode)
│   ├── merge_corpora.py               # merge verified corpora + R004 filter on incoming batches
│   ├── dataset_store.py               # build/verify/export/backup the SQLite dataset store
│   └── remove_dataset_versions.py     # admin removal of duplicate dataset versions (R019)
├── datasets/                          # frozen verified corpora + integrity manifest + store/ (section 11)
├── docs/
│   ├── METHODS_TRAJECTORY_CLEANING_MODEL.md  # scientific contract (rounds R001-..., method versions)
│   ├── METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md  # lineage-construction contract
│   ├── TRAINSET_REVIEW.md             # training-dataset review-app user guide
│   └── PLAN_DATASET_STORE.md          # dataset-store design + phase status
├── runs/                              # training runs, comparisons, figures, verification reports
├── tests/                             # unittest/pytest + Streamlit AppTest suites + fixtures
├── pyproject.toml                     # entry points: moma-lineage, moma-trajectory-cleaning, moma-trainset-review
├── README.md                          # user-facing quick start
└── AGENTS.md                          # this file
```

### Environment (important gotcha)

The Windows `python` on PATH does NOT have pandas/torch. Always use the conda
env with the editable install:

```powershell
$py = 'G:\miniconda3\envs\Moma_cellpose\python.exe'
cd G:\python_code\moma_lineage_analysis
& $py -c "import moma_lineage_analysis, torch; print(torch.cuda.is_available())"
```

Console scripts `moma-lineage`, `moma-trajectory-cleaning` and `moma-trainset-review`
exist in that env.
GPU: NVIDIA RTX 5090 (CUDA available; training uses `--device auto`).

---

## 3. Deep-learning models

### 3.1 Task and inputs

- One row = one cell observation; one trajectory = ordered rows of one
  `trajectory_key`; targets `division_point`, `excluded_point` (multi-label,
  independent). Division labels sit on the mother frame before the daughter
  length drop.
- Features: `time_h`, `spine_length` only. The normalizer (`TrajectoryNormalizer`)
  builds a 5-channel extended layout since round R002:
  `time, length, length_delta, log_growth_rate, time_since_last_drop`,
  clipped to [-12, 12], computed per trajectory (no fitted global statistics).
  Pre-R002 checkpoints declare `feature_compatibility="legacy"` (2 channels).
- Padding via `collate_trajectories` + validity masks; LSTM uses packed sequences.

### 3.2 Architectures (`models.py`)

| Model | Params (5-ch) | Default context | Notes |
|---|---:|---|---|
| `CNNLabeler` (`cnn`) | 63,938 | 17 frames | 4 conv blocks, centered conv, LayerNorm, GELU, dropout; fastest, smallest |
| `ResNetLabeler` (`resnet`) | 165,890 | 121 frames | 4 residual blocks, dilations 1/2/4/8 |
| `LSTMLabeler` (`lstm`) | 334,594 | full trajectory | 4-layer bidirectional LSTM, packed sequences; slowest |

R016 experiments may set `conv_dilations` explicitly for CNN/ResNet. Omitting it
preserves old defaults/checkpoints; the primary ResNet sweep uses kernel 3,
while the old kernel-5 RF121 model is a separate control.

### 3.3 Training defaults (`TrainConfig`)

AdamW, lr 1e-3, weight decay 1e-4, masked focal BCE (gamma 1.5, positive-class
weight `neg/pos` capped at 50), gradient clip 5.0, epochs 100, patience 15,
batch size 16 trajectories, seed 42, device auto. Early stopping restores the
minimum-validation-loss state; per-label thresholds are calibrated on validation
point F1 only (grid 0.05-0.95). `division_tolerance_frames = 1`.

### 3.4 Splitting and evaluation rules

- `split_trajectory_keys` keeps complete trajectories together; optional
  per-source (`source_dataset`) stratification. Default 70/15/15, seed 42.
  This is NOT FOV-held-out cross-validation - a documented limitation.
- Architecture selection uses ONLY validation metrics: arithmetic mean of
  validation division-event F1 and validation exclusion F1 (tie-breakers in
  `promote_best_model.ValidationScore.ranking_key`). Test metrics are never used
  for selection. Test metrics may be reported for acceptance, never to choose.
- Event metrics collapse contiguous division runs into one biological event and
  match within +/-1 frame.

### 3.5 Released checkpoints (`src/moma_lineage_analysis/models/trajectory_cleaning_run/`)

- `best_model.pt` - PRODUCTION checkpoint: **R019 CNN**, sha256
  `816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99`,
  promoted 2026-10-31 (R020, user decision; validation-rule winner, mean F1
  0.901205); trained on the five `reviewed_20261031_*` lineages (1,152
  trajectories / 385,815 rows). Its companions at the root (`history.csv`,
  `metrics.json`, `splits.json`) are the R019 CNN artifacts; `promotion.json`
  records the promotion scores/hashes/backup.
- `promotion_backups/20261030T184217.243155Z/` - previous R010 CNN artifacts
  (best_model.pt sha256 `6e7c5a58...`) plus the archived R010
  `promotion.json`, rollback source. Older backups:
  `20261017T124420.980675Z/` (R008 CNN) and `20261016T145403.716334Z/`
  (R004-era artifacts).
- `verified_20260916/{cnn,resnet,lstm}/` - R002 mixed-corpus checkpoints.
- `rdm_only_20260916/{cnn,resnet,lstm}/` - R003 RDM-Gluc-only specialists.
- The shared registry (`model_registry.py`) exposes the production R019 CNN plus
  selectable R015 and R002/R003 checkpoints. R015 is packaged under
  `merged_20261029/cnn/` and is non-production; R019 remains default.
  `MOMA_AI_MODEL_PATH` overrides the default checkpoint in the lineage app.

### 3.6 Dataset rounds (short history; full chain in the methods contract)

| Round | Corpus | Rows / trajectories | Outcome |
|---|---|---:|---|
| R004 (2026-09-17) | `datasets/verified_20260916/verified_training_data_no_longruns.csv` | 312,317 / 1,053 | production corpus, R004 filter (drop trajectories whose longest contiguous exclusion run >= 5 frames) |
| R006 / R007 / R009 | `verified_20261013` / `verified_20261016` / `verified_20261017` batches | 22,890/35, 31,384/48, 8,757/16 | additive curated batches (not trained standalone) |
| R008 (2026-10-16) | `datasets/verified_merged_20261016` | 365,283 / 1,134 | retrained cnn/resnet/lstm; user overrode rule and promoted the CNN |
| R010 (2026-10-17) | `datasets/verified_merged_20261017` | 373,440 / 1,149 | retrained cnn/resnet/lstm; validation rule winner = CNN; paired evaluation vs R008 CNN: significantly better exclusion AP (+0.025 / +0.034, replicates on leakage-free subsets), division unchanged |
| R011 (2026-10-17) | - | - | promotion of the R010 CNN (deployment only) |
| R012 (2026-10-17) | `datasets/store/lineage.sqlite3` (six frozen versions) | 375,348 unique observations / 1,152 trajectories | data-layer round: store built + verified, training reads it directly and records lineage ids; R010 reproduced (byte-identical splits, bit-identical resnet/lstm, cnn AP drift <=9e-5 = GPU nondeterminism); production unchanged |
| R013 (2026-10-17) | - | - | retirement round: six legacy corpus CSVs deleted after sign-off; store is the sole training source and exports them byte-identically on demand; `datasets/` 348 MB -> 29 MB; production unchanged |
| R014 (2026-10-29) | `verified_20261029@v1` | 25,728 / 48 | effective-label batch: 2,562 divisions, 98 exclusions, overlap 0; eight reused identifiers date-prefixed (biological identity unestablished); max exclusion run 3, no R004 drops |
| R015 (2026-10-29) | `verified_merged_20261029@v1` | 399,168 / 1,197 | retrained cnn/resnet/lstm, split 838/179/180; validation rule selects CNN (mean F1 0.8886); mixed descriptive paired panels do not support replacement; production unchanged |
| R016 (2026-10-29) | unchanged R015 lineage/split | 399,168 / 1,197 | temporal-context experiment, hidden64 fixed, 29 runs; validation-only lock selected RF17 for CNN and primary ResNet k3; no locked model beats the deployed R010 CNN on descriptive panels; no promotion |
| R018 (2026-10-29) | existing SQLite store; no new corpus row | 399,168 / 1,197 current R015 data; training-ready versions created on demand | independent training-review app; direct SQLite autosave, shared keyboard shortcuts, user-authorized label edits/permanent trajectory deletion via guarded backend; R015 selectable but non-production; independent review PASS |
| R019 (2026-10-31) | five reviewed batch versions composed (no merged corpus) | 385,815 / 1,152 | user-requested removal of the three duplicate `verified_merged_*` dataset versions; five `reviewed_20261031_*` training-ready versions finalized with the R004 membership filter; cnn/resnet/lstm retrained with unchanged configuration |
| R020 (2026-10-31) | - | - | promotion (deployment only): user-authorized promotion of the R019 CNN to production (sha256 `816a8b39...`); previous R010 CNN + promotion manifest backed up; registry/pins/labels/docs updated |

---

## 4. Workflow A - audit new raw training data and build a verified batch corpus

Input: human-curated exports `datasets/raw_train_data/<FOV>_CH_<n>_selected_labeled.csv`
(one trajectory per file; columns include `trajectory_key, fov_cell_index,
time_h, spine_length, division_point, excluded_point, effective_division_point`).

Steps (all from the repo root, `$py` = conda python above):

1. **Inventory + QC** (hard gates: schema, one key == filename stem, boolean
   labels, no missing values, finite `time_h`, finite positive `spine_length`,
   no duplicate `(key, fov_cell_index)`, no cross-file key collisions):
   ```powershell
   # --corpus must be an existing CSV; restore the R010 corpus from the store first
   # (python scripts\dataset_store.py export --dataset verified_merged_20261017
   #  --version 1 --output datasets\verified_merged_20261017\verified_training_data.csv)
   & $py scripts\qc_raw_train_data.py datasets\raw_train_data `
       --corpus datasets\verified_merged_20261017\verified_training_data.csv `
       --report datasets\raw_train_data\qc_report.json
   ```
   Exit code 0 = hard gates pass. Warnings include label-policy divergence
   (`division_point != effective_division_point`) and overlap with the existing
   corpus - inspect both.
2. **Decide duplicate keys.** Any raw key exactly matching a trajectory already
   in an existing corpus must be renamed with a date prefix
   (`<YYYYMMDD>_<raw_key>`, e.g. `20261017_FOV_0_CH_16`) and recorded in
   `<batch>/rename_map.json`. Also check prefix-normalized overlap and distinguish
   naming collisions from demonstrated biological re-curations; reused FOV/channel
   identifiers alone do not establish identity across acquisitions. Do not silently collide.
3. **Stage canonical labels** with the effective-label policy (`division_point`
   = `effective_division_point`, superseding raw labels on conflicted rows;
   division/exclusion overlap becomes 0):
   ```powershell
   & $py scripts\stage_verified_labels.py datasets\raw_train_data `
       datasets\<batch>\staged `
       --division-mode effective `
       --rename-map datasets\<batch>\rename_map.json `
       --report datasets\<batch>\staging_report.json
   ```
   Check the report: division labels removed/added, resolver statuses
   (`unchanged / coalesced_previous / shifted_previous / needs_review`), and
   `r004_filter_would_drop` (trajectories with an exclusion run >= 5 frames).
4. **Combine** the per-file staged CSVs in sorted filename order (pandas concat)
   into `datasets/<batch>/staged/combined_curated_<date>.csv`.
5. **Build the attested corpus** (keeps provenance columns and strips all
   AI/audit label columns):
   ```powershell
   & $py -m moma_lineage_analysis.trajectory_cleaning.prepare_verified_training_data `
       --source datasets\<batch>\staged\combined_curated_<date>.csv `
       --dataset-id curated_<date> `
       --verification-basis user_attested_<YYYY-MM-DD> `
       --output-csv datasets\<batch>\verified_training_data.csv `
       --manifest datasets\<batch>\dataset_manifest.json
   ```
   Output: namespaced keys `<dataset_id>::<key>`, 11-column schema, manifest
   with `human_verified: true` and per-source hashes. Copy the QC report into
   the batch folder.
6. **Verify independently** (do not trust the generator): recompute sha256, row/
   trajectory/label counts, schema, namespace, duplicate keys (exact +
   prefix-normalized), exclusion-run check, label staging vs the packaged
   resolver, and split feasibility. Optionally train a 1-epoch CPU smoke with
   `--require-human-verified`.
7. **Merge for training** (only when asked to retrain; the historical corpus CSVs
   were retired in phase 3 — restore any corpus from the store first with
   `python scripts\dataset_store.py export --dataset <id> --version 1 --output
   <path>`): apply the R004 filter to batches while merging:
   ```powershell
   & $py scripts\merge_corpora.py --output-dir datasets\verified_merged_<date> `
       --corpus datasets\verified_20260916\verified_training_data_no_longruns.csv `
       --filtered-corpus datasets\verified_20261013\verified_training_data.csv `
       --filtered-corpus datasets\verified_20261016\verified_training_data.csv `
       --filtered-corpus datasets\<batch>\verified_training_data.csv
   ```
   The merge fails if keys collide or if the pre-filtered corpus violates the
   R004 rule.
8. **Update documentation and the integrity contract**:
   - `datasets/README.md` (contents table + batch section + caveats),
   - `datasets/manifest.json` (new entry with sha256, rows, trajectories,
     construction, provenance files, caveats) - then re-verify every pin,
   - `docs/METHODS_TRAJECTORY_CLEANING_MODEL.md` (new round row, method version,
     data-structure row, reconciliation rows, change log).
9. **Ingest into the dataset store** (phase 1+): after the attested corpus and its
   `datasets/manifest.json` entry exist, run
   `python scripts\dataset_store.py build` then
   `python scripts\dataset_store.py verify` (must PASS: counts, canonical hash,
   legacy CSV sha256 / byte-identical export, retired state), and take a fresh
   backup with `python scripts\dataset_store.py backup`. `build` is idempotent and
   only inserts genuinely new versions; after phase 3 it skips a corpus whose CSV
   was retired and is already in the store (new batches import normally). A
   missing CSV for a dataset the store does not hold is an error - restore it from
   `datasets/store/retired_csv_manifest_20261017.json` first.

   Keep raw exports in place unless the user rotates them; freeze staged
   outputs because the combined staged CSV's hash is recorded in the corpus
   manifest (per-file staged CSVs are not hashed by the manifest).

**Definition of done for A:** 0 QC failures, documented duplicate/rename
decisions, staging counters reconcile arithmetically, attested manifest written,
independent verification PASS, integrity manifest + README + methods contract
updated, `pytest tests/trajectory_cleaning/test_contract_integrity.py -q` passes.

---

## 5. Workflow B - train models

Train all three architectures sequentially (same data/split, so their
`splits.json` must be identical - the promotion tool enforces this):

```powershell
foreach ($m in @('cnn','resnet','lstm')) {
  & $py -m moma_lineage_analysis.trajectory_cleaning.cli train `
    --dataset-lineage verified_merged_20261029@v1 `
    --store datasets\store\lineage.sqlite3 `
    --require-human-verified `
    --source-column source_dataset `
    --model $m `
    --output-dir runs\merged_20261029\$m `
    --device auto
}
```

- Training reads the frozen corpora from the SQLite store (phase 2):
  `--dataset-lineage <id>` is read-only and repeatable (compose versions; shared
  observations are collapsed only when identical); the lineage ids plus canonical
  hashes and legacy CSV sha256 are written to `metrics.json` as
  `dataset_lineage`. `--input <csv> --dataset-manifest ...` remains as the legacy
  path (same frames and split keys; restore a retired corpus from the store
  first).
- `--require-human-verified` refuses to train unless the dataset manifest declares
  `human_verified: true` (CSV path) or the store dataset is attested
  `human_verified = 1`; `--source-column` enables per-source stratified splits. Do
  not change default hyperparameters when the intent is a comparable round;
  document any change in the methods contract.
- Outputs per model: `best_model.pt`, `history.csv`, `metrics.json` (validation +
  test blocks, thresholds, split_summary, dataset_manifest/dataset_lineage
  provenance), `splits.json`.
- Build a comparison JSON (validation-rule winner + test metrics; see
  `runs/merged_20261017/r010_model_comparison.json`) and figures:
  ```powershell
  # --input must be an existing CSV; restore the corpus from the store first if
  # it was retired (see section 11)
  & $py -m moma_lineage_analysis.trajectory_cleaning.plot_training_results `
    --run-dir runs\<run> `
    --input datasets\verified_merged_20261017\verified_training_data.csv `
    --output-dir runs\<run>\figures
  ```
- Cross-round comparisons must be paired on identical test rows (same-population
  cross-evaluation); per-round `metrics.json` values from different corpora are
  not comparable. Watch for cross-round split contamination (a trajectory can be
  in another round's train split) and report the leakage-free subset too.
- Update `MODEL_REPORT.md` (round section, results table, figures) and the
  methods contract (round registry, method version, populations, change log).

---

## 6. Workflow C - update (promote) a model

> **Promotion requires an explicit user decision.** Never promote on your own
> because a model won the validation rule; the user decides deployment.

1. Confirm the candidate run and the validation-rule winner (the tool re-selects
   deterministically). Preview what will be replaced.
2. Run the promotion (hash-verified copy; previous artifacts backed up):
   ```powershell
   & $py -m moma_lineage_analysis.trajectory_cleaning.promote_best_model `
     --run-dir runs\reviewed_20261031 `
     --destination-dir src\moma_lineage_analysis\models\trajectory_cleaning_run
   ```
   It writes `promotion.json` (selected model, validation scores, artifact
   hashes, backup path) and copies `best_model.pt`, `history.csv`,
   `metrics.json`, `splits.json` to the destination root.
3. Post-promotion checklist - every item is mandatory:
   - verify the new `best_model.pt` sha256 equals the run's checkpoint and
     records into `promotion.json`; confirm the backup directory contains the
     previous artifacts (and keep them - they are the rollback source);
   - update the checkpoint hash pin in
     `tests/trajectory_cleaning/test_contract_integrity.py`
     (`PROMOTED_SHA256` + comment);
   - update the Streamlit `AI_MODEL_REGISTRY` production label/description in
     `app/Moma_lineage_streamlit.py` and the label expectation in
     `tests/trajectory_cleaning/test_streamlit_ai_panel.py`;
   - load-check + predict smoke with `load_checkpoint`/the predict CLI (finite
     outputs);
   - update `docs/METHODS_TRAJECTORY_CLEANING_MODEL.md` (deployment round row,
     method version, deployed-checkpoint row, promotion reconciliation row,
     change log), `MODEL_REPORT.md` (deployment status), `datasets/README.md` +
     `datasets/manifest.json` (`trained_models`, promotion status), and the
     top-level `README.md` if it names the promoted checkpoint;
   - run `pytest tests/trajectory_cleaning -q` and the full `pytest tests -q`.
4. **Gotcha:** `promote_best_model.py` backs up the four model artifacts but NOT
   the previous `promotion.json`. Archive the old manifest before promoting if
   you need a byte-level record; the decision itself stays documented in the
   methods contract and `MODEL_REPORT.md`.
5. **Rollback:** copy the four artifacts back from the backup directory
   (`promotion_backups/<timestamp>/`) and restore the previous sha pin/docs.

---

## 6b. Workflow D - review data and create a training-ready version

Use this for the focused **training-review** app, not for lineage construction
and not for model training. It runs on the existing SQLite store and is the
only supported place for user-authorized review mutations.

~~~powershell
$py = 'G:\miniconda3\envs\Moma_cellpose\python.exe'
cd G:\python_code\moma_lineage_analysis
& $py -m moma_lineage_analysis.trainset_review
# installed launcher (Scripts dir is not on PATH):
& 'G:\miniconda3\envs\Moma_cellpose\Scripts\moma-trainset-review.exe' --server.address 127.0.0.1 --server.port 8502
~~~

- Default store: `datasets/store/lineage.sqlite3`; override with
  `MOMA_TRAINSET_STORE` or the sidebar field. Use a copy for experiments.
- Opening/listing/plotting is read-only. Review tables are created only on the
  first mutating operation.
- The app has two tabs, **Review trajectories** and **Evaluate dataset**; there
  is no dataset CSV/export tab or `export_snapshot` action. Edits write directly
  to SQLite and autosave immediately with no Save button, including per-row
  label checkboxes in **Edit labels as a table**, reviewed marks and permanent
  deletion. Label edits apply to shared observations across every referencing
  version. **Measurements (`time_h`, `spine_length`) are immutable.** A
  permanent trajectory delete removes its observations and all version
  memberships; there is no in-app restore. The user-requested red **Remove
  trajectory from database** button directly below the plot deletes the current
  trajectory in one click without a confirmation field/dialog, saves immediately
  and advances to the next available queue item. The backend still enforces the
  captured trajectory key/revision and takes a pre-delete backup.
- The review plot uses `trajectory_view_style.py`: cycle-coloured segmented
  spine runs, log spine length, shared markers, lasso and unified x hover, plus
  a probability panel for previews. Both apps use its shared shortcut handler:
  ArrowLeft/ArrowRight previous/next; D add division, M remove division, E
  exclude, R restore, C clear labels, F resolve conflicts, U undo, K toggle
  reviewed, Delete remove trajectory, A apply predictions, T make training-ready,
  P preview and V evaluate. Each button displays its key in brackets. Letter
  keys accept either case. Only visible, enabled actions respond; hidden tabs
  and collapsed expanders do not. Ctrl/Alt/Meta, repeats and composition events
  are ignored, along with input/contenteditable/slider/combobox/table/dialog
  focus. The main lineage app retains its original bindings.
- Above the plot, a **Trajectory position** slider and **Trajectory number**
  numeric input choose a 1-based item in the filtered queue. Both synchronize
  with the dropdown/arrows and after filters, sorting, deletion or lineage
  changes; a one-item queue disables them and an empty queue hides them.
- Every label edit and deletion takes an online SQLite backup before writing;
  backups live in `review_backups/` beside the database. Stale sessions are
  rejected by strict compare-and-swap against a global revision.
- Undo restores the last label payload only, never a deletion or general
  database state.
- Editing/revising a frozen version in place copies its original pins/counts
  into review state, clears its frozen flag and stored content pin, and
  recomputes a working hash. Never claim the original frozen lineage remains
  byte-identical after user review; archival snapshots retain pre-change state.
- Training refuses a working/invalidated version. In the sidebar **Training
  version -> Create training-ready version**, attest that labels are final and
  click **Make training-ready** to create a **new** frozen dataset ID
  (`reviewed_<timestamp>`) with `human_verified = 1`, source-lineage provenance
  and an internal canonical-content pin. This is a training input stored in
  SQLite, not a CSV/file export. The new version applies the R004 exclusion-run
  filter to membership only; it never deletes trajectories. The core verifier
  accepts it only when audit/provenance/canonical hashes match.
- Once review data exists, historical CSV build/import is refused so a
  permanently deleted trajectory cannot be resurrected. Create a training-ready
  version, or build a separate historical store if importing old corpora is
  required.
- Model preview is explicit; calibrated thresholds are defaults only. Evaluation
  split badges (train/validation/test/unseen) describe the checkpoint's recorded
  split, not a new held-out estimate. Model previews do not promote checkpoints.

Full user-facing instructions: `docs/TRAINSET_REVIEW.md`. Implementation
evidence: `runs/trainset_review_implementation/`.

---

## 7. Reports and where the truth lives

| Artifact | Role |
|---|---|
| `docs/METHODS_TRAJECTORY_CLEANING_MODEL.md` | Scientific contract: method version, registered parameters, data-structure contract, METHOD_MANIFEST JSON block, round registry, declared-vs-actual reconciliation, change log. Update it every round. |
| `src/moma_lineage_analysis/trajectory_cleaning/MODEL_REPORT.md` | Model report: architectures, input normalization, training properties, per-round results, same-population comparisons, deployment status. |
| `datasets/README.md` | Human-readable corpus catalogue and per-batch provenance/caveats. |
| `datasets/manifest.json` | Integrity contract: sha256/rows/trajectories for every frozen corpus + trained models per corpus. Re-verify after any change. |
| `datasets/<corpus>/dataset_manifest.json` / `merge_manifest.json` | Per-corpus provenance; `human_verified: true` gates `--require-human-verified`. |
| `datasets/store/lineage.sqlite3` (+ `SCHEMA.md`, `backups/`, `retired_csv_manifest_20261017.json`) | Local SQLite dataset store (phases 1-3): de-duplicated observations, frozen dataset versions, lineage IDs, byte-identical CSV exports; training reads it directly and records lineage IDs; it is the sole training source after CSV retirement. |
| `src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion.json` | Deployment record: selected model, validation scores, artifact hashes, backup path. |
| `runs/<run>/r0xx_model_comparison.json`, `figures/`, `verification_report.json` | Run-level evidence: comparison table, plots, independent verification. |
| `promotion_backups/<timestamp>/` | Previous deployed artifacts (rollback + provenance). |
| `src/moma_lineage_analysis/trainset_review/` + `docs/TRAINSET_REVIEW.md` | R018 training-dataset review app and user guide: direct SQLite autosave, shared shortcuts, explicit label review, permanent deletion, working-version invalidation and training-ready version creation; no dataset export UI. Review evidence in `runs/trainset_review_implementation/`. |
| `src/moma_lineage_analysis/model_registry.py` | Shared packaged checkpoint registry: R019 production default, selectable R015 (non-production) and R002/R003 checkpoints. |

---

## 8. Governance rules (do and do not)

1. **Do not edit frozen corpora outside the review backend.**
   `datasets/*/verified_training_data*.csv`, `staged/`, `dataset_manifest.json`,
   `merge_manifest.json` are frozen; a new batch is a NEW folder. Rebuilding a
   corpus changes its sha and requires documentation + re-pinning. Only explicit
   user actions through `moma-trainset-review` may edit labels or permanently
   delete trajectories; those actions invalidate frozen/working training refs
   and must be followed by creating a new training-ready version. Measurements
   remain immutable.
2. **Do not select models on test data.** Selection is validation-only; test
   metrics are for acceptance reporting only.
3. **Never fabricate metrics, hashes, or provenance.** Recompute numbers from
   artifacts; if a check fails, report the failure rather than smoothing it.
4. **Provenance gate is mandatory for training:**
   `--dataset-manifest ... --require-human-verified`.
5. **Do not delete raw exports or backups** without user instruction;
   `datasets/raw_train_data/` is the curator's inbox (the user rotates files).
6. **One writer per file.** When several agents work in parallel, assign
   disjoint write scopes; the Lead reviews the final diff.
7. **Keep the R004 exclusion-run rule** (drop trajectories whose longest
   contiguous exclusion run >= 5 frames) when merging batches into a training
   corpus; it is a frozen production rule.
8. **Update all pins together:** corpus sha (datasets/manifest.json + corpus
   manifests), checkpoint sha (tests + docs + app), method version (contract
   header + METHOD_MANIFEST), and report dates.
9. **Verification:** after data/model/doc changes run at least
   `pytest tests/trajectory_cleaning/test_contract_integrity.py -q`; run the full
   `pytest tests -q` for code changes. Independent re-derivation by a second
   agent is expected for dataset builds and promotions.
10. **Report honestly, including caveats** (single seed, exclusion-positive
    scarcity, trajectory-grouped (not FOV-held-out) splits, mixed label policy,
    cross-round comparability limits).

---

## 9. Current state (lineage updated 2026-10-26; model/data as of 2026-10-31)

- Lineage-construction method `1.1.0`, round R002 (2026-10-26), `LOGIC_VERSION 2.3`; app statistics schema 2. Founder divisions count non-self children before the root stop; exact daughter-length conservation is accepted, with strict upper and first-daughter bounds retained. Fresh **Time Unit** is `s`; explicit `d` remains supported and unknown units raise `ValueError`. Statistics use stable numeric-time copies, finite positive fit observations, centered OLS with rank checks and explicit residual R², and `RuntimeWarning` for invalid observations/degenerate fits. Reconstruct with **Process Data** for new lineage rules; cached-statistics invalidation does not migrate existing annotations. Regression commands/outcomes are tracked in `docs/METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md`; R001 results remain historical.
- Upstream spine length/width are already µm and area µm² (`Cell` and `CellDT` calibrate by `umppx` before export); `time_s` comes from `time_points_seconds`. Do not rescale numerical lengths or change the minimum-length filter to repair the earlier pixels wording. `fov_cell_index` is a unique observation ID within the loaded FOV, not persistent identity.
- Cleaning-model method version `0.11.1`; R014/R015 are closed, R016 is closed, R018 is closed and R019 is independently verified PASS. R020 (2026-10-31) is the deployment round: the R019 CNN was promoted to production on explicit user request (checkpoint sha256 `816a8b39...`; previous R010 CNN backed up). R019 (2026-10-31) removed the three duplicate `verified_merged_*` dataset versions, finalized the five reviewed batches as `reviewed_20261031_*` training-ready lineages (R004 membership filter), and retrained cnn/resnet/lstm on their composition. R018 remains the independent training-review app. R001-R016 history and lineage method 1.1.0 / LOGIC_VERSION 2.3 are unchanged.
- Production checkpoint: **R019 CNN**, sha256
  `816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99`,
  promoted 2026-10-31 (R020, user decision) from `runs/reviewed_20261031/cnn/`;
  previous R010 CNN (`6e7c5a58...`) plus its archived `promotion.json` backed up
  at
  `src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261030T184217.243155Z/`.
- Current retraining corpus (R019): composition of the five reviewed
  training-ready lineages `reviewed_20261031_*` = 385,815 rows / 1,152
  trajectories / 33,665 divisions / 1,740 exclusions / 49 legacy overlap rows,
  eight sources. The reviewed population differs from R015 (user review
  permanently deleted 45 trajectories and edited labels before training), and
  finalization applied the R004 membership filter to three trajectories. The
  previous R015 lineage `verified_merged_20261029@v1#8decab478c80`
  (399,168 / 1,197) remains documented but its store version was removed with
  the two other merged duplicates on 2026-10-31.
- R010 results: cnn 43 epochs (test division-event F1 0.9957, exclusion F1 0.8512);
  resnet 25 epochs (0.9964 / 0.7973); lstm 25 epochs (0.9933 / 0.7993).
  Paired same-population comparison: R010 CNN exclusion AP +0.0245
  [+0.0049, +0.0489] vs R008 CNN on the R010 test split and +0.0335
  [+0.0120, +0.0598] on the R008 test split; replicates on leakage-free
  subsets (+0.0358 / +0.0337); division metrics unchanged/saturated.
- Dataset store (2026-10-31, R019): ten versions over 387,217 unique
  observations / 1,155 trajectories - five reviewed working originals
  (`frozen = 0`) and five `reviewed_20261031_*` training-ready versions
  (`frozen = 1`); 2,307 memberships / 23 ingest-provenance rows; review
  revision 567. The three `verified_merged_*` dataset versions were removed at
  the user's request (revision 562, 3,345 memberships, zero orphaned
  trajectories) and their manifest entries carry `store_removed` markers; the
  pre-removal database is `store/backups/lineage_20261030T181226.223866Z.sqlite3.gz`.
  Store verification after R019 passes with 5 WORKING + 5 database-native
  versions. Training reads the store directly and records lineage IDs; the six
  historical batch CSVs remain retired and restorable byte-identically from the
  live store. Full regressions for R018/R019 are recorded under
  `runs/trainset_review_implementation/` and `runs/reviewed_20261031/`.
- R015 results: cnn 40 epochs (best 25; test division-event F1 0.9954,
  exclusion F1 0.8026, AP 0.8779); resnet 21 (0.9952 / 0.7669 / 0.8527);
  lstm 23 (0.9921 / 0.7202 / 0.7237). Validation CNN winner mean F1 0.8886.
  Common historical test panel: 103 trajectories; R015 CNN exclusion F1/AP
  0.8099/0.8791 versus R010 production 0.8232/0.8999 (lower), division-event
  F1 0.9931 versus 0.9920. New-batch panel: 7 trajectories / 13 exclusions;
  R015 CNN exclusion F1/AP 0.4242/0.6273 versus 0.3750/0.4883 (higher, small
  support). No bootstrap/significance; evidence does not support replacement.
  Final verification: PASS_WITH_DOCUMENTED_CAVEATS; 441 recorded checks
  (317 data / 35 retirement / 65 training-cross-evaluation / 24 final integrity),
  no failures; production remains R010 CNN.
- R016 (registered 2026-10-29): fixed R015 lineage/split and hidden width 64;
  CNN kernel5 (63,938 params) and primary ResNet kernel3 (100,354 params) vary
  four-entry dilation schedules for nominal normalized-input RF 17-197. Old
  ResNet kernel5 RF121 (165,890) is a separate control. The eight schedules
  have dense support but dilation tap multiplicities change, so context is not
  an isolated causal variable or a raw end-to-end observation window.
  All 29 runs complete; lock selected RF17 for both families (three-seed
  validation 0.885582 +/- 0.004732 CNN; 0.887547 +/- 0.006512 primary ResNet;
  `test_evaluated_before_lock=false`). Locked three-seed test means (CNN
  0.7972 F1 / 0.8659 AP; ResNet 0.8043 / 0.8759) stay below the R010 production
  control (0.8183 / 0.8960; common panel 0.8999), so nothing here supports
  replacement. Smallest screened context ranked best, but this is not a
  universal optimum or causal RF effect. Independent verification:
  PASS_WITH_DOCUMENTED_CAVEATS, 21/21 checks, 183 replay checks per GPU, GPU0
  replay exact on all discrete metrics; Lead audit 438 checks PASS after an
  audit-only RF formula fix that did not change training code.
  Seed42 shortlist reuse creates selection optimism; seed spread is optimization
  noise, not FOV generalization. Authorized disjoint GPU workers change scheduling
  only; timings are device-specific and not compared across devices.
- R018 review app (2026-10-29): independent `moma-trainset-review` Streamlit
  app, nine source modules, shared registry (R010 production default; R015
  selectable but non-production). User-authorized app mutations are label edits
  and permanent whole-trajectory deletion; `time_h`/`spine_length` are immutable.
  Every edit/deletion is global-revision CAS-guarded and online-backed up under
  `review_backups/`; undo is label-only. Edits and reviewed marks save directly
  to SQLite with no Save button; per-row table checkboxes autosave too. The UI
  has only Review trajectories and Evaluate dataset tabs and no dataset CSV/
  export action. It uses the shared cycle-coloured plot style and shortcut
  handler. Reviewed frozen versions become working and training refs are refused
  until a new frozen training-ready version (`reviewed_<timestamp>`) is created
  with R004 membership filtering, attestation, source provenance and internal
  content pin. Historical CSV import is refused after review to prevent deleted-
  trajectory resurrection. Original frozen lineage bytes must not be claimed
  unchanged after edits. Final reannotation-render suite 425 passed / 4 skipped / 59
  subtests (`runs/trainset_review_implementation/pytest_fast_actions_final.txt`); prior
  independent task-26 verification PASS with zero open defects (backend 54,
  registry 25, scoped helper 86, AppTest 4, opaque/cache 13). The UI refinements
  were independently verified by task-29 (PASS, 14 checks: shared plot style,
  arrow/letter shortcuts with input guards, table-checkbox autosave persistence,
  no dataset export, sidebar-only training-ready creation, live WAL preserved).
  The subsequent user-requested one-click removal below the plot passed
  temporary-database browser verification (one delete event and backup, next
  trajectory selected) and tests for queue boundaries, empty filters, failed
  saves, stale clicks, persistence and unchanged remaining trajectories. See
  `runs/trainset_review_implementation/one_click_remove_report.json`.
  Training-ready flow, stale-view guard and auto/CUDA device resolution were
  re-tested. The review workspace is a `st.fragment`: label edits, deletion and
  reviewed actions rerun only that panel, not the page/sidebar. Mutations save
  inside widget callbacks (one rerun, not a second page refresh); the plot
  figure object is reused and only its traces are replaced; preview predictions
  are cached on measurements so label edits do not force re-inference. Save
  confirmation is a three-second bottom-left toast without the backup path, so
  the plot never shifts; there is no separate observation-ID picker and label
  actions use the plot lasso/box selection. Label edits update the cached queue
  row instead of re-running the aggregate queue query, advance version counts by
  the known label delta instead of full observation scans, and keep the large
  table grid collapsed unless **Enable table editor** is checked. On a
  representative live-size copy the save path fell from about 420 ms to about
  204 ms; the remaining response cost is Streamlit's Plotly element refresh. Implementation and verification used
  temporary databases only; the production checkpoint hash is unchanged. The
  live store already contains the user's own review activity from trying the
  app (a label save and a reviewed mark held in its WAL), which was preserved.
- R019 results (2026-10-31): retrained on the 385,815-row / 1,152-trajectory
  reviewed composition, split 806/173/173 (identical across models). CNN 31
  epochs (best 16), ResNet 20 (5), BiLSTM 29 (14); validation mean F1
  0.901205 / 0.860075 / 0.884530 selects the CNN; test division-event F1
  0.996583 / 0.995809 / 0.996189 and exclusion F1 0.797642 / 0.749091 / 0.815238
  (AP 0.878254 / 0.830463 / 0.875289). Descriptive same-row comparison with
  production R010: common clean panel 35 trajectories / 44 exclusions, R019 CNN
  exclusion F1/AP 0.8395/0.9034 vs production 0.8250/0.8618; new-batch panel
  7 / 14, 0.7586/0.8654 vs 0.3636/0.4804; full-test panel contaminated for
  production (130/173 keys seen in R010 training). Single seed, small panels,
  no bootstrap/significance; the comparison is pre-promotion context only.
  Artifacts: `runs/reviewed_20261031/` (model_comparison.json,
  production_comparison.json, training_summary.json, figures/, verification/).
  Independent verification: PASS, 29/29 checks, 0 failures, 3
  documentation-level discrepancies (duplicate pairs straddling splits,
  retained source_file metadata, AP-only CPU replay drift <=3.24e-5). Full
  pytest suite: 429 passed / 4 skipped / 59 subtests.
- R020 deployment (2026-10-31): the user requested promotion of the new CNN to
  the main app. `promote_best_model` re-applied the validation-only rule (CNN
  0.901205 > BiLSTM 0.884530 > ResNet 0.860075), required identical split
  manifests and hash-verified the copied artifacts: production `best_model.pt`
  sha256 `816a8b39...` (byte-identical to `runs/reviewed_20261031/cnn/best_model.pt`).
  Previous R010 CNN plus its archived `promotion.json` are under
  `promotion_backups/20261030T184217.243155Z/`; registry, AI-panel default
  label, contract-integrity pin, README/AGENTS/datasets docs updated; no data,
  split, threshold or training run changed. Health checks: load-checkpoint +
  predict smoke finite; post-promotion full pytest 430 passed / 3 skipped / 59
  subtests.
- Open items: FOV-held-out cross-validation and per-source uncertainty. Width tuning may now proceed if
  the user wants it, since the R016 context question is answered for the
  screened schedules; three seeds do not provide FOV robustness. The R019
  comparison panels are too small (44 and 14 exclusion positives) to support a
  strong generalization claim; R020 promoted the model on explicit user
  decision, so the next step is a larger leakage-free comparison or FOV-held-out
  design to validate the deployed model.

---

## 10. Known caveats / gotchas

- Plain `python` on PATH lacks pandas/torch - use the conda env interpreter.
- `promote_best_model.py` does not back up the previous `promotion.json`.
- Mixed label policy in merged corpora: 295 legacy R002-policy rows inside the
  frozen R004 corpus carry both labels; curated batches have overlap 0.
- Exclusion positives are scarce (a few hundred per test split); exclusion F1
  has wide sampling uncertainty - prefer AP and paired/bootstrap comparisons.
- Splits are trajectory-grouped; FOVs can appear in more than one partition, so
  metrics are optimistic about generalization to new FOVs. Cross-round splits
  also overlap: a trajectory in one round's test set may be in another round's
  training set (check before comparing rounds).
- Merged corpora are not byte-identical to a naive concatenation of the
  unfiltered batch CSVs (the R004 filter drops trajectories). After applying the
  documented R004 filter, re-running the merge reproduces the merged CSV
  byte-for-byte (verified for R010, sha 7ecc9753...).
- The six `datasets/verified_*/verified_training_data*.csv` files were retired
  (deleted) on 2026-10-17 after sign-off; the store exports them byte-identically
  on demand. `scripts/dataset_store.py verify` accepts `legacy_csv_state:
  retired`; use `verify --require-legacy-csv` to demand the pre-retirement state.
- Never open GUI figure windows during training. `plot_training_results.py` uses
  the non-interactive Agg backend and saves report PNGs; the standalone training
  experiment `tests/test_cell_lineage_sperate.py` saves figures under
  `tests/figures_cell_lineage_experiment/` (set `MOMA_SHOW_FIGURES=1` only when
  you deliberately want windows). Do not add `plt.show()` to training or plotting
  code - always `savefig` + `plt.close`.
- R016 nominal RF is a normalized-input property, not a strict raw observation
  window because full-trajectory normalization is unchanged. Winners are only
  best among screened/shortlisted schedules; the irregular grid does not support
  interpolation. Primary ResNet kernel3 differs from old kernel5, so do not pool
  the primary sweep with the separate control or infer pure RF causality.
- R016 nominal frames are not a fixed physical window: validation RF17 spans a
  median ~0.803 h and covers all 179 trajectories, while RF197 spans ~9.831 h
  and covers 105 trajectories (47.1% of row centers), varying by source. Compare
  contexts on the matched radius-98 interior, where locked-model scores exceed
  their boundary scores for every seed.
- R016 metric replay is device-sensitive: one CNN RF17 seed-42 validation
  probability lies within 1e-5 of the 0.87 exclusion threshold, so replaying it
  on another device flips that point (precision -0.002738, F1 -0.001255), and AP
  ties differ by <=1.6e-4 across devices. Compare discrete metrics on the
  original training device.
- R018 review app opens the editable live store. Use a copy for experiments.
  Permanent deletion has no in-app restore; backups under `review_backups/`
  beside the database are archival recovery, not an automatic undo. Do not
  claim edited historical frozen versions remain byte-identical.
- R019 removed the three `verified_merged_*` dataset versions from the live
  store (user request). Their historical lineage ids no longer resolve there;
  use `store/backups/lineage_20261030T181226.223866Z.sqlite3.gz` for any
  historical reproduction. Dataset-version removal is only allowed through
  `remove_dataset_versions`, which refuses to orphan trajectories.
- Two identical-content trajectory pairs exist in the live store (both reviewed
  batches vs their RDM-Gluc re-curations, `CH_16` and `CH_8`; the second was
  created by R018 label edits). They straddle R019 partitions
  (validation/train and test/validation), so the R019 test metrics carry an
  exact-content-twin leakage caveat; the independent verifier flagged this as a
  documentation-level discrepancy, not a count error. `source_file` also keeps
  three historical merged-CSV import records after the R019 removal (metadata
  only).
- GPU training is not bit-reproducible for the CNN: rerunning the same seed and
  data drifts weights by <=3e-4 and average precision by <=9e-5, while thresholds,
  discrete metrics, event counts and splits stay identical (ResNet/BiLSTM
  reproduced bit-identically in the phase-2 store run). Compare models on
  thresholds/discrete metrics and paired AP, not on bitwise checkpoint equality.

---

## 11. Dataset store (phases 1-3)

`datasets/store/lineage.sqlite3` holds ten versions over 387,217 unique
observations / 1,155 trajectories after the R019 review and cleanup (2026-10-31):
five reviewed working originals (`frozen = 0`) and five review-finalized
training-ready versions (`reviewed_20261031_*`, `frozen = 1`, R004 membership
filter). The three `verified_merged_*` dataset versions were removed from the
store at the user's request on 2026-10-31 because every trajectory they held is
also held by a batch version; observations and trajectory rows were unchanged
and the pre-removal database is archived in `store/backups/`. The original
six-version R012 store held 375,348 observations / 1,152 trajectories at 13.5 MB
versus 348 MB of CSV corpora.
Phases 1-3 are complete: the store is built and verified, the training CLI
reads it directly and records lineage IDs in `metrics.json`, and the six legacy
corpus CSVs were retired after user sign-off (restorable byte-identically from
the store). That original retirement reduced `datasets/` from ~348 MB to ~29 MB.
The two R014/R015 corpus CSVs were also retired on 2026-10-29 after independent
data verification; the ledger contains eight records. Store verification after
R019 passes with 5 WORKING + 5 database-native versions. Staged provenance and
raw exports remain.

- **Lineage ID:** `<dataset_id>@v<version>` (optionally `#<hash12>` content ref).
  Training metadata must record lineage IDs (phase 2).
- **Two hashes per version:** canonical content hash + legacy CSV sha256;
  exports are byte-identical to the historical CSVs, so corpus pins stay valid.
- **Immutable by default:** re-importing a trajectory key with different content
  is rejected; corrections create new versions. R018 explicitly allows direct
  user label edits/permanent trajectory deletion through the guarded review
  backend; reviewed versions become working and training requires a new frozen
  training-ready version.
- **Review extension:** review tables are created lazily on the first mutation;
  read-only app launch never writes. Backups live in `review_backups/` beside
  the selected database. Historical CSV import is refused once reviewed working
  data exists, preventing permanently deleted trajectories from returning.
- **Commands:** `python scripts\dataset_store.py build | verify | export | backup`;
  train with `moma-trajectory-cleaning train --dataset-lineage <id> --store
  datasets\store\lineage.sqlite3` (repeat `--dataset-lineage` to compose the five
  reviewed lineages). `build` skips retired corpora that are already in the
  store and manifest entries marked `store_removed`; `verify` accepts only CSVs
  recorded as retired in the retirement manifest (strict `--require-legacy-csv`
  demands the pre-retirement state).
- **Dataset-version removal (admin):** `python scripts\remove_dataset_versions.py
  --expected-revision N --reason ... --lineage-id <id> [--lineage-id ...]` removes
  whole duplicate versions only when every trajectory stays referenced by
  another version; it is review-revision CAS-guarded, takes an online backup and
  writes a `remove_dataset_versions` audit event.
- **Schema:** `datasets/store/SCHEMA.md`; design and phase plan:
  `docs/PLAN_DATASET_STORE.md`.
- **Phase 2 (done 2026-10-17):** store-direct training + `dataset_lineage` in
  `metrics.json`; R010 reproduced from `verified_merged_20261017@v1` in
  `runs/merged_20261017_store_repro/` (byte-identical splits, exact resnet/lstm,
  cnn AP drift <=9e-5 attributable to GPU nondeterminism).
- **Phase 3 (done 2026-10-17):** the six `verified_*/verified_training_data*.csv`
  files were retired after user sign-off; provenance JSONs and
  `datasets/manifest.json` were kept. Every retired file's sha256/size and a
  restore command are in `datasets/store/retired_csv_manifest_20261017.json`;
  `verify` reports `legacy_csv_state: retired`, and `verify --require-legacy-csv`
  passes again only after a corpus is restored.
