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
│   ├── moma_logic.py                  # lineage tracing, division detection, per-cycle stats (LOGIC_VERSION 2.2)
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
│   └── models/trajectory_cleaning_run/  # released + promoted checkpoints (see section 3.5)
├── scripts/
│   ├── qc_raw_train_data.py           # technical QC of raw *_selected_labeled.csv exports
│   ├── stage_verified_labels.py       # canonical label staging (R002 policy / effective mode)
│   ├── merge_corpora.py               # merge verified corpora + R004 filter on incoming batches
│   └── dataset_store.py               # build/verify/export/backup the SQLite dataset store
├── datasets/                          # frozen verified corpora + integrity manifest + store/ (section 11)
├── docs/
│   ├── METHODS_TRAJECTORY_CLEANING_MODEL.md  # scientific contract (rounds R001-..., method versions)
│   ├── METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md  # lineage-construction contract
│   └── PLAN_DATASET_STORE.md          # dataset-store design + phase status
├── runs/                              # training runs, comparisons, figures, verification reports
├── tests/                             # unittest/pytest + Streamlit AppTest suites + fixtures
├── pyproject.toml                     # entry points: moma-lineage, moma-trajectory-cleaning
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

Console scripts `moma-lineage` and `moma-trajectory-cleaning` exist in that env.
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

- `best_model.pt` - PRODUCTION checkpoint: **R010 CNN**, sha256
  `6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`,
  promoted 2026-10-17 (validation-rule winner, mean F1 0.8987); trained on
  `datasets/verified_merged_20261017` (1,149 trajectories). Its companions at
  the root (`history.csv`, `metrics.json`, `splits.json`) are the R010 artifacts;
  `promotion.json` records the promotion scores/hashes/backup.
- `promotion_backups/20261017T124420.980675Z/` - previous R008 CNN artifacts
  (best_model.pt sha256 `3e796c9e...`), rollback source. Older backups:
  `20261016T145403.716334Z/` (R004-era artifacts).
- `verified_20260916/{cnn,resnet,lstm}/` - R002 mixed-corpus checkpoints.
- `rdm_only_20260916/{cnn,resnet,lstm}/` - R003 RDM-Gluc-only specialists.
- The Streamlit app registry (`AI_MODEL_REGISTRY` in
  `app/Moma_lineage_streamlit.py`) exposes the production model plus the R002/R003
  checkpoints; `MOMA_AI_MODEL_PATH` overrides the default checkpoint.

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
   `<batch>/rename_map.json`. Also check prefix-normalized overlap and document
   biological re-curations. Do not silently collide.
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
    --dataset-lineage verified_merged_20261017@v1 `
    --store datasets\store\lineage.sqlite3 `
    --require-human-verified `
    --source-column source_dataset `
    --model $m `
    --output-dir runs\merged_20261017\$m `
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
     --run-dir runs\merged_20261017 `
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

---

## 8. Governance rules (do and do not)

1. **Do not edit frozen corpora.** `datasets/*/verified_training_data*.csv`,
   `staged/`, `dataset_manifest.json`, `merge_manifest.json` are frozen; a new
   batch is a NEW folder. Rebuilding a corpus changes its sha and requires
   documentation + re-pinning.
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

## 9. Current state (as of 2026-10-17)

- Method version `0.7.0`; rounds R001-R013 closed (R011 = R010 CNN promotion; R012 = dataset-store training layer; R013 = legacy-CSV retirement; neither R012 nor R013 changed a model/metric).
- Production checkpoint: R010 CNN, sha256
  `6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`;
  previous R008 CNN backed up at
  `src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261017T124420.980675Z/`.
- Current retraining corpus: store lineage `verified_merged_20261017@v1`
  (373,440 rows / 1,149 trajectories; split 804/172/173). Its CSV was retired in
  phase 3 and can be re-exported from the store on demand.
- R010 results: cnn 43 epochs (test division-event F1 0.9957, exclusion F1 0.8512);
  resnet 25 epochs (0.9964 / 0.7973); lstm 25 epochs (0.9933 / 0.7993).
  Paired same-population comparison: R010 CNN exclusion AP +0.0245
  [+0.0049, +0.0489] vs R008 CNN on the R010 test split and +0.0335
  [+0.0120, +0.0598] on the R008 test split; replicates on leakage-free
  subsets (+0.0358 / +0.0337); division metrics unchanged/saturated.
- Dataset store (phases 1-3, 2026-10-17): `datasets/store/lineage.sqlite3` holds
  the six frozen corpora as versioned membership sets over 375,348 de-duplicated
  observations (13.5 MB; byte-identical CSV exports verified). Training reads the
  store directly and records lineage IDs; R010 was reproduced from
  `verified_merged_20261017@v1` (`runs/merged_20261017_store_repro/`). The six
  legacy corpus CSVs were retired after sign-off and are restorable from the
  store; `datasets/` is now ~29 MB.
- Open items: FOV-held-out cross-validation, per-source uncertainty (trajectory
  bootstrap), seed-robustness, and a next retraining round if new curated data is
  added.

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
- GPU training is not bit-reproducible for the CNN: rerunning the same seed and
  data drifts weights by <=3e-4 and average precision by <=9e-5, while thresholds,
  discrete metrics, event counts and splits stay identical (ResNet/BiLSTM
  reproduced bit-identically in the phase-2 store run). Compare models on
  thresholds/discrete metrics and paired AP, not on bitwise checkpoint equality.

---

## 11. Dataset store (phases 1-3)

`datasets/store/lineage.sqlite3` is a local SQLite store holding the frozen
corpora as versioned membership sets over de-duplicated observations (375,348
unique observations / 1,152 trajectories; 13.5 MB versus 348 MB of CSV corpora).
Phases 1-3 are complete: the store is built and verified, the training CLI
reads it directly and records lineage IDs in `metrics.json`, and the six legacy
corpus CSVs were retired after user sign-off (restorable byte-identically from
the store). `datasets/` dropped from ~348 MB to ~29 MB.

- **Lineage ID:** `<dataset_id>@v<version>` (optionally `#<hash12>` content ref).
  Training metadata must record lineage IDs (phase 2).
- **Two hashes per version:** canonical content hash + legacy CSV sha256;
  exports are byte-identical to the historical CSVs, so corpus pins stay valid.
- **Immutable:** re-importing a trajectory key with different content is rejected;
  corrections create new versions.
- **Commands:** `python scripts\dataset_store.py build | verify | export | backup`;
  train with `moma-trajectory-cleaning train --dataset-lineage <id> --store
  datasets\store\lineage.sqlite3`. `build` skips retired corpora that are already
  in the store; `verify` accepts only CSVs recorded as retired in the retirement
  manifest (strict `--require-legacy-csv` demands the pre-retirement state).
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
