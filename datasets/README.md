# datasets/

Frozen, human-verified training corpora for the trajectory-cleaning models,
stored once in the local SQLite dataset store (`store/lineage.sqlite3`) since
phase 3. Integrity contract: per-file sha256 in `manifest.json` and per-version
canonical hashes in the store.

## Contents

| File | Role | Rows | Trajectories |
|---|---|---:|---:|
| `verified_20260916/verified_training_data_no_longruns.csv` | Production training corpus (round R004) | 312,317 | 1,053 |
| `verified_20261013/verified_training_data.csv` | Curated batch 2026-10-13 (round R006; merged in R010) | 22,890 | 35 |
| `verified_20261016/verified_training_data.csv` | Curated batch 2026-10-16 (round R007; merged in R010) | 31,384 | 48 |
| `verified_20261017/verified_training_data.csv` | Curated batch 2026-10-17 (round R009; merged in R010) | 8,757 | 16 |
| `verified_merged_20261016/verified_training_data.csv` | R008 retraining corpus = R004 + batches 20261013/20261016 (R004 filter applied to batches) | 365,283 | 1,134 |
| `verified_merged_20261017/verified_training_data.csv` | R010 retraining corpus = R004 + batches 20261013/20261016/20261017 (R004 filter applied to batches) | 373,440 | 1,149 |

**Phase 3 (2026-10-17): all six CSVs listed above are retired** (deleted after
user sign-off). The store holds the same frozen rows; each export is
byte-identical to the retired file and matches its pinned sha256. Every retired
file, its sha256/size and a restore command are recorded in
`store/retired_csv_manifest_20261017.json`; restore, e.g.:

```powershell
python scripts\dataset_store.py export --dataset verified_merged_20261017 --version 1 `
    --output datasets\verified_merged_20261017\verified_training_data.csv
```

Schema (one row = one cell observation):
`trajectory_key, fov_cell_index, time_h, spine_length, division_point, excluded_point, source_dataset, source_file, source_file_sha256, source_trajectory_key, source_fov_cell_index`

Targets: `division_point`, `excluded_point`. Features: `time_h`, `spine_length`.

## Provenance

Built from human-curated `selected_labeled.csv` exports via
`trajectory_cleaning/prepare_verified_training_data.py` (R002 corpus, 1,070
trajectories), then filtered per the R004 rule — drop 17 trajectories whose
longest contiguous exclusion run is ≥ 5 frames (analysis-driven cell-cycle
removals, not growth noise; flagged keys in
`longrun_excluded_trajectories.json`). Full chain:
`docs/METHODS_TRAJECTORY_CLEANING_MODEL.md`.

- `dataset_manifest.json` — construction manifest of the upstream R002 corpus
  (per-source files, hashes, label policy, human-verified attestation)
- `longrun_excluded_trajectories.json` — the 17 R004-filtered trajectory keys
- `splits_no_longruns.json` — train/validation/test split used for the R004 runs

Note: the upstream R002 corpus and the raw curated exports (needed to rebuild
this trainset from scratch) are intentionally **not** part of this package.

## Batch 2026-10-13 (`verified_20261013/`)

35 newly curated single-trajectory exports (`raw_train_data/`) were verified
(`scripts/qc_raw_train_data.py`; the transient raw-inbox report was rotated away - canonical copy: `verified_20261013/qc_report.json`).
11 keys duplicating R004-corpus trajectories were renamed with a `20261013_`
prefix (`verified_20261013/rename_map.json`), so the batch is now
duplication-free. Labels were staged with
`scripts/stage_verified_labels.py --division-mode effective`
(report: `verified_20261013/staging_report.json`): **`division_point` =
`effective_division_point`** — the final revised cycle-end boundary recomputed
via the packaged resolver (1,932 divisions; 22 raw labels superseded, 6
effective-only boundaries added; division/exclusion overlap = 0). Combined
under `dataset_id = curated_20261013` with `prepare_verified_training_data.py`.

Caveat: `curated_20261013::FOV_121_CH_7` carries a 5-frame exclusion run
(would be dropped by the R004 filter at merge time). Its raw exports were
rotated out of `raw_train_data/`; the batch's QC report is archived at
`verified_20261013/qc_report.json`.

## Batch 2026-10-16 (`verified_20261016/`)

48 newly curated single-trajectory exports (rotated `raw_train_data/`
contents) were verified (`scripts/qc_raw_train_data.py`; the transient raw-inbox report
was rotated away and no durable R007 copy remains). 18 keys duplicating R004-corpus
trajectories were renamed with a `20261016_` prefix
(`verified_20261016/rename_map.json`) — duplication-free against both existing
corpora. Labels staged with `--division-mode effective`
(`verified_20261016/staging_report.json`): 2,698 divisions (37 raw labels
superseded, 7 effective-only added; division/exclusion overlap = 0). Combined
under `dataset_id = curated_20261016`.

Caveat: `curated_20261016::FOV_53_CH_11` carries a 5+-frame exclusion run
(would be dropped by the R004 filter at merge time).

## Batch 2026-10-17 (`verified_20261017/`)

16 newly curated single-trajectory exports (8,757 rows; five 432-frame grids
and eleven 600-frame-profile exports — 9 complete 600-row files plus the
599- and 598-row gapped siblings) passed technical QC
(`scripts/qc_raw_train_data.py --corpus datasets/verified_merged_20261016/...`:
0 hard-gate failures; 553 raw division labels, 138 excluded labels). Nine keys
exactly duplicating R004-corpus trajectories were renamed with a `20261017_`
prefix (`verified_20261017/rename_map.json`); no other key duplicated an
existing corpus, exact or prefix-normalized. Labels were staged with
`scripts/stage_verified_labels.py --division-mode effective`
(report: `verified_20261017/staging_report.json`): **`division_point` =
`effective_division_point`** — 529 divisions (25 raw labels superseded:
3 `coalesced_previous`, 1 `shifted_previous`, 21 `needs_review`; 1
effective-only boundary added; division/exclusion overlap = 0). The export's
`effective_division_point` was verified to reproduce the packaged resolver
bitwise for all 16 trajectories before staging. Combined under
`dataset_id = curated_20261017` with `prepare_verified_training_data.py`;
the batch is namespaced `curated_20261017::<key>` and was **merged in R010**
into `verified_merged_20261017` (production corpus unchanged).
`source_trajectory_key` holds the post-rename staged key; the pre-rename raw
key is recoverable from `verified_20261017/rename_map.json` (raw exports are
unmodified). The raw exports remain in `raw_train_data/`; the QC report is
stored both there and at `verified_20261017/qc_report.json`.

Caveats (all audited, none blocking):

- `curated_20261017::FOV_14_CH_14` carries a 16-frame contiguous exclusion run
  (rows 569-584, 28.60-29.35 h; 74 excluded rows total, plus 6/5/5-frame runs)
  and would be dropped by the R004 production filter (>= 5 frames) at merge.
- `curated_20261017::20261017_FOV_13_CH_19`: 13 of its raw division labels sat
  on rows also labeled excluded; the staged policy supersedes those raw labels,
  leaving 23 divisions (26 excluded rows in short runs, longest 4 frames), so
  its effective division labels stop at 12.72 h.
- `FOV_12_CH_17` has 599 of 600 frames (one gap larger than 2x the median step)
  and `FOV_15_CH_2` has 598; every other file is a complete 432/600-frame grid.
  Both are complete exports with internal missing frames (strictly increasing
  `time_h`, durations matching their movie siblings), not truncated files;
  features agree with the raw exports to <= 3e-15 relative (measured max
  2.01e-15; pandas float-parser artifact, not data corruption) and labels and
  row membership are exact.
- The 600-frame / 30.1 h grid (9 complete files plus the 599- and 598-row
  siblings) is a new movie length with no precedent in the current corpus
  (longest existing trajectory 28.3 h); this is a data-addition property, not a
  QC defect. Five 432-frame / 21.6 h files match the RDM-Gluc profile. Three
  432-grid files carry the usual cyclic timestamp jitter (step 0.028-0.072 h),
  the same pattern present in the RDM-Gluc corpus.
- Four trajectories keep only 1-2 effective divisions (`FOV_14_CH_1`: 1 after
  2 superseded labels; `FOV_14_CH_3`: 2 after 1; `FOV_15_CH_23` and
  `FOV_15_CH_2`: 2 each with no supersession). Their spine-length drops were cross-checked:
  they are genuinely non-dividing/arrested tracks, not under-labeled, but they
  shift the low-division tail (the existing 1,134-trajectory corpus has a single
  trajectory with <= 2 divisions).
- `curated_20261017::20261017_FOV_0_CH_16` is content-identical (same
  observations and labels) to `rdm_gluc_toggle_20260905::FOV_0_CH_16`: the
  renamed re-curation is an exact copy. It is retained because the batch is
  frozen, but it duplicates one trajectory in the merged training corpora
  (found during the 2026-10-17 dataset-store de-duplication audit).
- 16 trajectories support only a 12/2/2 trajectory split; the batch is too
  small to train standalone and is intended for merging into the production
  corpus (R004 + curated batches) like the earlier batches.

## Merged corpus 2026-10-17 (`verified_merged_20261017/`)

`scripts/merge_corpora.py` concatenated the four verified sources — the
pre-filtered R004 corpus (`--corpus`) plus the three curated batches
(`--filtered-corpus`, R004 long-exclusion-run filter applied to each). The
filter dropped exactly `curated_20261013::FOV_121_CH_7`,
`curated_20261016::FOV_53_CH_11` and `curated_20261017::FOV_14_CH_14`;
retained batch rows are 22,236 / 30,730 / 8,157 and the longest exclusion run
over all 1,149 trajectories is 4 frames. Result: **373,440 rows / 1,149
trajectories / 32,470 `division_point` / 1,924 `excluded_point`**, sha256
`7ecc9753cf02c0dfc020fd70f6b077f997799d46d4f3af3fd89d12a16eaee2bc`
(provenance: `verified_merged_20261017/merge_manifest.json`).

Trajectories per source dataset: baseline_0_10h 477, baseline_0_8h 19,
rdm_gluc_toggle_20260905 546, starvation_20251111 11, curated_20261013 34,
curated_20261016 47, curated_20261017 15.

Label-policy note: 295 rows carry both labels — legacy R002-policy residue
inside the frozen R004 corpus only (baseline_0_10h 161, baseline_0_8h 8,
rdm_gluc_toggle_20260905 126, starvation 0); all three curated batches have
overlap 0. Documented and expected, not a defect.

Round R010 trained cnn/resnet/lstm with the unchanged default TrainConfig
(seed 42, max 100 epochs, patience 15, batch 16, GPU) via the packaged CLI
(`--require-human-verified --source-column source_dataset`); the per-source
stratified seed-42 split is 804 train / 172 validation / 173 test trajectories
(261,377 / 55,866 / 56,197 rows; every source in every partition). Selection
is validation-only (mean of validation division-event F1 and exclusion F1);
test metrics are never used for selection:

| Model | Epochs | Val mean F1 | Test division-event F1 | Test exclusion F1 |
|---|---:|---:|---:|---:|
| cnn | 43 | 0.8987 | 0.9957 | 0.8512 |
| resnet | 25 | 0.8929 | 0.9964 | 0.7973 |
| lstm | 25 | 0.8642 | 0.9933 | 0.7993 |

The validation rule selects the **cnn**, and it was **promoted on 2026-10-17**
by user decision (round R011): production is now the R010 CNN
(`src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt`,
sha256 `6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`,
byte-identical to `runs/merged_20261017/cnn/best_model.pt`); the previous R008
CNN is backed up under `promotion_backups/20261017T124420.980675Z`. Artifacts:
`runs/merged_20261017/{cnn,resnet,lstm}/` (`best_model.pt`, `history.csv`,
`metrics.json`, `splits.json`),
`r010_model_comparison.json`, `figures/` (loss curves, F1 comparison, label
balance) and `verification_report.json` (independent verifier: corpus 11/11
checks PASS, training artifacts 8/8 PASS, 0 blocking issues).

## Dataset store (phases 1-3)

`datasets/store/lineage.sqlite3` is a local SQLite store that holds the frozen
corpora as versioned membership sets over **de-duplicated observations**: the six
CSV corpora contain 1,114,071 rows for only 375,348 unique observations (1,152
trajectories, 2.97x duplication), so the store keeps each observation once. It is
13.5 MB versus 348 MB of CSV corpora, with a 6.6 MB compressed backup under
`datasets/store/backups/`.

- **Lineage IDs:** a frozen corpus version is addressed as
  `<dataset_id>@v<version>` (e.g. `verified_merged_20261017@v1`, optionally plus
  `#<hash12>` content ref). Training runs record these IDs in `metrics.json`
  (phase 2, done 2026-10-17).
- **Two hashes per version:** a canonical content hash computed from the stored
  rows, and the legacy CSV sha256. Exports reproduce the historical CSV
  **byte-for-byte** (verified for all six corpora), so the integrity pins in
  `datasets/manifest.json` remain valid.
- **Immutable:** corrections create new versions; importing changed labels under
  an existing trajectory key is rejected.

Commands (from the repository root):

```powershell
python scripts\dataset_store.py build              # import new corpora (skips retired CSVs already in the store)
python scripts\dataset_store.py verify             # hashes, counts, byte-identical exports (retired state expected)
python scripts\dataset_store.py verify --require-legacy-csv  # strict pre-retirement mode
python scripts\dataset_store.py export --dataset verified_merged_20261017 --output out.csv
python scripts\dataset_store.py backup             # VACUUM INTO + gzip
```

Schema and conventions: `datasets/store/SCHEMA.md`. Design/phase plan:
`docs/PLAN_DATASET_STORE.md`.

**Phase status (2026-10-17):** phase 1 (build + verify), phase 2 (SQLite-direct
training + lineage IDs in run metadata) and phase 3 (retirement of the six
legacy corpus CSVs) are complete. Training reads the store via
`moma-trajectory-cleaning train --dataset-lineage verified_merged_20261017@v1`;
the legacy `--input <csv>` path remains available after restoring a corpus.
`datasets/` is now ~29 MB (was ~348 MB), and the store exports remain
byte-identical to every retired CSV and its integrity pin.

Duplicate finding from the store audit: `curated_20261017::20261017_FOV_0_CH_16`
is content-identical (observations and labels) to
`rdm_gluc_toggle_20260905::FOV_0_CH_16` - the renamed re-curation is an exact copy.
It is retained because the R009 batch is frozen; it contributes one duplicated
trajectory to the merged corpora.

## Usage

```powershell
# Train from the dataset store (lineage ids recorded in metrics.json; canonical
# split semantics of trajectory_cleaning.data.split_trajectory_keys)
moma-trajectory-cleaning train `
  --dataset-lineage verified_merged_20261017@v1 `
  --store datasets\store\lineage.sqlite3 `
  --require-human-verified `
  --source-column source_dataset `
  --output-dir runs\my_run

# Predict with the shipped production checkpoint
moma-trajectory-cleaning predict `
  --input my_trajectories.csv `
  --checkpoint src\moma_lineage_analysis\models\trajectory_cleaning_run\best_model.pt `
  --output predictions.csv
```