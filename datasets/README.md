# datasets/

Frozen, human-verified training corpora for the trajectory-cleaning models,
stored once in the local SQLite dataset store (`store/lineage.sqlite3`) since
phase 3. Integrity contract: per-file sha256 in `manifest.json` and per-version
canonical hashes in the store. R018 adds explicit user-authorized review
mutations: label edits or permanent trajectory deletion can turn a frozen
version into working data until it is finalized as a new dataset ID.

**R019 (2026-10-31):** the user reviewed the five batch corpora in the
training-review app, removed the three duplicate `verified_merged_*` dataset
versions from the store (they only re-listed trajectories already held by the
batch versions), finalized five `reviewed_20261031_*` training-ready versions
with the R004 membership filter, and retrained cnn/resnet/lstm on their
composition (**385,815 rows / 1,152 trajectories**). Observations and trajectory
rows were not changed by the removal; the pre-removal store is archived under
`store/backups/`. **R020 (2026-10-31):** the user authorized promotion of the
new CNN, so the production checkpoint is now the R019 CNN (sha256
`816a8b39…`); the previous R010 CNN is archived under
`models/trajectory_cleaning_run/promotion_backups/20261030T184217.243155Z/`.
See `runs/store_cleanup_20261031/`, `runs/reviewed_20261031/` and
`runs/reviewed_20261031/production_promotion.json`.

## Contents

| File | Role | Rows | Trajectories |
|---|---|---:|---:|
| `verified_20260916/verified_training_data_no_longruns.csv` | Production training corpus (round R004) | 312,317 | 1,053 |
| `verified_20261013/verified_training_data.csv` | Curated batch 2026-10-13 (round R006; merged in R010) | 22,890 | 35 |
| `verified_20261016/verified_training_data.csv` | Curated batch 2026-10-16 (round R007; merged in R010) | 31,384 | 48 |
| `verified_20261017/verified_training_data.csv` | Curated batch 2026-10-17 (round R009; merged in R010) | 8,757 | 16 |
| `verified_merged_20261016/verified_training_data.csv` | R008 retraining corpus (CSV retired; dataset version removed from the store in R019) | 365,283 | 1,134 |
| `verified_merged_20261017/verified_training_data.csv` | R010 retraining corpus (CSV retired; dataset version removed from the store in R019) | 373,440 | 1,149 |
| `verified_20261029@v1` | R014 curated batch 2026-10-29; merged in R015 and R019 | 25,728 | 48 |
| `verified_merged_20261029@v1` | R015 retraining corpus (CSV retired; dataset version removed from the store in R019) | 399,168 | 1,197 |
| `reviewed_20261031_*` (store only) | five R019 training-ready versions finalized from the reviewed batches; composition 385,815 rows / 1,152 trajectories | 385,815 | 1,152 |

**Phase 3 (2026-10-17): the six historical corpus CSVs are retired** (deleted after
user sign-off). The store holds the same rows (now reviewed working versions,
`frozen = 0`); each working export still matches its stored working hash, and
the pinned historical CSV sha256 applies to the pre-review frozen state. Every
retired file, its sha256/size and a restore command are recorded in
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
key is recoverable from `verified_20261017/rename_map.json`. The R009 raw inbox
was later rotated by the user; its durable QC report remains at
`verified_20261017/qc_report.json`.

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

The validation rule selected the **cnn**, and it was **promoted on 2026-10-17**
by user decision (round R011): it served as production until the R020 promotion
of the R019 CNN on 2026-10-31 (the R010 checkpoint
`6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b` is still
byte-identical to `runs/merged_20261017/cnn/best_model.pt` and is archived with
its promotion manifest under
`src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261030T184217.243155Z/`).
The R008 CNN remains in `promotion_backups/20261017T124420.980675Z/`. Artifacts:
`runs/merged_20261017/{cnn,resnet,lstm}/` (`best_model.pt`, `history.csv`,
`metrics.json`, `splits.json`),
`r010_model_comparison.json`, `figures/` (loss curves, F1 comparison, label
balance) and `verification_report.json` (independent verifier: corpus 11/11
checks PASS, training artifacts 8/8 PASS, 0 blocking issues).

## Batch 2026-10-29 (R014, `verified_20261029/`)

The rotated raw inbox contains 48 single-trajectory exports: 47 files of 540
rows and `FOV_140_CH_18` with 348 rows, totaling **25,728 observations**.
Technical QC passed all hard gates. The canonical reports are
`verified_20261029/qc_report.json`, `staging_report.json`, and
`runs/audit_20261029/preflight.json`; raw exports are retained unchanged.

Labels were staged with `--division-mode effective`: **2,562 divisions =
2,609 raw - 51 superseded + 4 added**, with 98 excluded observations and zero
division/exclusion overlap. Resolver statuses are 2,558 `unchanged`, 5
`coalesced_previous`, 4 `shifted_previous`, and 42 `needs_review`.
The exported effective boundaries reproduce the current packaged resolver
bitwise. The 42 `needs_review` raw sources are suppressed by the effective
policy and merge the flanking cycles; this computation is not new human
adjudication. The 348-row `FOV_140_CH_18` export ends earlier (17.8540 h) but
has no internal frame gaps.
The longest exclusion run is 3 frames, so no trajectory meets the R004
exclusion threshold (>= 5).

Eight reused channel identifiers receive a `20261029_` prefix in
`verified_20261029/rename_map.json`. These are naming collisions: biological
identity across acquisitions is unestablished. All eight identifier-collision pairs share zero time/length pairs (lengths
rounded to 8 decimal places), and no exact feature or effective-label payload
duplicate with history was found. Shared names do not establish biological
re-curation. The batch uses `curated_20261029::<post-rename key>`,
with the original key recoverable from the rename map.

Attestation: `user_attested_2026-10-29`, `human_verified: true`, recorded in
`verified_20261029/dataset_manifest.json`. The combined staged source is
`staged/combined_curated_20261029.csv` (sha256
`0f2aa9e035bea148031c0e22dafad584c306a5b56dd54908f54ba892887947ca`).
Store lineage `verified_20261029@v1#f3c99c422a3d` has canonical hash
`f3c99c422a3d5c668a40e1c6e3c36982067766581fd4e4203fc81d9aecdf60e2` and
CSV sha256 `95654104ae88a50a432064c5772c49e550a0fb6f6d969fbb2a74c0f72d8030e6`.
The redundant materialized corpus CSV was retired after independent data
verification; staged provenance and human-readable reports remain.

## Merged corpus 2026-10-29 and retraining (R015)

`verified_merged_20261029@v1` extends the frozen
`verified_merged_20261017@v1` membership with all 48 R014 trajectories. The
R004 rule drops none from the new batch. Result: **399,168 rows / 1,197
trajectories / 35,032 division / 2,022 exclusion positives**, with the same 295
legacy overlap rows and the previously documented identical-content pair
retained in the historical membership. The eight source counts are the seven
R010 counts above plus `curated_20261029` = 48.

The store-backed corpus verification passed 317 independent checks
(`runs/merged_20261029/verification_report.json`, data phase).
Provenance: `verified_merged_20261029/merge_manifest.json`; canonical hash
`8decab478c8037d93d412215fd338f1f8f12626b9979d897de9f112537549561`, CSV
sha256 `d8a288a1833858e95f5138d02f1473737d585c44d5cd7a6266b43c1fff9dd566`.
The packaged CLI trains CNN, ResNet, and BiLSTM directly from this store lineage
under `runs/merged_20261029/`, with unchanged defaults (seed 42, per-source
stratification, effective-label semantics for curated batches). The split is
838 train / 179 validation / 180 test trajectories. CNN wins the validation-only
rule (mean F1 0.8886); test scores never select the architecture.
`runs/merged_20261029/model_comparison.json` records the complete model table.
`cross_eval_vs_production.json` compares identical rows on a 103-trajectory
historical test panel (no content-hash overlap with old/new train/validation)
and a small seven-trajectory new-batch test panel. On the former, R015 CNN
exclusion F1/AP is 0.8099/0.8791 versus production 0.8232/0.8999 (lower).
On the latter, it is 0.4242/0.6273 versus 0.3750/0.4883 (higher, but only 13
exclusion positives). No significance test or bootstrap was run; these mixed
descriptive results do not support replacing production. Production remains the
R010 CNN; no R015 promotion is authorized.

## R016 temporal-context experiment (same frozen dataset, 2026-10-29)

R016 adds no observations, labels or store versions. It reuses
`verified_merged_20261029@v1` (canonical hash `8decab478c8037d93d412215fd338f1f8f12626b9979d897de9f112537549561`)
and its frozen seed-42 split: 838 train / 179 validation / 180 test trajectories.
The protocol is `runs/context_20261029/protocol.json`, registered before training.
This is a model-schedule experiment, not a new corpus or promotion.

Validation has 179 trajectories / 59,643 rows / 325 exclusions; 74 trajectories
(41.3%) are shorter than 197 frames. The fixed radius-98 common interior has
28,096 rows / 147 exclusions. Interior/boundary and trajectory-length strata
<197 versus >=197 are diagnostics, not replacements for all-row validation
selection. Supplementary validation diagnostics for all 29 runs are planned
after locking.

At fixed hidden width 64, eight dilation schedules are screened for CNN kernel5
(63,938 parameters) and primary ResNet kernel3 (100,354 parameters). Original
ResNet kernel5 RF121 (165,890 parameters) is a separate control. Nominal dense
context is in normalized-input frames, not a raw end-to-end window. Dilation
spacing/path multiplicities also change, so this is not pure RF causal inference.
Seed-42 shortlists are repeated on seeds43/44, then one schedule per family is
locked before test evaluation. All three seeds for each winner are reported.

**R016 results:** all 29 runs finished and the lock selected RF17 for both CNN
and primary ResNet (three-seed validation means 0.885582 and 0.887547). No
locked model beats the deployed R010 CNN on the descriptive test panels
(production exclusion AP 0.8960 full test / 0.8999 common panel). Independent
verification is PASS_WITH_DOCUMENTED_CAVEATS (21/21 checks; replay exact on the
original device, with one borderline threshold point and AP ties <=1.6e-4
across devices). Reusing seed42
from the shortlist in the final mean introduces selection optimism. Three-seed
spread measures optimization noise, not FOV or dataset uncertainty. Any winner
is only best among screened/shortlisted schedules; the irregular grid does not
support a universal or interpolated optimum. Previously reported test panels
are descriptive, not fresh external holdouts. Production remains R010 CNN;
width tuning is deferred.

## Reviewing and finalizing training data (R018, 2026-10-29)

The independent `moma-trainset-review` app (`src/moma_lineage_analysis/trainset_review/`)
reviews the SQLite store directly. Its default path is
`datasets/store/lineage.sqlite3`; `MOMA_TRAINSET_STORE` overrides it. Opening,
listing versions, loading trajectories and plotting are read-only. Review
extension tables are created only on the first mutation, so read-only opening
does not change the store.

**User-authorized mutations.** Label edits autosave immediately. Measurements
(`time_h`, `spine_length`) are immutable in this release. A permanent trajectory
delete removes the trajectory's observations, undo payloads and memberships from
every dataset version; there is no in-app restore. Deletion requires typing the
exact trajectory key and showing its impact. Each label edit, conflict resolution
and deletion takes an online SQLite backup first under `review_backups/` beside
the selected database, with path, sha256, bytes, global revision and audit event.

**Working versions.** A mutation clears the affected version's frozen flag and
legacy export pin, copies its original pins/counts into review state, and
recomputes a working canonical hash/counts/order in the live database. Training
resolution refuses a working or invalidated version. Therefore the original
frozen lineage or its CSV export must **not** be described as byte-identical
after user edits, even though its original pins and an archival backup are
retained.

**Finalization.** The reviewer chooses a new dataset ID, ticks human attestation
and finalizes. This creates a new frozen v1 membership with the R004 exclusion-run
filter applied to the new membership only (not a deletion), sets
`human_verified = 1`, records source-lineage provenance, and pins a byte-exact
native review export hash. The core store verifier accepts such a native version
only when its finalize audit event, provenance, canonical hash and export hash
match. Training must target this new frozen lineage.

Once a store contains reviewed working data, historical CSV build/import is
refused so a permanently deleted trajectory cannot be resurrected. Build a
separate historical store if an old corpus must be imported.

Two packaged checkpoints are selectable in the app: the default production R019
CNN and the R015 CNN (sha256
`9753298022805b4bc9f277d70e4da29aa8d88cac84a9575691477f332ff21687`, trained on
`verified_merged_20261029@v1`). Selecting R015 does not promote it. Evaluation
badges identify the checkpoint's recorded train/validation/test membership, or
`unseen`; agreement on training data is not a new held-out result.

The guide is [the training-review app guide](../docs/TRAINSET_REVIEW.md). At this
writing the full suite passes 388 tests with 4 skips and 59 subtests. Independent review is
PASS with zero open defects: backend 54, registry 25, scoped helper 86, AppTest
4 and opaque/cache 13 checks pass. Browser deletion confirmation, navigation,
finalize selection, stale-view guard and auto/CUDA device resolution were fixed
and re-tested; live store and production checkpoint hashes stayed unchanged.

## Dataset store (phases 1-3)

As of 2026-10-31 (R019), `datasets/store/lineage.sqlite3` holds **387,217
unique observations / 1,155 trajectory keys / ten versions / 2,307
trajectory-version memberships**: five reviewed working originals
(`frozen = 0`) and five review-finalized training-ready versions
(`reviewed_20261031_*`, `frozen = 1`). The three `verified_merged_*` dataset
versions were removed at the user's request on 2026-10-31 (review revision 562)
because every trajectory they held is also a member of a batch version; the
removal deleted only membership/metadata rows and left all observations and
trajectories intact. The pre-removal database is
`store/backups/lineage_20261030T181226.223866Z.sqlite3.gz`. The original six
R012 CSV corpora contained 1,114,071 rows for 375,348 unique observations (1,152
keys, 2.97x duplication); their 13.5 MB store versus 348 MB of CSV corpora is a
historical size comparison. Backups remain under `datasets/store/backups/`.

- **Lineage IDs:** a frozen corpus version is addressed as
  `<dataset_id>@v<version>` (e.g. `verified_merged_20261017@v1`, optionally plus
  `#<hash12>` content ref). Training runs record these IDs in `metrics.json`
  (phase 2, done 2026-10-17).
- **Two hashes per version:** a canonical content hash computed from the stored
  rows, and the legacy CSV sha256. Exports reproduce the historical CSV
  **byte-for-byte** (verified for all eight corpora), so the integrity pins in
  `datasets/manifest.json` remain valid.
- **Immutable:** corrections create new versions; importing changed labels under
  an existing trajectory key is rejected.

Commands (from the repository root):

```powershell
python scripts\dataset_store.py build              # import new corpora (skips retired CSVs already in the store)
python scripts\dataset_store.py verify             # hashes, counts, byte-identical exports (retired state expected)
python scripts\dataset_store.py verify --require-legacy-csv  # strict pre-retirement mode
python scripts\dataset_store.py export --dataset reviewed_20261031_verified_20261029 --output out.csv
python scripts\remove_dataset_versions.py --expected-revision N --reason "..." --lineage-id <id>  # admin duplicate removal
python scripts\dataset_store.py backup             # VACUUM INTO + gzip
```

Schema and conventions: `datasets/store/SCHEMA.md`. Design/phase plan:
`docs/PLAN_DATASET_STORE.md`.

**Phase status (2026-10-17):** phase 1 (build + verify), phase 2 (SQLite-direct
training + lineage IDs in run metadata) and phase 3 (retirement of the six
legacy corpus CSVs) are complete. Training reads the store via repeated
`--dataset-lineage` arguments (R019 composes the five `reviewed_20261031_*`
lineages); the legacy `--input <csv>` path remains available after restoring a
corpus.
The original R013 retirement reduced `datasets/` from ~348 MB to ~29 MB.
Current totals include the new raw/staged data and additional backups; the
store exports remain byte-identical to every retired CSV and its integrity pin.

The two new materialized corpus CSVs were retired on 2026-10-29 after
independent data verification (317 checks PASS). The retirement ledger now
contains eight records; `runs/audit_20261029/retirement_report.json` records
18 post-retirement checks PASS, including all canonical hashes and
byte-identical exports. Staged files and raw exports remain. Full regressions:
`runs/merged_20261029/pytest_full.txt` (230 passed / 4 skipped / 59 subtests).

Duplicate finding from the historical store audit: `curated_20261017::20261017_FOV_0_CH_16`
is content-identical (observations and labels) to
`rdm_gluc_toggle_20260905::FOV_0_CH_16` - the renamed re-curation is an exact copy.
A second identical pair appeared after the R019 review edits:
`curated_20261017::20261017_FOV_0_CH_8` equals
`rdm_gluc_toggle_20260905::FOV_0_CH_8`. Both pairs are retained (frozen batch
identity); the R019 composition collapsed zero observation rows because the
duplicates are different trajectory keys.

## Usage

```powershell
# Train from the dataset store (lineage ids recorded in metrics.json; canonical
# split semantics of trajectory_cleaning.data.split_trajectory_keys)
moma-trajectory-cleaning train `
  --dataset-lineage reviewed_20261031_verified_20260916_no_longruns@v1 `
  --dataset-lineage reviewed_20261031_verified_20261013@v1 `
  --dataset-lineage reviewed_20261031_verified_20261016@v1 `
  --dataset-lineage reviewed_20261031_verified_20261017@v1 `
  --dataset-lineage reviewed_20261031_verified_20261029@v1 `
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