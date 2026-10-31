# Dataset store - schema and conventions (phases 1-3)

**Location:** `datasets/store/lineage.sqlite3` (SQLite, WAL mode, single-user local; **schema version 2**).
**Built by:** `python scripts/dataset_store.py build` (imports the corpora listed in `datasets/manifest.json`).
**Verified by:** `python scripts/dataset_store.py verify` (hash/count checks plus byte-identical export comparison).
**Backup:** `python scripts/dataset_store.py backup` -> `datasets/store/backups/lineage_<UTC>.sqlite3.gz`.

The store holds **datasets only**. Training runs keep their small metadata files
and record the dataset lineage IDs they used (phase 2); no run/evaluation/promotion
tables exist. Since phase 3 the store is the sole training source and the legacy
corpus CSVs are retired (restorable byte-identically, see below).

## Lineage ID

A frozen corpus version is addressed by:

    <dataset_id>@v<version>              lineage id, e.g. verified_merged_20261017@v1
    <dataset_id>@v<version>#<hash12>     content ref, e.g. verified_merged_20261017@v1#e3af02bd392a

`hash12` is the first 12 hex characters of `dataset_version.canonical_hash`.
Training metadata must record the lineage id (and optionally the content ref).

## Tables

| Table | Grain | Key columns |
|---|---|---|
| `schema_version` | one row per applied schema version | version, applied_at |
| `source_file` | one imported CSV per corpus | filename + corpus (UNIQUE together), sha256, rows |
| `trajectory` | one trajectory | trajectory_id, trajectory_key UNIQUE, source_dataset, source_trajectory_key, source_file, source_file_sha256, rename_of, label_policy, content_hash |
| `observation` | one cell observation (stored once) | (trajectory_id, fov_cell_index) PK; time_h, spine_length, division_point (0/1), excluded_point (0/1) |
| `dataset` | logical corpus | dataset_id PK, kind (production/batch/merged), label, round_name (R004/R006/R007/R008/R009/R010), verification_basis, human_verified |
| `dataset_version` | frozen snapshot of a corpus | version_id PK, (dataset_id, version) UNIQUE, canonical_hash, rows, trajectories, division_points, excluded_points, overlap_rows, legacy_csv_path, legacy_csv_sha256, frozen |
| `dataset_member` | membership plus row order | (version_id, trajectory_id) PK, ordinal UNIQUE per version |
| `ingest_provenance` | per-version provenance files | (version_id, path) PK, role, sha256 |

## Invariants

1. **Observations are stored once.** A trajectory key has exactly one content hash;
   importing the same key with different labels/features raises an error
   (labels are never rewritten in place - re-curations get new keys/versions).
2. **Versions are immutable** (frozen = 1); corrections create a new version.
3. **Two hashes per version:** `canonical_hash` is computed from the stored rows
   (trajectory_key + content hash); `legacy_csv_sha256` is the sha256 of the
   historical CSV. Exports reproduce that CSV **byte-for-byte** (verified for all
   six imported corpora), so the existing integrity pins stay valid.
4. **Row order is preserved** through `dataset_member.ordinal`; within a trajectory
   observations are ordered by `fov_cell_index` (the historical CSV order).
5. **`source_fov_cell_index` equals `fov_cell_index`** in all existing corpora and is
   reconstructed on export from `trajectory.source_*` plus the observation key.

## Current contents (updated 2026-10-31, after R019 cleanup)

The live database holds **five reviewed working versions and five
review-finalized training-ready versions**. The user reviewed/edited the five
batch versions through `moma-trainset-review` (labels edited in place; 45
trajectories permanently deleted across the store), so the originals are
`frozen = 0` working versions with their original pins preserved in
`review_version_state`.

The three `verified_merged_*` dataset versions were **removed from the store**
on 2026-10-31 at the user's request (revision 562): every trajectory they held
is also a member of a batch version, so the removal deleted only 3,345
membership rows, three version rows, their provenance and review-state rows -
observations and trajectory rows were unchanged (387,217 / 1,155 before and
after). The pre-removal database is archived as
`backups/lineage_20261030T181226.223866Z.sqlite3.gz` (sha256
`bad95fa07a659036cc70defb539a107f9ec7108c249d73e39adf727063e104f7`) and in the
review snapshot
`review_backups/lineage_review_20261030T181234.830951Z_r561_remove_dataset_versions.sqlite3`.
Historical R008/R010/R015 lineage ids therefore no longer resolve against the
live store; their CSV hashes remain pinned in `datasets/manifest.json`
(`store_removed` markers) and in `retired_csv_manifest_20261017.json`.

| Lineage ID | State | Rows | Trajectories | Div / excl |
|---|---|---:|---:|---|
| verified_20260916_no_longruns@v1 | working | 305,651 | 1,019 | 26,649 / 1,147 |
| verified_20261013@v1 | working | 20,928 | 32 | 1,760 / 195 |
| verified_20261016@v1 | working | 26,153 | 40 | 2,239 / 245 |
| verified_20261017@v1 | working | 8,757 | 16 | 528 / 142 |
| verified_20261029@v1 | working | 25,728 | 48 | 2,562 / 102 |
| reviewed_20261031_verified_20260916_no_longruns@v1 | frozen training-ready | 305,503 | 1,018 | 26,639 / 1,138 |
| reviewed_20261031_verified_20261013@v1 | frozen training-ready | 20,928 | 32 | 1,760 / 195 |
| reviewed_20261031_verified_20261016@v1 | frozen training-ready | 25,499 | 39 | 2,192 / 237 |
| reviewed_20261031_verified_20261017@v1 | frozen training-ready | 8,157 | 15 | 512 / 68 |
| reviewed_20261031_verified_20261029@v1 | frozen training-ready | 25,728 | 48 | 2,562 / 102 |

The five finalized versions apply the R004 long-exclusion-run filter to their
new membership only and drop three trajectories
(`baseline_0_10h::FOV_28_CH_2`, `curated_20261016::FOV_53_CH_11`,
`curated_20261017::FOV_14_CH_14`); nothing is deleted from the store. Their
composition is the R019 training corpus: **385,815 rows / 1,152 trajectories /
33,665 division points / 1,740 exclusion points / 49 legacy overlap rows**, eight
sources, zero duplicate observation rows.

Schema version 2 (2026-10-17) corrects metadata defects found by the independent
phase-1 audit: `source_file` is unique per (filename, corpus) so all six corpora are
recorded, and corpus `kind`/`round_name` are explicit (production R004; batches
R006/R007/R009; merged R008/R010). Observations, labels, hashes and exports were
unchanged by the migration; a fresh rebuild is identical to the migrated database.

Store totals (2026-10-31): **387,217 unique observations / 1,155 trajectories**;
2,307 membership rows / 23 ingest-provenance records / ten dataset versions;
review revision 567. Original R012 totals (375,348 observations / 1,152
trajectories) and the pre-review eight-version totals (401,076 / 1,200) are
historical. Duplicate audit: two content-identical trajectory pairs remain -
`rdm_gluc_toggle_20260905::FOV_0_CH_16` / `curated_20261017::20261017_FOV_0_CH_16`
and `rdm_gluc_toggle_20260905::FOV_0_CH_8` / `curated_20261017::20261017_FOV_0_CH_8`
(review edits made the second pair identical; the first is the documented R009
re-curation). Both pairs straddle R019 partitions (CH_16 validation/train;
CH_8 test/validation), so one test and one validation trajectory have an exact
content twin elsewhere - an explicit leakage caveat for the R019 metrics. After
the version removal, `source_file` also keeps three historical merged-CSV
import records (filename/sha256/rows only); no dataset, dataset_version,
dataset_member, ingest_provenance or review-state rows remain for those
corpora.

## Retired legacy CSVs (phase 3, 2026-10-17)

The six original `verified_*/verified_training_data*.csv` files were deleted
after user sign-off on 2026-10-17. On 2026-10-29, two new combined corpus CSV
materializations were also retired after 317 independent data checks passed.
The store remains the sole training source for the five batch lineages; the
retirement ledger retains all eight export hashes and restore commands.
The three merged CSV materializations can no longer be re-exported from the
live store after the 2026-10-31 removal; restore them from the pre-removal
backup if a historical rebuild is ever required. Each remaining version's
export still reproduces the retired file byte-for-byte and matches its stored
`legacy_csv_sha256`, so the integrity pins in `datasets/manifest.json` remain
valid for the retained corpora.

- `verify` treats a missing legacy CSV as expected (`legacy_csv_state: retired`)
  and still compares the export sha256 to the stored pin.
- `verify --require-legacy-csv` fails while a CSV is retired (strict
  pre-retirement mode) and passes again after a corpus is restored.
- `retired_csv_manifest_20261017.json` records each retired file's path, bytes,
  sha256, lineage id, content ref, and a `dataset_store.py export` restore
  command.
## Boundaries of the current state (phases 1-3)

- The training CLI reads the store directly (`--dataset-lineage`); the CSV path is
  available only after restoring a retired corpus from the store.
- All eight combined `datasets/verified_*/verified_training_data*.csv` files are retired;
  the five batch corpora remain restorable byte-identically from the live store,
  and the three merged corpora are restorable from the pre-removal backup.
- The live store now also holds five review-finalized training-ready versions;
  `verify` reports the working originals as `WORKING` and the finalized ones as
  `PASS` (`legacy_csv_state: database_native`).
- No training-run tables; run metadata references lineage IDs (phase 2).
