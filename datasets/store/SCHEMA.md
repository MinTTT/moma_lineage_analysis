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

## Current contents (imported 2026-10-17)

| Lineage ID | Rows | Trajectories | Div / excl | Legacy CSV sha256 |
|---|---:|---:|---|---|
| verified_20260916_no_longruns@v1 | 312,317 | 1,053 | 27,415 / 1,355 | a2b3793d... |
| verified_20261013@v1 | 22,890 | 35 | 1,932 / 229 | 15721ab7... |
| verified_20261016@v1 | 31,384 | 48 | 2,698 / 300 | e4fced1a... |
| verified_20261017@v1 | 8,757 | 16 | 529 / 138 | 247da098... |
| verified_merged_20261016@v1 | 365,283 | 1,134 | 31,958 / 1,860 | 4e198bfb... |
| verified_merged_20261017@v1 | 373,440 | 1,149 | 32,470 / 1,924 | 7ecc9753... |

Schema version 2 (2026-10-17) corrects metadata defects found by the independent
phase-1 audit: `source_file` is unique per (filename, corpus) so all six corpora are
recorded, and corpus `kind`/`round_name` are explicit (production R004; batches
R006/R007/R009; merged R008/R010). Observations, labels, hashes and exports were
unchanged by the migration; a fresh rebuild is identical to the migrated database.

Store totals: **375,348 unique observations / 1,152 trajectories**; 3,435 membership
rows; DB 13.5 MB (versus 348 MB of CSV corpora). Duplicate audit: exactly one pair of
trajectory keys has identical content - `rdm_gluc_toggle_20260905::FOV_0_CH_16` and the
renamed re-curation `curated_20261017::20261017_FOV_0_CH_16` (see the R009 caveat in
`datasets/README.md`).

## Retired legacy CSVs (phase 3, 2026-10-17)

The six `verified_*/verified_training_data*.csv` files were deleted after user
sign-off; the store is the sole training source. Each version's export still
reproduces the retired file byte-for-byte and matches its stored
`legacy_csv_sha256`, so the integrity pins in `datasets/manifest.json` remain
valid.

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
- The six `datasets/verified_*/verified_training_data*.csv` files are retired but
  restorable byte-identically (see above).
- No training-run tables; run metadata references lineage IDs (phase 2).
