# Plan: local SQLite dataset store

Status: **phases 1-3 complete** (2026-10-17): the store is built/verified,
training reads it directly with lineage IDs recorded, and the six legacy corpus
CSVs were retired after user sign-off (restorable byte-identically from the
store). `datasets/` dropped from ~348 MB to ~29 MB.
Owner decisions are recorded in section 2. Implementation lives in
`src/moma_lineage_analysis/trajectory_cleaning/dataset_store.py` and
`scripts/dataset_store.py`; the schema reference is `datasets/store/SCHEMA.md`.

## 1. Problem and measured baseline

The frozen corpora shared observations massively: 6 corpora contained 1,114,071
CSV rows for only 375,348 unique observations (1,152 trajectories) - a 2.97x
duplication - because each merged corpus re-materialised its constituents. The
`datasets/` tree was 348 MB; `runs/` (all training evidence) was only 5.5 MB.
None of the duplicated observations disagreed on labels or features, so storing
each observation once is safe.

## 2. User decisions (approved 2026-10-17)

1. Database location: `datasets/store/lineage.sqlite3`.
2. Training reads SQLite **directly**; run metadata records the dataset lineage
   IDs used for training.
3. Legacy `verified_*/` CSV corpora are deleted **after phase-2 sign-off**
   (phase 3; not before).
4. Only **datasets** go into the database (no run/evaluation/promotion tables).
5. Human-readable provenance stays on disk (`datasets/verified_*/` JSON reports,
   `datasets/README.md`, `datasets/manifest.json`).
6. Single-user local only (one SQLite file, WAL, no server).

## 3. Architecture

- One SQLite file is the source of truth: observations stored once, keyed by
  `(trajectory_id, fov_cell_index)`.
- A corpus is a **frozen dataset version** = membership set of trajectories
  (`dataset_member`), not a copy of rows.
- Every version keeps two hashes: `canonical_hash` (computed from stored rows) and
  `legacy_csv_sha256` (the historical integrity pin).
- Exports are deterministic and **byte-identical** to the historical CSV, so the
  store can reproduce training inputs exactly.
- Lineage ID = `<dataset_id>@v<version>` (optionally plus `#<hash12>` content ref).

Data model, tables and invariants: see `datasets/store/SCHEMA.md`.

## 4. File organisation

```text
datasets/
  store/
    lineage.sqlite3                  # source of truth (WAL)
    lineage.sqlite3-wal / -shm
    backups/lineage_<UTC>.sqlite3.gz # after every ingest + periodic
    SCHEMA.md                        # schema/conventions (human-readable)
  verified_*/                        # corpus provenance JSONs + staged files
                                     # (pinned corpus CSVs retired in phase 3)
  manifest.json                      # integrity pins (unchanged, human-readable)
  README.md                          # corpus catalogue + store section
raw_train_data/                      # curator inbox (unchanged)
runs/<run>/                          # training evidence; metadata records lineage IDs (phase 2)
```

Exports are on-demand (e.g. `datasets/store/exports/`); they duplicate CSV bytes
and are safe to delete, so verification exports are removed after each verify run.

## 5. Workflows (target state)

A. **Ingest a curated batch** - unchanged audit pipeline (QC -> staging ->
   attested corpus). The importer (phase 1) then stores new trajectories and
   creates a new `dataset_version`; content conflicts abort the import.
B. **Compose a training set** - choose dataset versions (e.g. R004 + batches)
   and the R004 filter policy; record the resulting lineage IDs.
C. **Train** - the CLI reads SQLite directly (phase 2, implemented), materialises the split
   deterministically, and writes `train_set` lineage IDs + content hashes into
   `metrics.json` (no CSV copies in run directories).
D. **Evaluate/compare** - evaluations record which lineage IDs were used.
E. **Audit** - `verify` recomputes hashes/counts and re-derives exports;
   `backup` snapshots with VACUUM INTO + gzip.

## 6. Phases

| Phase | Scope | Status |
|---|---|---|
| 0. Design sign-off | this plan + user decisions | done 2026-10-17 |
| 1. Build + import + verify | schema, importer, exporter, verifier, backup; training path untouched | **done 2026-10-17** |
| 2. Training integration | read SQLite directly; record lineage IDs in run metadata; reproduce R010 (same split keys and metrics) | **done 2026-10-17** |
| 3. Retire CSV corpora | delete `datasets/verified_*/verified_training_data*.csv` after phase-2 sign-off; keep provenance JSONs and hashes in `datasets/manifest.json` | **done 2026-10-17** |
| 4. Hardening | immutability tests, migration policy, scheduled backups/restore drill | pending |

## 7. Phase 1 verification evidence (2026-10-17)

- Database: `datasets/store/lineage.sqlite3`, 13.5 MB, `PRAGMA integrity_check` = ok.
- Imported 6 dataset versions (all `v1`); 375,348 unique observations / 1,152
  trajectories; 3,435 membership rows; 16 provenance-file records.
- Per version: row / trajectory / label counts and canonical hash recomputed and
  matched; legacy CSV sha256 matched the file on disk; **export byte-identical to
  the legacy CSV** (all six sha256 values equal the pins in `datasets/manifest.json`).
- Backup: `datasets/store/backups/lineage_20261017T132240.874489Z.sqlite3.gz`
  (6.6 MB, sha256 `a2d0c12d35f39de634ca49981ec2770ef9385df65b68baf597cc3b17f8b1585b`).
- Rebuild determinism: rebuilding into a fresh database produced identical
  version records (canonical hashes, legacy shas, counts, membership) and
  identical store totals, so the import is reproducible and `build` is idempotent.
- Post-audit metadata corrections (independent verifier D1-D3, 2026-10-17):
  schema version 2 makes `source_file` unique per (filename, corpus) so all six
  corpora are recorded (previously 2 due to a filename-only unique key), and
  corpus kind/round metadata is explicit (previously inferred by string sniffing and
  wrong/NULL for batches and merged corpora). Neither defect affected observations,
  labels, hashes, exports or memberships; an in-place migration produced a database
  identical to a fresh build (versions/members/source files/totals compared).
- Phase 3 (retirement, 2026-10-17): user sign-off after the phase-2 report; a
  strict store verification passed while all six CSVs were present
  (`verify --require-legacy-csv`), then only the six pinned corpus CSVs were
  deleted, then normal verification passed with every version reported as
  `legacy_csv_state: retired` and each export still matching its stored legacy
  sha256. `datasets/store/retired_csv_manifest_20261017.json` records every
  retired file (path, bytes, sha256, lineage id) and its
  `scripts/dataset_store.py export` restore command. Provenance JSONs, corpus
  manifests, staged provenance, `datasets/manifest.json`, the store and its
  backup remain.
- Phase 2 (training integration, 2026-10-17): the training CLI accepts
  `--dataset-lineage <id>` (repeatable, read-only store access) and records a
  `dataset_lineage` block in `metrics.json` (lineage ids, content refs, canonical
  hashes, legacy CSV sha256, composition and trajectory-key digest). A
  store-loaded frame is dtype-exact identical to the legacy CSV read and yields
  the same split keys; R010 was reproduced from `verified_merged_20261017@v1`
  (`runs/merged_20261017_store_repro/reproduction_report.json`):
  resnet/lstm bit-identical (weights and all metrics), cnn identical thresholds /
  discrete metrics / split bytes with only average-precision drift <=8.95e-5 that a
  CSV-only control run reproduces at the same magnitude (GPU nondeterminism).
- Finding: exactly one pair of trajectory keys shares identical content
  (`rdm_gluc_toggle_20260905::FOV_0_CH_16` vs the renamed R009 re-curation
  `curated_20261017::20261017_FOV_0_CH_16`); recorded as a data caveat, not silently
  collapsed.

## 8. Risks and mitigations

- **SQLite single writer / local file only** - one process writes, WAL readers,
  no network share.
- **Corruption** - backups after ingest, `integrity_check` in `verify`, retired CSVs
  restorable from the store, raw inbox retained.
- **Label edits** - rejected at import; re-curations must create new trajectory versions.
- **Legacy pin drift** - both hashes stored; exports proven byte-identical.
- **Governance** - frozen versions, independent verification each round,
  contract/report updates (AGENTS.md phase note).

## 9. Phase 2 (done 2026-10-17) and phase 3

1. `moma-trajectory-cleaning train --dataset-lineage <id>` reads the store
   read-only; `--store` defaults to `datasets/store/lineage.sqlite3`; the flag is
   repeatable to compose versions (duplicate observations are collapsed only
   after proving they are identical). `--input <csv> --dataset-manifest ...`
   remains as the legacy path.
2. `metrics.json` (and the checkpoint metadata) gain a `dataset_lineage` block:
   store path, schema version, per-lineage id/content-ref/canonical hash/legacy
   CSV sha256/counts, composition rows/trajectories, and a deterministic
   `trajectory_keys_sha256`. `training_data.source` distinguishes
   `sqlite_dataset_store` from `csv`.
3. R010 was reproduced from `verified_merged_20261017@v1` in
   `runs/merged_20261017_store_repro/` with a byte-identical `splits.json` for all
   three models and identical thresholds/discrete metrics. ResNet and BiLSTM
   reproduced bit-identically; the CNN differs only in average precision
   (<=8.95e-5), which a CSV-only control run reproduces at the same magnitude
   (GPU nondeterminism), so the store data path is exact.
4. Phase 3 **(done 2026-10-17)**: the six legacy `verified_*/verified_training_data*.csv`
   files were deleted after sign-off; provenance JSONs and `datasets/manifest.json`
   were kept, and the docs record the store as the sole training source. Restore
   any corpus with the `restore_command` in
   `datasets/store/retired_csv_manifest_20261017.json` and re-run
   `verify --require-legacy-csv`.
