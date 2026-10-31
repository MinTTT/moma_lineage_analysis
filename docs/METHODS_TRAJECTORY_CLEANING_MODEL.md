# Methods: trajectory cleaning model

| Contract field | Value |
|---|---|
| Research task | trajectory cleaning model |
| Analysis ID | `trajectory_cleaning_model` |
| Method version | `0.11.1` |
| Status | Reconciled (R020: user-authorized promotion of the R019 CNN to production; new checkpoint sha256 `816a8b39...`, previous R010 CNN backed up; R019 cleanup/retraining independently verified PASS 29/29 checks) |
| Created | 2026-09-16 |
| Last reconciled | 2026-10-31 |
| Project root | `G:/python_code/moma_lineage_analysis` |
| Contract owner | Research team |
| Prior method version | `0.11.0` |

This file is the durable scientific contract for this research task. The active Methods prose, controlled registries, executable code, actual run manifest, and outputs must be reconciled after every analysis round.

## Layer 1 — Active Methods

### Research question and scope

Can a sequence model that sees only `time_h` and `spine_length` reproduce human
trajectory-cleaning decisions for mother-machine data? Every observation in a
single-channel lineage trajectory receives two independent multi-label outputs:

1. `division_point` — the observation is the mother frame immediately before a
   cell-division event.
2. `excluded_point` — the observation is noise (segmentation or drift artifact)
   and should be excluded from downstream growth-rate analysis.

Deliverables: a trained checkpoint consumable by the Streamlit lineage app
(`Moma_lineage_streamlit.py`), calibrated per-label thresholds, and held-out
validation/test metrics. Out of scope: using fluorescence, morphology beyond
spine length, or inter-channel information; unattended deletion of data (the
model only proposes labels for human review).

### Study design and provenance

Training data are curated exports of the Streamlit lineage app, combined and
attested by `trajectory_cleaning/prepare_verified_training_data.py`. The
historical R002 training corpus is
`verified_20260916/verified_training_data.csv`
(SHA-256 `deff25a96a7f572b86b0ce7a033b3c4744098d0e73462d44d3db896810c23cba`,
318,745 rows, 1,070 trajectories; archived in the legacy workspace
`Moma_cellPose/trajectory_cleaning_run/`, not shipped in this package) with
provenance recorded in
`datasets/verified_20260916/dataset_manifest.json`
(schema 1.0, `human_verified: true`). Source datasets: expanded baseline
0-10 h, retained old-only 0-8 h subset, a starvation condition, and the
21.6 h RDM-Gluc Toggle-L3 experiment (560 trajectories, 432 frames each,
~3 min sampling). An explicit supersession rule (`baseline_0_8h` ->
`baseline_0_10h`) omits 83 duplicated trajectories (9,935 rows). Colliding raw
trajectory keys are namespaced as `<source_dataset>::<source_trajectory_key>`.
Upstream processing: Cellpose segmentation and `Moma_CPM_FOV.py`
cell-parameter extraction; only canonical final human labels are targets — AI
predictions and audit fields are stripped. For the RDM-Gluc source, the audit
label `effective_division_point` was reconciled into `division_point` before
staging (R002 label policy): rows flagged `coalesced_previous` (98) have their
`division_point` label dropped because the event is already labeled at the
previous time point; 26 `effective_division_point`-only rows without a
resolution flag are kept as divisions; `needs_review` (145) and
`shifted_previous` (26) rows keep their original `division_point`. Points
labeled both division and excluded remain excluded. The reconciled staging file
is `verified_20260916/rdm_gluc_toggle_20260905_resolved_labels.csv` (archived in
the legacy workspace, not shipped in this package). The R001 corpus
`verified_20260718/verified_training_data.csv` remains
as the historical reference (also archived in the legacy workspace). The R004
production corpus additionally removes
every trajectory whose longest contiguous exclusion run is 5 frames or more
(17 of 1,070 trajectories, 6,428 rows) because those runs encode analysis-driven
removal of whole cell cycles rather than growth noise; the filtered corpus is
`datasets/verified_20260916/verified_training_data_no_longruns.csv`
(SHA-256 `a2b3793d2138d806f5d5a85c3f51e15b9629953b4c0ef54b9490b74f99bc0092`,
312,317 rows, 1,053 trajectories) with fixed splits in `splits_no_longruns.json`
and the removed keys in `longrun_excluded_trajectories.json`.

Round R006 (2026-10-13) adds a new curated batch,
`datasets/verified_20261013/verified_training_data.csv` (22,890 rows, 35
trajectories, SHA-256 `15721ab757e956fc5c8a79855f67ce5713cc5a619a1b88ed2a80404b4c2870b3`),
built from 35 single-trajectory app exports under `datasets/raw_train_data/`.
The exports passed technical QC (`scripts/qc_raw_train_data.py`). Two user
decisions shaped the final build: (1) `effective_division_point` is declared
the final true division label, so labels were staged with
`scripts/stage_verified_labels.py --division-mode effective` — division_point
equals the resolver's effective boundary (1,932 divisions; 22 raw labels at
coalesced/shifted/needs_review sources superseded, 6 effective-only boundaries
added; division/exclusion overlap is 0 by construction); (2) 11 keys that
duplicate trajectories already present in the R004 production corpus were
renamed with a `20261013_` prefix (`datasets/verified_20261013/rename_map.json`),
so the batch is duplication-free while the date prefix preserves provenance.
The batch is namespaced `curated_20261013::<key>` and is **not yet merged**
into the R004 production corpus;
`curated_20261013::FOV_121_CH_7` carries a 5-frame contiguous exclusion run that
the R004 filter would drop. Note: the effective-label semantics of this batch
differ from the historical R002 staging policy used for the RDM-Gluc source
(which kept raw labels at shifted/needs_review excluded sources); any future
merge must record this label-policy difference.

Round R007 (2026-10-16) adds a second curated batch,
`datasets/verified_20261016/verified_training_data.csv` (31,384 rows, 48
trajectories, SHA-256 `e4fced1a3958840954e334ac49acfdeeb0d6f4b2602a35760154e5f2ea13cf76`),
from the rotated contents of `datasets/raw_train_data/` (the 35 previous raw
exports were removed by the user; the R006 batch's QC report is archived at
`verified_20261013/qc_report.json`). The same pipeline was applied: technical
QC (all 48 files pass), `effective_division_point` as the division label
(2,698 divisions; 37 raw labels superseded, 7 effective-only added;
division/exclusion overlap 0), and a `20261016_` date prefix for 18 keys that
duplicate R004-corpus trajectories
(`datasets/verified_20261016/rename_map.json`). The batch is namespaced
`curated_20261016::<key>`, is duplication-free against both existing corpora,
and is **not yet merged**; `curated_20261016::FOV_53_CH_11` carries a 5+-frame
exclusion run that the R004 filter would drop.

Round R008 (2026-10-16) merges the R004 production corpus with both curated
batches into `datasets/verified_merged_20261016/verified_training_data.csv`
(365,283 rows, 1,134 trajectories, SHA-256
`4e198bfb85ab35da2694a888c7fa2ce67c5b5ac338a0596d91796162ea407613`;
`scripts/merge_corpora.py`). The R004 long-exclusion-run filter was applied to
the two batches, dropping exactly `curated_20261013::FOV_121_CH_7` and
`curated_20261016::FOV_53_CH_11`. The merged corpus mixes two label policies:
R004-corpus rows follow the historical R002 staging (295 division/exclusion
overlap rows), batch rows follow effective-label semantics (overlap 0); this is
a documented, accepted difference. Retraining used the unchanged default
configuration (seed 42, 100 epochs, patience 15, batch 16, per-source
stratified 70/15/15 split = 793/170/171 with every source in every partition)
via the packaged CLI on GPU. Validation-only selection (mean of division-event
F1 and exclusion F1) selected the **BiLSTM** (0.8987) over CNN (0.8915) and
ResNet (0.8907). The user then overrode the rule and promoted the **CNN**
(rationale recorded in `promotion.json`: best test exclusion F1 0.8452, tail
recall 0.9959, smallest and fastest model). The R008 promotion made this the production checkpoint (SHA-256
`3e796c9efbf1946bb5d371808d7faabcbf148851427e79c20ed1dc2068db5a2c`) until the
R011 promotion of 2026-10-17; the previous R004 CNN is backed up under
`promotion_backups/`. Test metrics are
evaluated on the new merged split and are not paired-comparable with
R002/R004-era test splits.

Round R009 (2026-10-17) adds a third curated batch,
`datasets/verified_20261017/verified_training_data.csv` (8,757 rows, 16
trajectories, SHA-256
`247da098e97d060c4c24696ef4ed78862a0ef3932b717492d21f1c8219d8ad67`), built from
16 newly added single-trajectory exports under `datasets/raw_train_data/`
(still in place; five 432-frame / 21.6 h files and eleven 600-frame / 30.1 h
files, two of them complete-duration exports of 599 and 598 rows). Technical
QC (`scripts/qc_raw_train_data.py --corpus
datasets/verified_merged_20261016/verified_training_data.csv`) passed all 16
files with 0 hard-gate failures (553 raw division / 138 excluded labels) and
flagged 25 rows where `division_point != effective_division_point` plus one
`effective_division_point`-only row. Nine keys exactly duplicating R004-corpus
trajectories were renamed with a `20261017_` prefix
(`datasets/verified_20261017/rename_map.json`); exact-key overlap after
renaming is NONE, no other key duplicated an existing corpus, exact or
prefix-normalized, and the normalized overlap with R004 is exactly the 9
documented re-curations. Labels were staged with
`scripts/stage_verified_labels.py --division-mode effective --rename-map`
(report: `datasets/verified_20261017/staging_report.json`): 529 divisions =
553 raw - 25 superseded + 1 effective-only, resolver statuses 528 unchanged, 3
`coalesced_previous`, 1 `shifted_previous`, 21 `needs_review`; division and
exclusion overlap is 0. The export's `effective_division_point` column was
independently verified to reproduce the packaged resolver bitwise for all 16
trajectories, and staging reproduces those labels exactly (0 mismatches, 0
rows missing); features agree with the raw exports to <= 3e-15 relative
(measured max 2.01e-15; a pandas float-parser artifact, not data corruption)
with labels and row membership exact. The
batch is namespaced `curated_20261017::<key>` under `dataset_id
curated_20261017` (`verification_basis user_attested_2026-10-17`) via
`prepare_verified_training_data.py` and is **not yet merged** into the
R008-era production corpus.

Audited caveats, recorded but none blocking:
`curated_20261017::FOV_14_CH_14` carries a 16-frame contiguous exclusion run
(74 excluded rows; rows 569-584) and would be dropped by the R004 filter at
merge time — no other trajectory has a run of 5 or more frames (next longest
4); the 13 `needs_review` supersessions sit in
`curated_20261017::20261017_FOV_13_CH_19`, which keeps 23 divisions and a
longest run of 4; four trajectories retain only 1-2 divisions (`FOV_14_CH_1`
1; `FOV_14_CH_3`, `FOV_15_CH_23`, `FOV_15_CH_2` 2 each) after spine-drop
checks confirmed genuinely non-dividing or arrested tracks, not
under-labeling; the 600-frame / 30.1 h grid is a new movie length with no
precedent in the pre-batch corpus (longest existing trajectory 28.3 h);
`FOV_12_CH_17` (599 rows) and `FOV_15_CH_2` (598 rows) are complete-duration
exports with internal missing frames, not truncated files; and `FOV_10_CH_17`
has one effective division at row 568 (28.55 h) with a daughter/mother spine
ratio of 0.256 versus a batch median of 0.586 (5th percentile 0.535), flagged
for review only. The canonical `split_trajectory_keys` split (seed 42) is
feasible only at 12/2/2 trajectories: train 6,861 rows (386 division / 60
excluded positives), validation 1,032 (61 / 75), test 864 (82 / 3). With 16
trajectories the batch is below a standalone training population, so it is
additive and intended for merging into the production corpus, not standalone
training.

Round R010 (2026-10-17) merges all four verified sources with
`scripts/merge_corpora.py` into
`datasets/verified_merged_20261017/verified_training_data.csv` (373,440 rows,
1,149 trajectories, 32,470 `division_point` / 1,924 `excluded_point`; SHA-256
`7ecc9753cf02c0dfc020fd70f6b077f997799d46d4f3af3fd89d12a16eaee2bc`): the
pre-filtered R004 corpus (`--corpus`) plus the three curated batches
(`--filtered-corpus`). The R004 filter dropped exactly
`curated_20261013::FOV_121_CH_7`, `curated_20261016::FOV_53_CH_11` and
`curated_20261017::FOV_14_CH_14`; retained batch rows are 22,236 / 30,730 /
8,157 and the longest exclusion run over all 1,149 trajectories is 4 frames.
Seven source datasets contribute trajectories: baseline_0_10h 477,
baseline_0_8h 19, rdm_gluc_toggle_20260905 546, starvation_20251111 11,
curated_20261013 34, curated_20261016 47, curated_20261017 15. The 295
division/exclusion overlap rows are all legacy R002-policy residue inside the
frozen R004 corpus (baseline_0_10h 161, baseline_0_8h 8,
rdm_gluc_toggle_20260905 126, starvation 0); the three curated batches
contribute 0. Provenance is recorded in
`datasets/verified_merged_20261017/merge_manifest.json` (SHA-256
`94a54d9169ac66e81dde8728222fa7b067078ef8cd302ffaae967c96c4df6b6d`,
`human_verified: true`, `verification_basis: constituent_corpora_attested`).
The per-source stratified seed-42 split is 804 train / 172 validation / 173
test trajectories (train 261,377 rows / 22,689 division / 1,304 excluded
positives; validation 55,866 / 4,842 / 314; test 56,197 / 4,939 / 306) with
every source in every partition.

Round R010 retrained all three architectures with the unchanged default
configuration (seed 42, max 100 epochs, patience 15, batch 16) via the packaged
CLI with `--require-human-verified --dataset-manifest
datasets/verified_merged_20261017/merge_manifest.json --source-column
source_dataset` on GPU. Validation-only selection (mean of division-event F1
and exclusion F1) selected the **cnn** (0.8987) over ResNet (0.8929) and
BiLSTM (0.8642). Validation division-event F1 is 0.9953 / 0.9962 / 0.9946 and
validation exclusion F1 0.8020 / 0.7896 / 0.7339 (cnn/resnet/lstm); test
division-event F1 0.9957 / 0.9964 / 0.9933, test exclusion F1 0.8512 / 0.7973
/ 0.7993, test tail recall 0.9949 / 0.9929 / 0.9939 and last-event miss rate
0.0058 / 0.0058 / 0.0116. Training took 43 / 25 / 25 epochs (0.61 / 0.55 /
10.27 min). **No promotion was performed at round close**: unlike R008 the validation
rule was not overridden, so the production checkpoint stayed unchanged and
promotion remained a user decision; the user then promoted the R010 CNN on
2026-10-17 (round R011). An independent verifier re-checked the frozen corpus
(11/11 checks) and the training artifacts (8/8 checks), confirmed the splits
are identical to the canonical recomputation, that all checkpoints load, and
that the source hashes match the merge manifest
(`runs/merged_20261017/verification_report.json`, all PASS).

Round R011 (2026-10-17) is a deployment round. The user promoted the R010 CNN
to production with
`python -m moma_lineage_analysis.trajectory_cleaning.promote_best_model
--run-dir runs/merged_20261017 --destination-dir
src/moma_lineage_analysis/models/trajectory_cleaning_run`; the tool re-applies
the validation-only selection rule (cnn 0.898668 > resnet 0.892905 > lstm
0.864244; test metrics are never read), copies the winner's four artifacts
(`best_model.pt`, `history.csv`, `metrics.json`, `splits.json`) with a
hash-verified copy, and rewrites `promotion.json` (`promoted_at_utc
2026-10-17T12:44:20.980675Z`). The new production checkpoint
`src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt` has
SHA-256 `6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`
and is byte-identical to `runs/merged_20261017/cnn/best_model.pt`; its
`metrics.json` metadata still points at the R010 corpus (sha256 `7ecc9753…`,
373,440 rows / 1,149 trajectories) and the promoted thresholds are 0.68 /
0.89. The promoted metrics are the R010 CNN metrics: validation division-event
F1 0.9953 and exclusion F1 0.8020; test division-event F1 0.9957, exclusion F1
0.8512, division-point F1 0.9940, tail recall 0.9949 and last-event miss rate
0.0058 (43 epochs). The previous R008 CNN (SHA-256
`3e796c9efbf1946bb5d371808d7faabcbf148851427e79c20ed1dc2068db5a2c`) and its
histories/metrics/splits are backed up under
`src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261017T124420.980675Z`.
The promotion tool backs up the four model artifacts but replaces
`promotion.json` without archiving it, so the R008 promotion manifest is no
longer in that file; the R008 decision remains recorded in this contract (R008
round-registry row, R008 reconciliation row and 0.4.0 change-log row) and in
`runs/merged_20261016/r008_model_comparison.json`, and the R004-era
`promotion.json` survives in the earlier `20261016T145403.716334Z` backup. No
model, metric, parameter, or corpus changed; only the deployed checkpoint and
the app's default model label changed.

Round R012 (2026-10-17) changes only the training data layer. The frozen
corpora are now stored once in the local SQLite dataset store
`datasets/store/lineage.sqlite3` (schema version 2; 375,348 de-duplicated
observations / 1,152 trajectories across six dataset versions, 13.5 MB versus
348 MB of CSV corpora). Each corpus version keeps a canonical content hash, a
legacy CSV sha256 and its byte-identical CSV export; observations are stored
once and versions are immutable (`datasets/store/SCHEMA.md`). Training reads the
store read-only by dataset lineage id (`--dataset-lineage
verified_merged_20261017@v1`), records the lineage ids, content refs, canonical
hashes and legacy CSV sha256 in `metrics.json` as `dataset_lineage`, and keeps
the legacy `--input <csv> --dataset-manifest ...` path for comparison. A
store-loaded frame is dtype-exact identical to the legacy CSV read, so split
keys are byte-identical. Reproduction: resnet and BiLSTM retrained bit-identically
(weights and all metrics); the CNN reproduced identical thresholds, discrete
metrics and splits, with only average-precision drift <=8.95e-5 that a CSV-only
control run reproduces at the same magnitude (GPU nondeterminism). No model,
metric, parameter or corpus changed; the production checkpoint is unchanged
(`runs/merged_20261017_store_repro/reproduction_report.json`).
Round R013 (2026-10-17) executes the user-approved phase 3 of the dataset-store
plan: after the phase-2 sign-off, the six legacy corpus CSVs
(`datasets/verified_*/verified_training_data*.csv`, 338.9 MB) were deleted; the
SQLite dataset store is now the sole training source. Deletion was preceded by a
strict store verification (`verify --require-legacy-csv`, PASS) and followed by a
normal verification (`verify`, PASS with all six versions reported as
`legacy_csv_state: retired`), because each store export still reproduces the
retired CSV byte-for-byte and matches its stored sha256. Only the six pinned
corpus CSVs were removed; provenance JSONs, per-corpus manifests, staged
provenance, `datasets/manifest.json`, the store and its backup remain, and
`datasets/store/retired_csv_manifest_20261017.json` records every retired file
with its sha256 and a `scripts/dataset_store.py export` restore command. CSV paths
named later in this contract describe the frozen corpora at the time of their
rounds; their files are retired and restorable from the store.
`datasets/` dropped from ~348 MB to ~29 MB. No model, metric, parameter or
deployed checkpoint changed.
Round R014 (2026-10-29) adds `verified_20261029@v1`: 25,728 rows from
48 newly curated exports (47 x 540 rows and `FOV_140_CH_18` x 348). Technical
QC passes all hard gates. Effective-label staging produces 2,562 divisions =
2,609 raw - 51 superseded + 4 added, 98 exclusions and zero overlap. Resolver
statuses are 2,558 `unchanged`, 5 `coalesced_previous`, 4 `shifted_previous`,
and 42 `needs_review`; exported effective labels match the packaged resolver
bitwise. The 42 `needs_review` raw sources are suppressed by effective policy,
merging flanking cycles; this is not new human adjudication. `FOV_140_CH_18`
has 348 rows ending earlier (17.8540 h) but no internal gaps. The independent
raw audit is `PASS_CURRENT_BATCH_WITH_WARNINGS`; it does not independently
establish biological labels or human verification. A separate synthetic finding
(`streamlit_api.py` audit-decision ID float coercion for integer IDs >2**53)
does not affect current-batch boundaries or effective targets; code remains
out of scope for this reporting round. Evidence:
`runs/audit_20261029/raw_independent_audit.json`. The longest exclusion run is
3 frames, so the R004 rule drops none.
Eight reused channel identifiers are date-prefixed in
`datasets/verified_20261029/rename_map.json`. Biological identity across
acquisitions is unestablished; naming overlap is not evidence of re-curation.
No exact measurement/payload duplicate with history was found. The attested
batch is namespaced `curated_20261029::<post-rename key>`, with
`human_verified: true` and `verification_basis: user_attested_2026-10-29`.
The CSV sha256 is
`95654104ae88a50a432064c5772c49e550a0fb6f6d969fbb2a74c0f72d8030e6` and
canonical store hash is
`f3c99c422a3d5c668a40e1c6e3c36982067766581fd4e4203fc81d9aecdf60e2`.
Evidence: `datasets/verified_20261029/{qc_report,staging_report,dataset_manifest}.json`
and `runs/audit_20261029/{preflight,build_report}.json`.

Round R015 (2026-10-29) extends frozen `verified_merged_20261017@v1` with all
48 R014 trajectories to form `verified_merged_20261029@v1#8decab478c80`:
399,168 rows / 1,197 trajectories, 35,032 division and 2,022 exclusion labels,
with 295 historical overlap rows retained. The eight source populations are
baseline_0_10h 477, baseline_0_8h 19, RDM-Gluc 546, starvation 11, and curated
batches 34 / 47 / 15 / 48. The previous identical-content pair in R010 remains
frozen; no historical corpus is rewritten. The canonical hash is
`8decab478c8037d93d412215fd338f1f8f12626b9979d897de9f112537549561`, CSV
sha256 `d8a288a1833858e95f5138d02f1473737d585c44d5cd7a6266b43c1fff9dd566`;
provenance is `datasets/verified_merged_20261029/merge_manifest.json`.
The packaged CLI trains CNN, ResNet, and BiLSTM directly from the store with
unchanged seed-42 defaults and per-source stratification. Final training
comparison and paired same-population evaluation are complete. The two redundant
materialized corpus CSVs were retired after independent data verification (317
checks PASS), and post-retirement store verification passes 18 checks
(`runs/audit_20261029/retirement_report.json`). The store holds 401,076 unique
observations / 1,200 trajectory keys / eight versions / 4,680 trajectory-version
memberships (14,458,880 bytes). Staged provenance,
raw exports and human-readable reports remain. Production stays the R010 CNN;
no promotion is authorized by this retraining round.

Round R016 (2026-10-29) registers a temporal-context experiment before training
(`runs/context_20261029/protocol.json`, protocol sha256
`dc67f1ccebe956e4636ddab7708eca173e02fe0de46a24df7b86c906878bfb96`).
It varies four-entry dilation schedules while holding hidden width 64, four
layers, dropout 0.15, five inputs and other optimization settings fixed. The
nominal dense receptive field is `R = 1 + 4 * sum(d)`, with every integer
offset covered and zero holes in the normalized-input support. This is **not a
strict raw-data end-to-end observation window**: full-trajectory normalization
statistics and phase features are unchanged. Screened contexts are 17, 33, 49,
65, 97, 121, 161 and 197, with dilations [1,1,1,1], [1,1,2,4], [1,2,3,6],
[1,2,5,8], [1,3,7,13], [1,3,8,18], [1,4,11,24] and [1,4,12,32]. Each entry
applies once per CNN convolution and twice per ResNet residual block.

The CNN family retains kernel 5 and 63,938 parameters. The primary ResNet family
uses kernel 3 with four residual blocks and 100,354 parameters; changing dilation
does not change parameter count within either family. The original ResNet kernel
5, dilation [1,2,4,8], RF121 model (165,890 parameters) is a separate seed-42
descriptive control and is excluded from the primary shortlist. Optional
`conv_dilations` remains backward-compatible with the old defaults. Parameter
equality does not isolate receptive-field causality: dilation changes tap
multiplicities along with context, and the primary ResNet kernel differs from
the old kernel-5 default. Dense support alone does not remove those confounds.

The frozen R015 corpus and per-source seed-42 split are unchanged (838/179/180
trajectories). Validation has 179 trajectories / 59,643 rows / 325 exclusion
positives; 74 trajectories are shorter than 197 frames (41.3%). A fixed common
interior at radius 98 has 28,096 rows / 147 exclusion positives for matched
coverage diagnostics. Finalists will be compared on that common interior versus
its boundary and on trajectories shorter versus at least 197 frames. Supplementary
validation diagnostics for all 29 runs are planned by the reporting step after
locking; these are diagnostics, not extra selection criteria.

The seed-42 screen trains 16 family configurations plus the separate original
ResNet control. The top three contexts per architecture on validation all-row
mean of division-event F1 and exclusion F1 proceed to seeds 43 and 44 on the
same frozen split. Ties resolve by validation exclusion F1, division-event F1,
minimum validation loss, then smaller nominal RF. One context per architecture
is selected by the mean of this score over seeds 42/43/44 (same tie order) and
locked **before any test evaluation**. Expected total: 29 training runs. All
three seeds for each locked winner are reported; no best seed is selected.
Because seed 42 chooses the shortlist and is reused in the final mean, the mean
is subject to selection optimism. Three-seed variation measures optimization
noise, not dataset, acquisition or FOV-generalization uncertainty. The irregular
grid does not support a smooth interpolated or universal optimum; any winner is
only best among the screened/shortlisted schedules. Original controls are
single-seed descriptive comparisons, not seed-robust superiority tests.

All 29 runs completed. The seed-42 screen froze the shortlist at CNN RF17/33/121
and primary ResNet RF17/33/97; the lock (`test_evaluated_before_lock: false`)
then selected **RF17 for both families** on the three-seed validation mean
(CNN 0.885582 ± 0.004732; primary ResNet 0.887547 ± 0.006512). Shortlisted CNN
alternatives scored lower (RF33 0.877089 ± 0.003709; RF121 0.849800 ± 0.020845,
the largest seed spread). Within the screened schedules the smallest context
ranked highest, but selection optimism, one split, an irregular grid and
changing dilation path multiplicities prohibit any universal-optimum or causal
receptive-field conclusion.

Locked three-seed test means are CNN RF17 exclusion F1 0.7972 / AP 0.8659
(common panel AP 0.8771) and primary ResNet RF17 0.8043 / 0.8759 (0.8745),
against single-seed controls R015 CNN 0.8026 / 0.8779 (0.8791) and R010
production CNN 0.8183 / 0.8960 (0.8999). **No locked context model surpasses the
deployed checkpoint on these descriptive panels; the primary ResNet family also
differs from the deployed kernel-5 architecture.** Matched radius-98 interior
scores exceed boundary scores for every locked seed, nominal frames correspond
to unequal physical spans (RF17 median ~0.803 h with complete coverage of all
179 validation trajectories; RF197 ~9.831 h covering 105 trajectories and 47.1%
of row centers), and per-source exclusion support ranges from 104 positives
(RDM-Gluc) down to 15 and 2, so small strata are not interpretable. The R015
test, 103-key historical panel and seven new-batch trajectories were previously
reported and remain descriptive, not fresh external holdouts. Width tuning stays
deferred; production remains the R010 CNN and no R016 promotion is authorized.
Independent verification is PASS_WITH_DOCUMENTED_CAVEATS (21/21 checks, no
failures, 183 replay checks per GPU); the Lead's audit of 438 passing checks
included a correction to an audit-only ResNet RF formula that did not touch
training code. GPU0 replay reproduced stored metrics exactly; one CNN RF17
seed-42 validation probability lies within 1e-5 of the 0.87 exclusion threshold,
so cross-device replay flips that single point (precision -0.002738,
F1 -0.001255) and AP ties differ by <=1.6e-4 across devices. Compare discrete
metrics on the original device.

Round R018 (2026-10-29) integrates the independent **training-dataset review
app** (`moma-trainset-review`) as an operational and data-editing round. It
trains no model, reports no new model metric and performs no promotion. Prior
R017 latest-only outputs are not registered in this contract and are not used or
characterised here; R018 is documented separately.

The app source is `src/moma_lineage_analysis/trainset_review/` (nine Python
modules: CLI, entry point, Streamlit UI, store, review, plots, evaluation,
dataset I/O, package init), with the registered console command
`moma-trainset-review`. Launch from the conda environment:

```powershell
& 'G:\miniconda3\envs\Moma_cellpose\python.exe' -m moma_lineage_analysis.trainset_review
& 'G:\miniconda3\envs\Moma_cellpose\Scripts\moma-trainset-review.exe' --server.address 127.0.0.1 --server.port 8502
```

The environment Scripts directory is not on PATH. The default database is
`datasets/store/lineage.sqlite3`; `MOMA_TRAINSET_STORE` or the sidebar
overrides it. Opening/listing/plotting is read-only and does not create review
tables. Review extension tables are added lazily on the first mutation. No live
database was mutated during implementation.

**User-authorized mutation authority.** For app review only, the prior
"historical data are immutable" operating rule is explicitly superseded by user
authority: (1) labels may be edited and autosave immediately, and (2) a whole
trajectory may be permanently deleted. Individual measurements
(`time_h`, `spine_length`) remain immutable in this release. Label edits apply
to shared observations and therefore affect every dataset version that
references the trajectory. Deletion hard-removes its observations, undo
payloads and every version membership; there is no in-app restore. Deletion
requires the exact trajectory key plus an impact display.

Every label edit, conflict resolution and deletion takes an online SQLite backup
first. Backups are stored under `review_backups/` beside the selected database
(inside its parent directory), with path, sha256, bytes, global revision and an
audit event. Strict compare-and-swap on the global revision rejects stale
sessions, and a stale-view guard prevents applying an interaction generated
after another session changed the database. Undo restores only the previous
label payload, never a deletion or general database state.

**Working versions.** When frozen data are reviewed, the original pins, counts,
frozen flag and legacy export hash are copied into review state, an audit event
and snapshot are recorded, and the live version is marked `frozen = 0` with its
legacy export pin cleared and a working canonical hash/counts/order recomputed.
Training resolution refuses invalidated working versions until a new version is
finalized. It is therefore inaccurate to say original frozen lineage bytes or
exports remain byte-identical after user review; archival backups retain the
pre-change state.

**Finalization.** The reviewer selects a new dataset ID (pattern
`^[A-Za-z0-9_.-]+$`), ticks human attestation, and finalizes. The resulting
frozen v1 membership applies the R004 long-exclusion-run filter to the new
membership only (not deletion), records `human_verified = 1` with
`verification_basis = user_reviewed_in_trainset_app`, source-lineage provenance,
a recomputed canonical hash and a byte-exact native export hash. The core store
verifier accepts a review-native version only if its finalize audit event,
provenance, canonical hash and export hash all match. Historical CSV
build/import is refused once a store has reviewed working data, preventing a
deleted trajectory from being resurrected; a separate historical store is
required for such imports.

**Model evaluation exposed in the app.** The shared registry retains the
production **R010 CNN** as default (sha256
`6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`) and adds the
selectable packaged **R015 CNN** from `merged_20261029/cnn` (sha256
`9753298022805b4bc9f277d70e4da29aa8d88cac84a9575691477f332ff21687`;
`verified_merged_20261029@v1`, RF17). Selecting R015 is not promotion. Preview
thresholds start at the checkpoint's calibrated values; applying suggestions is
explicit. Cached inference is keyed by frame content, checkpoint bytes, device,
thresholds and revision. Evaluation split badges (train/validation/test/unseen)
describe the checkpoint's recorded split; agreement is a review aid, not a new
held-out estimate.

R018 acceptance is complete. Final full suite: **388 passed / 4 skipped / 59
subtests** in 81.96 s; launcher help, backend, registry, scoped helper, AppTest
and opaque-ID/cache checks pass. Independent review is **PASS with zero open
defects**. The initial browser permanent-delete confirmation defect
(UI-DELETE-CONFIRM-001) was fixed with an atomic form and re-tested on a fresh
TEMP browser: wrong key errors; exact key saves with a snapshot and removes the
trajectory. Navigation callbacks, finalize selection, stale-view protection and
auto/CUDA device resolution were also fixed and re-tested. Live database and
production checkpoint hashes stayed unchanged; production remains R010 and no
promotion is authorized.

Round R019 (2026-10-31) is a user-requested store-cleanup and retraining round.
It trains the three architectures with unchanged configuration and performs
**no promotion at round close**: production was still the R010 CNN until the
R020 promotion that followed. The round also removes three
duplicate dataset versions from the live store, which is a new data-layer
operation documented here.

**Cleanup request and preconditions.** The user identified the three
`verified_merged_*` dataset versions as duplicates of the five batch versions
and asked to remove them from the database. Before mutating, the store proved
that the union of the three merged memberships (1,152 trajectories) is a subset
of the five batch memberships (1,155 trajectories) with **zero unique
trajectories**, and that the five batch versions cover all 387,217 stored
observations (sum of their row counts equals the observation table). The three
versions were already working (`frozen = 0`) because review edits had
invalidated them; their trajectories, observations and labels are shared with
the batch versions.

**Removal.** `trainset_review.store.remove_dataset_versions` (new admin
capability behind `scripts/remove_dataset_versions.py`) removed
`verified_merged_20261016@v1`, `verified_merged_20261017@v1` and
`verified_merged_20261029@v1` at expected review revision 561 in one
transaction: 3,345 dataset-membership rows, three `dataset_version` rows, their
`ingest_provenance` and `review_version_state` rows and the three `dataset`
rows. The function refuses to orphan data: it deletes only versions whose every
trajectory is still referenced by another version. It took an online snapshot
(`review_backups/lineage_review_20261030T181234.830951Z_r561_remove_dataset_versions.sqlite3`),
recorded a `remove_dataset_versions` audit event at revision 562, and the
pre-removal database backup is
`store/backups/lineage_20261030T181226.223866Z.sqlite3.gz` (sha256
`bad95fa0...`). Observations and trajectories are unchanged: **387,217 rows /
1,155 trajectories** before and after. The three manifest entries are marked
`store_removed` (their CSV hashes stay documented); `scripts/dataset_store.py`
`corpus_specs` skips them, so `verify` expects only the five retained
lineages. Historical R008/R010/R015 lineage ids no longer resolve against the
live store; the pre-removal backup retains them for any historical rebuild.

**Finalization.** Reviewed working versions cannot be training inputs, so each
of the five batches was finalized into a new frozen dataset
(`reviewed_20261031_verified_20260916_no_longruns`, `..._verified_20261013`,
`..._verified_20261016`, `..._verified_20261017`, `..._verified_20261029`, all
`@v1`) with `verification_basis = user_attested_2026-10-31`, the R004
long-exclusion-run filter on the new membership only, `human_verified = 1`,
source-lineage provenance and a byte-exact native export hash (review revisions
563-567). R004 drops three trajectories
(`baseline_0_10h::FOV_28_CH_2`, `curated_20261016::FOV_53_CH_11`,
`curated_20261017::FOV_14_CH_14`) from the finalized membership; nothing is
deleted. The five finalized versions contain **385,815 rows / 1,152
trajectories / 33,665 divisions / 1,740 exclusions / 49 legacy overlap rows**,
eight sources, and compose with zero duplicate observation rows. The 2026-10-13
batch needs no R004 drop because review relabeled
`curated_20261013::FOV_121_CH_7` (longest exclusion run 5 -> 3); the batch still
shrank from 35 to 32 trajectories through three review deletions (22,890 ->
20,928 rows). Store verification after finalization reports five `WORKING`
originals and five `PASS` database-native versions with no failures
(`runs/store_cleanup_20261031/verify_after_finalize.json`).

**Training.** The three architectures were trained sequentially from the
five-lineage composition (`--dataset-lineage` repeated five times,
`--require-human-verified --source-column source_dataset`); no merged corpus
was created. TrainConfig, ModelConfig and the seed-42 per-source trajectory
split were unchanged. The split is **806 train / 173 validation / 173 test**
trajectories (269,699 / 58,007 / 58,109 rows; 23,499 / 5,029 / 5,137 divisions;
1,196 / 270 / 274 exclusions), identical across the three runs and exactly
reconstructed from the store composition. Epochs were 31 / 20 / 29 (best
16 / 5 / 14). Validation division-event/exclusion F1: CNN 0.996910 / 0.805501
(mean **0.901205**), ResNet 0.996415 / 0.723735 (0.860075), BiLSTM
0.996512 / 0.772549 (0.884530); the validation-only rule selects the **CNN**.
Test (acceptance only, never selection): division-event F1
0.996583 / 0.995809 / 0.996189, exclusion F1 0.797642 / 0.749091 / 0.815238,
exclusion AP 0.878254 / 0.830463 / 0.875289. These metrics are not comparable
with R015 per-round values because the reviewed population and labels differ
(45 trajectories had been permanently deleted and many labels edited before
training).

**Descriptive same-row comparison with production.** Each checkpoint used its
own validation thresholds on identical rows. Common test panel unseen in both
rounds: 35 trajectories / 10,317 rows / 873 divisions / 44 exclusions; R019 CNN
division-event F1 0.9983, exclusion F1 0.8395, AP 0.9034 versus production R010
0.9948 / 0.8250 / 0.8618. New-batch 2026-10-29 test panel: 7 trajectories /
3,780 rows / 388 divisions / 14 exclusions; R019 CNN 0.9948 / 0.7586 / 0.8654
versus production 0.9834 / 0.3636 / 0.4804. The full R019 test panel
(173 trajectories / 58,109 rows / 274 exclusions) gives R019 CNN
0.9966 / 0.7976 / 0.8782 versus production 0.9934 / 0.7846 / 0.8623, but
production had seen 130 of those 173 keys in its own train/validation, so that
panel is not a clean head-to-head. The R015 CNN is contaminated on every panel
because its training corpus contains nearly all R019 trajectories. The panels
carry only 44 and 14 exclusion positives, so the differences are descriptive
single-seed observations with no bootstrap or significance test; they do not
authorize or recommend a promotion decision, which remains the user's.

**Independent verification.** A separate read-only verifier recomputed the
cleanup, finalization hashes/counts, R004 rule, manifest/build consistency,
split reconstruction from the store composition and the validation-winner
metrics from the checkpoints: **PASS, 29/29 checks, 0 failures**
(`runs/reviewed_20261031/verification/`). Three documentation-level
discrepancies are recorded, none changing a claimed number:
(1) the live store contains **two** content-identical trajectory pairs, not one -
the documented R009 `CH_16` re-curation and a second `CH_8` pair created by
review label edits; the `CH_16` pair straddles validation/train and the `CH_8`
pair straddles test/validation, so one test trajectory and one validation
trajectory have an exact content twin in another partition (a leakage caveat,
not a key-based count error); (2) `source_file` keeps the three historical
merged-CSV import records after the version removal (metadata only; no dataset,
version, membership or provenance rows remain); (3) CPU metric replay differs
from `metrics.json` only in average precision (up to 3.24e-5; every event count
and point P/R/F1 is exact), consistent with documented device nondeterminism.

Round R020 (2026-10-31) is the deployment round requested by the user after the
R019 report ("promote new cnn model to the main app"). It changes no data, no
training run, no hyperparameter and no metric computation; it promotes the R019
CNN that the validation-only rule selected.

- **Candidate.** `runs/reviewed_20261031/cnn/`: validation division-event F1
  0.996910, exclusion F1 0.805501, mean F1 **0.901205** (highest of the three;
  BiLSTM 0.884530, ResNet 0.860075). Test metrics are acceptance reporting only:
  division-event F1 0.996583, exclusion F1 0.797642, exclusion AP 0.878254.
- **Tool and selection.** `python -m
  moma_lineage_analysis.trajectory_cleaning.promote_best_model --run-dir
  runs/reviewed_20261031 --destination-dir
  src/moma_lineage_analysis/models/trajectory_cleaning_run` re-applies the
  deterministic validation-only ranking, requires byte-identical split manifests
  across cnn/resnet/lstm, stages the four artifacts, verifies sha256 after
  replacement and writes `promotion.json` (schema 1, `uses_test_metrics: false`,
  all validation scores and artifact hashes).
- **Result.** Production `best_model.pt` is byte-identical to
  `runs/reviewed_20261031/cnn/best_model.pt`: sha256
  `816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99`
  (266,165 bytes); `history.csv` `ed01562d...`, `metrics.json`
  `bdd1c390...`, `splits.json` `5fb772fc...` are the R019 CNN artifacts and
  `promotion.json` records the hash-verified copy and scores
  (`promotion_backups/20261030T184217.243155Z/`).
- **Rollback source.** The previous R010 CNN checkpoint
  (`6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b`) and its
  `history.csv`/`metrics.json`/`splits.json` were backed up under
  `src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261030T184217.243155Z/`.
  Because the tool does not archive the previous `promotion.json`, the R010
  manifest was copied into that backup directory manually (and retained at
  `runs/reviewed_20261031/promotion_previous_R010.json`).
- **Post-promotion checklist.** New hash equals the run checkpoint and
  `promotion.json`; backup contains the previous artifacts; `load_checkpoint` +
  predict smoke produced finite outputs; the shared registry
  (`model_registry.py`), Streamlit AI-panel label test and contract-integrity
  pin were updated; `datasets/manifest.json`, README, AGENTS.md and this
  contract record the deployment. No corpus, dataset version, split, threshold
  or training code changed. The R019 same-row comparison remains descriptive
  single-seed context for the user's decision; it was not used to select the
  model.

### Population, analysis unit, and independent replicate

- Row grain: one observation = one cell at one time point in one trajectory,
  keyed by (`trajectory_key`, `fov_cell_index`).
- Analysis unit for metrics: the observation (point-level P/R/F1, AP) and the
  division event (event-level P/R/F1 with `division_tolerance_frames = 1`).
- Independent replicate: the trajectory; trajectories never cross
  train/validation/test partitions. This is NOT FOV-held-out validation — one
  FOV may contribute trajectories to more than one partition (recorded as a
  limitation).
- Inclusion: every trajectory in the verified CSV with at least two time
  points. Exclusion: rows failing preflight feature validation (non-finite
  `time_h`/`spine_length`, non-increasing time) are dropped by
  `TrajectoryDataset` construction; rows failing the Streamlit preflight
  (`preflight_ai_trajectory_inputs`) are retained for audit but scored by the
  deterministic validation rule instead of the network.

### Preprocessing and quality control

1. `load_trajectory_csv` / `validate_trajectory_dataframe`: coerce
   `time_h`, `spine_length` to finite floats; parse label columns
   (`true/false/1/0/yes/no`); reject missing trajectory keys.
2. Rows are sorted within each trajectory by (`time_h`, original row
   position) with mergesort; the original row order is retained via
   `row_positions` so predictions map back to the input CSV order.
3. `TrajectoryNormalizer.transform` (per trajectory, using the full
   trajectory — offline cleaning, not causal):
   - time feature: elapsed time in units of the trajectory's median positive
     sampling interval, divided by `time_step_scale = 100`;
   - length feature:

$$
f_{length} = \operatorname{clip}\!\left(\frac{\log(\ell_t / \operatorname{median}(\ell))}{1.4826\, \mathrm{mad}(\log(\ell)) \vee 0.05},\; -12,\; 12\right)
$$

   Non-positive spine lengths warn and are clamped before the log.
   - Round-R002 boundary-safe dynamics features (extended layout, five
     channels; the two v0.1.0 channels plus):
     - `length_delta`: first difference of the raw (unclipped) length feature
       along the trajectory, padded with 0 at the first frame — the division
       signature is a large negative value;
     - `log_growth_rate`: `length_delta` divided by the frame-time difference
       (per median sampling interval) — the physiological growth rate;
     - `time_since_last_drop`: frames (in median sampling intervals, divided by
       `time_step_scale = 100`) since the previous `length_delta <= -1.0` event
       — a boundary-safe proxy for cell-cycle phase that does not depend on
       absolute experiment time.
   These three features are stationary with respect to absolute experiment
   time, addressing the R002 root cause (temporal extrapolation of the
   absolute-time channel beyond the training range). `TrajectoryNormalizer`
   defaults to the extended layout; `feature_compatibility="legacy"`
   reproduces the v0.1.0 two-channel layout and is restored automatically for
   pre-R002 checkpoints.
4. Padding: `collate_trajectories` zero-pads to the batch maximum length and
   returns a boolean validity mask; all losses and metrics are mask-weighted.
5. No missing-value imputation: invalid feature rows are excluded from the
   model input (train) or routed to the validation rule (inference).

### Variables and calculations

Model outputs are per-point logits `z_t = (z_t^div, z_t^exc)`; probabilities
are `p_t = \sigma(z_t)`. Training minimizes a mask-weighted focal binary
cross-entropy with positive-class weighting:

$$
\mathcal{L} = \frac{1}{|\lbrace(t,k): m_{t,k}\rbrace|} \sum_{t,k} m_{t,k}\, w_{y}\, \big(1 - p_{t,k}^{(y)}\big)^{\gamma}\, \mathrm{bce}(z,\, y)\bigr|_{t,k}
$$

with `gamma = 1.5`, `w_y = clip(N_neg / N_pos, 1, 50)` per label computed on
the training split only.

Decision rule at inference:

$$
\hat{y}_t^{div} = \mathbb{1}[p_t^{div} \ge \tau_{div}], \qquad \hat{y}_t^{exc} = \mathbb{1}[p_t^{exc} \ge \tau_{exc}]
$$

Thresholds `\tau` maximize point-level F1 on the validation split over a grid
`0.05..0.95` (91 points). Division-event metrics cluster predicted and true
division indices with `cluster_event_indices`; a prediction within
`division_tolerance_frames = 1` of a true event counts as a true positive.
Division/exclusion label conflicts are resolved by
`resolve_division_exclusion_conflicts` (drop-ratio evidence threshold 0.75;
statuses `unchanged`, `coalesced_previous`, `shifted_previous`,
`needs_review`).

### Statistical analysis

- Split: `split_trajectory_keys`, seed 42, 70/15/15
  train/validation/test over trajectory keys, stratified (when feasible) by
  (has-excluded, division-count bin), performed independently per
  `source_dataset` so every condition appears in every partition.
- Architectures compared: 1D CNN (`CNNLabeler`, 62,978 params with 2-channel
  input; 63,938 with 5-channel input), dilated 1D ResNet (`ResNetLabeler`,
  165,698 params with 2-channel input; 165,890 with 5-channel input),
  bidirectional LSTM (`LSTMLabeler`, 333,058 params with 2-channel input;
  334,594 with 5-channel input). Default `ModelConfig` since R002: input 5,
  output 2, hidden 64, 4 layers, kernel 5, dropout 0.15. (Pre-R002
  checkpoints carry `input 2` in their saved config.)
- Optimization: AdamW (lr 1e-3, weight decay 1e-4), grad-norm clip 5.0, batch
  16, at most 100 epochs, early stopping (patience 15) restoring the
  minimum-validation-loss state.
- Model selection: maximize mean of validation division-event F1 and
  validation exclusion F1; validation only — test metrics are reported but
  never used for selection or thresholding.
- Estimators: point-level precision/recall/F1 and average precision per label;
  division-event precision/recall/F1 and mean absolute frame error; plus the
  R002 position-stratified metrics — division recall in the head (first
  quintile) and tail (last quintile) of each trajectory by relative position,
  and the fraction of trajectories whose LAST true division event is missed
  (`last_division_event.miss_rate`, the user-reported failure mode). No
  uncertainty intervals are currently reported (limitation). All seeds fixed at
  42; no multiplicity correction (single comparison framework). R016 is a
  preregistered exception for training-seed repeats 43/44; split seed 42 stays
  frozen and shortlist reuse in the final mean retains selection optimism.

### Figures and tables

Report figures are regenerated by `trajectory_cleaning/plot_training_results.py`
into `trajectory_cleaning/figures/<run_id>/`:

- `training_loss_curves.png` — train/validation mask-weighted focal loss vs
  epoch per architecture; diamond marks the restored best epoch. Colors:
  CNN #0072B2, ResNet #E69F00, BiLSTM #009E73.
- `model_f1_comparison.png` — validation vs test division-event F1 and
  exclusion F1 per architecture (fixed split, no error bars).
- `dataset_label_balance.png` — canonical-label prevalence and positive
  support per split.

Tables in `trajectory_cleaning/MODEL_REPORT.md`: architecture comparison,
split summary, and the model-selection metric table (reported to 3 decimals).
The Streamlit app renders trajectory plots (spine length vs `time_h`) with
model probability overlays and lasso-based manual curation; those are
interactive diagnostics, not contract figures.

### Validation and acceptance criteria

- `fov_cell_index` unique within a trajectory frame; trajectory keys unique,
  non-missing, hashable; time strictly increasing within a trajectory after
  sorting (ties rejected unless identical — see unresolved items).
- Every trajectory has >= 2 time points; trajectories failing this are listed
  in the prepare-report `trajectories_with_fewer_than_two_retained_points`.
- Normalization invariance identity (unit test):
  `transform(t, l) == transform(a*t + b, c*l)` within 2e-6.
- Checkpoint round trip: `save_checkpoint` -> `load_checkpoint` reproduces
  `predict_dataframe` probabilities exactly (CPU).
- Split completeness: train/validation/test keys disjoint and covering all
  keys.
- Validation-gate test battery: 78 unittest cases in this directory
  (normalization, dataset, collate, split, loss, training smoke test, threshold
  calibration, checkpoint I/O, event clustering, preflight, conflict
  resolution, apply/export attachment, promotion, data preparation).
- Failure behavior: invalid labels, non-finite features, empty frames, bad
  thresholds, and malformed checkpoints raise `ValueError`/`TypeError` —
  never silent coercion.

### Interpretation boundary and limitations

- `spine_length` is a segmentation-derived proxy for cell size; the model
  inherits every upstream segmentation bias.
- Normalization uses the full trajectory (non-causal); the model is only valid
  for offline cleaning of complete trajectories.
- Held-out partitions share FOVs across splits — acquisition-specific patterns
  can leak; treat metrics as upper bounds until FOV-held-out cross-validation
  is run.
- The starvation condition is severely under-represented (2 trajectories per
  held-out partition); starvation exclusion F1 (test 0.273) is the clearest
  known weakness.
- Thresholds and model selection both consume the validation split; no
  separate calibration split exists.
- Predictions are decision support: the deployed contract is human-reviewed
  previews, never unattended deletion.
- `excluded_point` is the rarer, noisier label; exclusion metrics have wide
  (unquantified) uncertainty.

## Layer 2 — Controlled Registry and Constraints

### Terminology registry

| Metric ID | Canonical term | Exact field | Symbol | Unit | Data grain | Formula or algorithm | Interpretation | Limitation | Plot label | Deprecated aliases | Status / introduced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `DIVISION_POINT_LABEL` | division point label | `division_point` | y_div | boolean | observation | Human-curated flag: observation is the mother frame immediately before a division event | Canonical training/inference target for division | Subjective curation; overlap with exclusion label allowed | Division point | None | Active / 0.1.0 |
| `EXCLUDED_POINT_LABEL` | excluded point label | `excluded_point` | y_exc | boolean | observation | Human-curated flag: observation is noise and must be excluded from growth analysis | Canonical training/inference target for exclusion | Rarest, least stable label; starvation support tiny | Excluded point | None | Active / 0.1.0 |
| `PRED_DIVISION_PROB` | predicted division probability | `pred_division_probability` | p_div | probability in [0, 1] | observation | sigmoid of model division logit | Model confidence that the observation is a division point | Not probabilistically calibrated; threshold tuned for F1 | P(division) | None | Active / 0.1.0 |
| `PRED_EXCLUDED_PROB` | predicted exclusion probability | `pred_excluded_probability` | p_exc | probability in [0, 1] | observation | sigmoid of model exclusion logit; forced to 1.0 for validation-rule rows | Model confidence that the observation is excludable | Validation-rule rows are deterministic, not model confidence | P(excluded) | None | Active / 0.1.0 |
| `POINT_F1` | point-level F1 | `point_level.<label>.f1` in metrics.json | F1 | unitless | observation, per label | harmonic mean of precision and recall at the calibrated threshold | Per-label binary classification quality | Threshold-dependent; threshold chosen on validation | Point F1 | None | Active / 0.1.0 |
| `POINT_AP` | point-level average precision | `point_level.<label>.average_precision` | AP | unitless | observation, per label | `sklearn.metrics.average_precision_score` | Threshold-independent ranking quality | — | AP | None | Active / 0.1.0 |
| `DIVISION_EVENT_F1` | division-event F1 | `division_event.f1` in metrics.json | F1_event | unitless | division event | event-level P/R/F1 with `cluster_event_indices`, tolerance ±1 frame | Biological correctness of division timing | Tolerant by design; a ±1 frame error counts as a true positive | Event F1 | None | Active / 0.1.0 |
| `DIVISION_EVENT_MAE` | division-event mean absolute frame error | `division_event.mean_absolute_error_frames` | MAE_frames | frames | matched division event | mean \|predicted frame − true frame\| over matched events | Timing sharpness of matched events | Defined only over true positives | Event MAE | None | Active / 0.1.0 |
| `SPINE_LENGTH` | spine length | `spine_length` | ell_t | µm (calibrated upstream) | observation | fitted cell-spine arc length multiplied by `umppx` upstream (`cell_coord_utility.Cell`; DT: `_length_px * umppx`) | Proxy for cell size; drives both labels | Segmentation-derived; calibration depends on acquisition `umppx` | Spine length (µm) | None | Active / 0.1.0 |
| `DIVISION_RECALL_TAIL` | tail-quintile division recall | `position.division_recall_tail_quintile.recall` | R_tail | unitless | observation, division positives with relative position >= 0.8 | fraction of true division points in the last trajectory quintile predicted at the calibrated threshold | Boundary recall — the R002 failure-mode metric | Quintile boundaries per trajectory; support varies | Tail division recall | None | Active / 0.2.0 |
| `DIVISION_RECALL_HEAD` | head-quintile division recall | `position.division_recall_head_quintile.recall` | R_head | unitless | observation, division positives with relative position <= 0.2 | fraction of true division points in the first trajectory quintile predicted at the calibrated threshold | Baseline recall for contrast with tail | Quintile boundaries per trajectory | Head division recall | None | Active / 0.2.0 |
| `LAST_EVENT_MISS_RATE` | last-division-event miss rate | `position.last_division_event.miss_rate` | M_last | unitless | trajectory | fraction of trajectories with at least one true division whose LAST true division event is unmatched (tolerance ±1 frame) | The user-reported failure mode (missed final division) | Trajectory-level; no uncertainty interval | Last event miss rate | None | Active / 0.2.0 |

Units documentation correction (2026-10-26): the supported upstream `Cell`
implementation multiplies fitted spine length by `umppx`
(`G:/python_code/Moma_cellPose/cell_coord_utility.py:1156–1157`), and `CellDT`
uses `_length_px * umppx` (`cell_morphology_compat.py:182`). The upstream CSV
exporters carry those calibrated attributes through unchanged; the full source
locators are recorded in `docs/METHODS_LINEAGE_TRAJECTORY_CONSTRUCTION.md`.
The earlier pixels wording was incorrect. This corrects prose only: numerical
features, normalization, frozen corpora, model artifacts, and the minimum-length
filter are unchanged; no cleaning-model method-version or round is added.

### Parameter registry

| Parameter ID | Code name | Active value | Unit | Role | Selection basis | Sensitivity requirement | Version introduced |
|---|---|---:|---|---|---|---|---|
| `MODEL_CONFIG` | `ModelConfig` defaults | input 5, output 2, hidden 64, layers 4, kernel 5, dropout 0.15 (pre-R002 checkpoints: input 2) | — | Architecture capacity | Fixed default shared across the three compared architectures | Architecture comparison is the sensitivity analysis | 0.1.0 (input 5 from 0.2.0) |
| `TRAIN_CONFIG` | `TrainConfig` defaults | epochs 100, batch 16, lr 1e-3, wd 1e-4, focal_gamma 1.5, max_pos_weight 50, patience 15, seed 42 | — | Optimization | Empirically stable on the verified corpus | Retrain with a second seed before claiming robustness | 0.1.0 |
| `DIVISION_TOLERANCE` | `division_tolerance_frames` | 1 | frames | Event matching tolerance in division-event metrics | One sampling interval matches human curation precision | Report tolerance sensitivity (0/1/2) at next retrain | 0.1.0 |
| `THRESHOLD_GRID` | `best_f1_threshold` grid | 0.05-0.95, 91 points | probability | Validation threshold calibration | Covers observed operating region | Widen if a future checkpoint saturates grid edges | 0.1.0 |
| `SPLIT_FRACTIONS` | `split_trajectory_keys` | validation 0.15, test 0.15, seed 42 | trajectory fraction | Data partitioning | Standard small-corpus split; per-source stratification | FOV-held-out CV is the required robustness check | 0.1.0 |
| `NORMALIZER_SCALE` | `TrajectoryNormalizer` | clip 12, time_step_scale 100, min_length_scale 0.05, MAD consistency 1.4826, feature_compatibility extended/legacy (5-channel layout since 0.2.0) | — | Feature scaling | Unit/scale invariance with robust (MAD) statistics; extended layout adds boundary-safe dynamics features | — | 0.1.0 (feature_compatibility from 0.2.0) |
| `R016_CONTEXT_GRID` | `conv_dilations` | 17, 33, 49, 65, 97, 121, 161, 197 via registered four-entry schedules | normalized-input frames (not raw end-to-end window) | Experimental dense temporal support | Best among screened/shortlisted schedules only; validation-only lock; seed repeats measure optimization noise | Dilation path multiplicities change with RF; diagnose common radius 98 and length strata <197 / >=197 | 0.9.0 |
| `R016_CAPACITY` | `hidden_size`, `kernel_size` | hidden 64 fixed; CNN kernel 5 = 63,938 params; primary ResNet kernel 3 = 100,354; old ResNet kernel 5 control = 165,890 | parameters | Isolate the schedule question at fixed within-family capacity | Defer width tuning; do not pool the primary k3 ResNet sweep with the old k5 control | Same parameters do not isolate RF causality; kernel differs in primary ResNet family | 0.9.0 |
| `R018_MUTATION_AUTHORITY` | review backend | label edit/global autosave and permanent whole-trajectory delete; `time_h`/`spine_length` immutable | — | User-authorized local training-data review | Explicit per-action user confirmation; delete requires exact key + impact | Every mutation CAS-guarded and backed up; working versions cannot train | 0.10.0 |
| `R018_FINALIZATION` | review `finalize` | new dataset ID; R004 membership filter; human attestation; native export hash | dataset version | Produce a new frozen training lineage | User-reviewed content and R004 production rule | Core verify accepts only matching finalize audit + source provenance + canonical/export hashes | 0.10.0 |
| `AI_MODEL_REGISTRY` | shared registry / review app | R019 production default; R015 selectable, non-production | model checkpoint | Human-reviewed preview selection | Released checkpoints only; R015 not promoted | R019 default (promoted R020); selecting R015 is not deployment | 0.11.1 |
| `R020_PROMOTION` | `promote_best_model` | validation-only selection; hash-verified four-artifact copy to the package root | production checkpoint | User-authorized deployment | Previous artifacts + previous promotion manifest archived under `promotion_backups/` | Copy is byte-identical to the run checkpoint; test metrics never used for selection | 0.11.1 |
| `DIVISION_DROP_RATIO` | `division_drop_ratio` | 0.75 | length ratio | Conflict-resolution evidence threshold | A true division roughly halves length; 0.75 leaves noise margin | Sensitivity unknown; flagged in unresolved items | 0.1.0 |

### Data structure contract

| Object or table | Source | Grain | Unique key | Required fields | Inclusion/exclusion behavior | Version or fingerprint |
|---|---|---|---|---|---|---|
| verified training CSV | `verified_20260916/verified_training_data.csv` (archived in legacy workspace, not shipped) | one row = observation | (`trajectory_key`, `fov_cell_index`) | `trajectory_key`, `fov_cell_index`, `time_h`, `spine_length`, `division_point`, `excluded_point`, provenance fields | supersession omission; invalid feature rows omitted from model input | sha256 deff25a96a7f572b86b0ce7a033b3c4744098d0e73462d44d3db896810c23cba; 318,745 rows / 1,070 trajectories (R002) |
| verified training CSV (R004, shipped) | `datasets/verified_20260916/verified_training_data_no_longruns.csv` (retired in R013; restore via store export) | one row = observation | (`trajectory_key`, `fov_cell_index`) | same schema | R004 long-run exclusion filter (17 trajectories dropped) | sha256 a2b3793d2138d806f5d5a85c3f51e15b9629953b4c0ef54b9490b74f99bc0092; 312,317 rows / 1,053 trajectories |
| verified training CSV (batch R009) | `datasets/verified_20261017/verified_training_data.csv` (retired in R013; restore via store export) | one row = observation | (`trajectory_key`, `fov_cell_index`) | same schema; namespaced `curated_20261017::<key>`; `source_trajectory_key` is the post-rename key (pre-rename key in `rename_map.json`) | effective-label staging (`--division-mode effective`); merged into the R010 retraining corpus (`verified_merged_20261017`), not into the production corpus | sha256 247da098e97d060c4c24696ef4ed78862a0ef3932b717492d21f1c8219d8ad67; 8,757 rows / 16 trajectories |
| verified training CSV (R010 merged) | `datasets/verified_merged_20261017/verified_training_data.csv` (retired in R013; restore via store export) | one row = observation | (`trajectory_key`, `fov_cell_index`); keys pre-namespaced `<source_dataset>::<key>` | same schema | concatenation of the 4 verified sources; R004 filter applied to the 3 batches (dropped `FOV_121_CH_7`, `FOV_53_CH_11`, `FOV_14_CH_14`); legacy R002-policy overlap rows in the frozen R004 portion retained | sha256 7ecc9753cf02c0dfc020fd70f6b077f997799d46d4f3af3fd89d12a16eaee2bc; 373,440 rows / 1,149 trajectories |
| verified training CSV (historical) | `verified_20260718/verified_training_data.csv` (archived in legacy workspace, not shipped) | one row = observation | (`trajectory_key`, `fov_cell_index`) | same schema | supersession omission | sha256 ef731c00bd774a6dfadcbf9c2d5e3f9cd934548ca9ffb5a3457bd6c515a7fc10; 77,246 rows / 510 trajectories (R001, superseded by verified_20260916) |
| dataset manifest | `datasets/verified_20260916/dataset_manifest.json` | one object = corpus provenance | `schema_version` 1.0 | `human_verified`, `canonical_target_columns`, `sources`, `supersession`, `namespace` | gates `--require-human-verified` training runs | user_attested_2026-09-16 |
| run artifacts | `models/trajectory_cleaning_run/verified_20260916/<model>/` (cnn, resnet, lstm, shipped under `src/moma_lineage_analysis/`) | one directory = trained model | model name | `best_model.pt`, `history.csv`, `splits.json`, `metrics.json` | selection artifacts retained per architecture | run verified_20260916 |
| deployed checkpoint | `models/trajectory_cleaning_run/best_model.pt` (shipped under `src/moma_lineage_analysis/`) | one file = promoted CNN checkpoint | checkpoint sha256 | model_config, normalizer, thresholds, metadata, state_dict | promotion is verified backup + hash-checked copy | sha256 816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99 (R019 CNN, promoted 2026-10-31 by user decision from `runs/reviewed_20261031/cnn/`; previous R010 CNN `6e7c5a58…` backed up with its promotion record under `promotion_backups/20261030T184217.243155Z`) |
| verified batch R014 | `verified_20261029@v1` in `datasets/store/lineage.sqlite3` | one row = observation | (`trajectory_key`, `fov_cell_index`) | 11-column canonical schema; `curated_20261029::<post-rename key>` | effective labels; no R004 drops; eight naming collisions date-prefixed | CSV sha256 `95654104ae88a50a432064c5772c49e550a0fb6f6d969fbb2a74c0f72d8030e6`; canonical `f3c99c422a3d5c668a40e1c6e3c36982067766581fd4e4203fc81d9aecdf60e2`; 25,728 rows / 48 trajectories |
| merged training corpus R015 | `verified_merged_20261029@v1` in `datasets/store/lineage.sqlite3` | one row = observation | (`trajectory_key`, `fov_cell_index`) | same schema; eight sources | frozen R010 + all R014; 295 legacy overlap rows retained | CSV sha256 `d8a288a1833858e95f5138d02f1473737d585c44d5cd7a6266b43c1fff9dd566`; canonical `8decab478c8037d93d412215fd338f1f8f12626b9979d897de9f112537549561`; 399,168 rows / 1,197 trajectories |
| R016 registered context protocol | `runs/context_20261029/protocol.json` | one pretraining protocol = two families, eight schedules plus one old-ResNet control; 29 expected training runs | `round`, dataset lineage/split, architectures, schedules, seeds, selection/reporting rules | Frozen R015 store/split; CNN and primary ResNet k3; old ResNet k5 RF121 control separate; final lock before test; all three winner seeds | protocol sha256 `dc67f1ccebe956e4636ddab7708eca173e02fe0de46a24df7b86c906878bfb96`; corpus canonical `8decab478c80…`; no data changes |
| R018 review state / audit | `datasets/store/lineage.sqlite3` review extension tables (lazy) | one global revision; per-event audit; per-version original pins; payloads | global singleton / event / version id | revision, event type, actor, snapshot path+sha256+bytes; original frozen state | created only on outer transaction first mutation; read-only opening never creates | CAS global revision; stale writes rejected |
| R018 reviewed working version | live `dataset_version` row after a mutation | one dataset version | (`dataset_id`, `version`) | working canonical hash/counts/order; `frozen = 0`; original pins in review state | labels shared globally; delete removes trajectory/memberships; training resolution refuses | not byte-identical to original frozen version after review |
| R018 review-finalized dataset | new `dataset_id@v1` from app finalization | one frozen dataset version | (`dataset_id`, version) | canonical hash, native export sha256, human_verified=1, source lineage | R004 filter on new membership only; no deleted-trajectory resurrection via CSV import | accepted by core verify only with matching finalize audit + provenance + hashes |
| R019 duplicate dataset-version removal | `trainset_review.store.remove_dataset_versions` / `scripts/remove_dataset_versions.py` | one removed dataset version | (`dataset_id`, `version`) | removal audit event, pre-removal snapshot, surviving membership proof | only versions whose every trajectory stays referenced by another version; refuses orphaned observations | review revision CAS; 3 versions / 3,345 memberships removed in R019; observations and trajectories unchanged |
| R019 reviewed training-ready lineages | five `reviewed_20261031_*@v1` frozen versions | one dataset version per batch | (`dataset_id`, `version`) | canonical hash, native export sha256, human_verified=1, source-lineage provenance, R004 membership filter | totals 385,815 rows / 1,152 trajectories / 33,665 divisions / 1,740 exclusions / 49 overlap; three R004-dropped keys are excluded from membership only | verify `PASS` (database_native); source batches remain working versions |
| R019 composed training corpus | composition of the five reviewed lineages via repeated `--dataset-lineage` | one row = observation | (`trajectory_key`, `fov_cell_index`) | 11-column canonical schema; eight sources; zero duplicate observation rows | split 806/173/173; no merged dataset version is created in the store | lineage ids and canonical hashes recorded in `runs/reviewed_20261031/*/metrics.json` |
| Streamlit trajectory dict | `Moma_lineage_streamlit.py` session state | mapping key = trajectory_key | trajectory key | `trajectory` frame with `fov_cell_index`, `time_h`, `spine_length` | preflight drops invalid rows to validation-rule scoring | TRAJECTORY_STATISTICS_SCHEMA_VERSION = 2 (independent lineage method 1.1.0 / logic 2.3) |

### Global analysis constraints

- Do not silently redefine, rename, clip, normalize, interpolate, exclude, or aggregate a registered metric.
- R018 data mutations are only explicit user actions in the training-review app; never edit the live store from scripts, tests, or model-training paths. Original frozen lineage/export bytes do not remain identical after review; finalize new IDs.
- Treat repeated observations from one independent unit as dependent.
- Export the exact data used for each plot or table.
- Preserve non-comparable historical outputs under versioned filenames.
- Record all exploratory choices as exploratory or retrospective.
- Delimit display equations with `$$` on separate lines; never use `\[` and `\]`.

### Plot contract

| Plot ID | Metrics | Population | Aggregation | Uncertainty | Axes / transformation | Style / colors | Exported point table |
|---|---|---|---|---|---|---|---|
| `LOSS_CURVES` | train/validation masked focal loss | all three architectures | per epoch | none | epoch vs loss | CNN #0072B2, ResNet #E69F00, BiLSTM #009E73; diamond at best epoch | `history.csv` per model |
| `F1_COMPARISON` | division-event F1, exclusion F1 | validation and test partitions | per architecture | none | model vs F1 | grouped bars, same model colors | `metrics.json` per model |
| `LABEL_BALANCE` | label prevalence and support | train/validation/test | per label per split | none | split vs count/fraction | two-panel bars | verified training CSV |

### Output registry

| Output ID | Path | Content | Method version | Generated by | Validation status |
|---|---|---|---|---|---|
| `METHODS_CONTRACT` | `docs/METHODS_TRAJECTORY_CLEANING_MODEL.md` | Living Methods contract | 0.11.1 | scientific-analysis-loop | R020 promoted the R019 CNN; R019 cleanup/retraining independently verified PASS; R018 closed; R016 closed |
| `R004_MODELS` | `trajectory_cleaning_run/no_longruns_20260916/` (archived in legacy workspace; promoted CNN shipped as `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt`) | R004 cnn/resnet/lstm checkpoints, histories, metrics, comparison | 0.3.0 | `dev_code/r004_train_no_longruns.py` (archived) | 86-test battery OK; R004 CNN promoted via `promotion.json` |
| `R004_CORPUS` | `datasets/verified_20260916/verified_training_data_no_longruns.csv` (retired in R013; restore via store export) | R004 production training corpus (frozen) | 0.3.1 | round R004 filter over R002 corpus | sha256 verified 2026-10-13 (`a2b3793d…`); 312,317 rows / 1,053 trajectories |
| `R009_CORPUS` | `datasets/verified_20261017/verified_training_data.csv` (retired in R013; restore via `scripts/dataset_store.py export`) | R009 batch training corpus (now merged into the R010 retraining corpus) | 0.4.1 | `scripts/qc_raw_train_data.py` + `scripts/stage_verified_labels.py` + `trajectory_cleaning/prepare_verified_training_data.py` | sha256 `247da098…` verified 2026-10-17; 8,757 rows / 16 trajectories (independent verifier PASS) |
| `R010_MODELS` | `runs/merged_20261017/` | R010 cnn/resnet/lstm checkpoints, histories, metrics, splits, comparison, figures, verification report | 0.5.0 | packaged `moma-trajectory-cleaning train` on `verified_merged_20261017` (GPU) | independent verifier PASS (corpus 11/11, artifacts 8/8); validation-selected cnn; promoted 2026-10-17 (R011, see `R011_PROMOTION`) |
| `R011_PROMOTION` | `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt` (+ `history.csv`, `metrics.json`, `splits.json`, `promotion.json`) | promoted production checkpoint (R010 CNN) | 0.5.1 | `trajectory_cleaning/promote_best_model.py --run-dir runs/merged_20261017` | hash-verified copy of `runs/merged_20261017/cnn/best_model.pt` (sha256 `6e7c5a58…`); previous R008 CNN backed up under `promotion_backups/20261017T124420.980675Z`; contract-integrity pin updated |
| `R012_STORE_TRAINING` | `datasets/store/lineage.sqlite3` (+ `SCHEMA.md`, `backups/`), `runs/merged_20261017_store_repro/` | SQLite dataset store (six frozen versions) + store-direct training reproduction of R010 (cnn/resnet/lstm) | 0.6.0 | `trajectory_cleaning/dataset_store.py`, `scripts/dataset_store.py`, `trajectory_cleaning/cli.py --dataset-lineage` | store verify 6/6 versions PASS (counts, canonical hashes, legacy CSV sha256, byte-identical exports); independent phase-1 auditor PASS; R010 reproduction: byte-identical splits, resnet/lstm bit-identical, cnn AP drift <=8.95e-5 matched by a CSV-only control (GPU nondeterminism); no CSVs deleted |
| `R013_STORE_RETIREMENT` | `datasets/store/retired_csv_manifest_20261017.json` (+ six `datasets/verified_*/` provenance folders) | retirement record for the six legacy corpus CSVs + post-retirement verification evidence | 0.7.0 | `python scripts/dataset_store.py verify` before/after deletion; explicit user sign-off | strict verify PASS before deletion (14 checks); normal verify PASS after deletion with all six versions `legacy_csv_state: retired`; exports still byte-identical to the pinned sha256; `datasets/` 348 MB -> 29 MB; restore command recorded per corpus |
| `R014_CORPUS` | `datasets/verified_20261029/` + `verified_20261029@v1` | attested batch, staging/QC/rename provenance | 0.8.0 | effective-label staging + verified preparation + store ingestion | 25,728 rows / 48 trajectories; build report PASS; no R004 drops |
| `R015_MODELS` | `runs/merged_20261029/` | CNN/ResNet/BiLSTM checkpoints, histories, metrics, splits, comparison, paired evaluation and figures | 0.8.0 | packaged CLI, `--dataset-lineage verified_merged_20261029@v1` | CNN validation winner 0.888577; mixed descriptive common/new-batch panels; no replacement support; no promotion |
| `R016_CONTEXT` | `runs/context_20261029/` | protocol, environment/preservation records, screen/repeat/selection_lock/evaluation/summary/validation_diagnostics/control_evaluation, figures and verification_report | 0.9.0 | `scripts/sweep_temporal_context.py` and `scripts/report_temporal_context.py` on frozen `verified_merged_20261029@v1` | 29 runs complete; RF17 locked for both families; no locked model exceeds production; verification PASS_WITH_DOCUMENTED_CAVEATS (21 checks); no promotion |
| `R018_REVIEW_APP` | `src/moma_lineage_analysis/trainset_review/`, `docs/TRAINSET_REVIEW.md`, `runs/trainset_review_implementation/` | independent Streamlit review/edit/finalize app; guarded SQLite review backend; user guide; test evidence | 0.10.0 | `moma-trainset-review` / `python -m moma_lineage_analysis.trainset_review` | final suite 388 passed / 4 skipped / 59 subtests; independent verification PASS with zero open defects; no promotion |
| `R019_STORE_CLEANUP` | `runs/store_cleanup_20261031/` (`before_state.json`, `after_removal_state.json`, `finalize_results.json`, `training_frame.json`, `verify_after_finalize.json`), `store/backups/lineage_20261030T181226.223866Z.sqlite3.gz` | duplicate-version removal evidence, pre/post store states, finalization results and store verification | 0.11.0 | `scripts/remove_dataset_versions.py`, `trainset_review.store.finalize` | removal at revision 562 (3 versions / 3,345 memberships, 0 orphaned trajectories); 5 finalized lineages at revisions 563-567; verify PASS (5 WORKING + 5 PASS) |
| `R019_MODELS` | `runs/reviewed_20261031/` | CNN/ResNet/BiLSTM checkpoints, histories, metrics, splits, model comparison, production comparison, figures, training summary | 0.11.0 | packaged CLI with five repeated `--dataset-lineage` args on the reviewed lineages | CNN validation winner 0.901205; test div-event/exclusion F1 0.9966/0.7976 (AP 0.8783); descriptive same-row panels vs the then-production R010; promoted in R020 |
| `R020_PROMOTION` | `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt` (+ `history.csv`, `metrics.json`, `splits.json`, `promotion.json`), `promotion_backups/20261030T184217.243155Z/` | promoted production checkpoint (R019 CNN) and rollback backup of the R010 CNN | 0.11.1 | `trajectory_cleaning/promote_best_model.py --run-dir runs/reviewed_20261031` on user decision | hash-verified copy of `runs/reviewed_20261031/cnn/best_model.pt` (sha256 `816a8b39…`); validation-only rule re-applied (cnn 0.901205 / BiLSTM 0.884530 / ResNet 0.860075); previous R010 CNN + its archived `promotion.json` backed up; contract-integrity pin and registry labels updated |

### Declared method manifest

The JSON below is the machine-readable intended contract used for code-method reconciliation.

<!-- METHOD_MANIFEST_START -->
```json
{
  "analysis_id": "trajectory_cleaning_model",
  "method_version": "0.11.1",
  "entry_points": [
    "src/moma_lineage_analysis/trajectory_cleaning/cli.py train",
    "src/moma_lineage_analysis/trajectory_cleaning/cli.py predict",
    "src/moma_lineage_analysis/trajectory_cleaning/prepare_verified_training_data.py",
    "src/moma_lineage_analysis/trajectory_cleaning/promote_best_model.py",
    "src/moma_lineage_analysis/trajectory_cleaning/plot_training_results.py",
    "src/moma_lineage_analysis/app/Moma_lineage_streamlit.py (AI cleaning panel)",
    "scripts/qc_raw_train_data.py (raw export QC)",
    "scripts/stage_verified_labels.py (label staging: R002 policy / --division-mode effective)",
    "src/moma_lineage_analysis/trajectory_cleaning/dataset_store.py (SQLite store: schema/import/export/verify/lineage loading)",
    "scripts/dataset_store.py (store build/verify/export/backup)",
    "src/moma_lineage_analysis/trajectory_cleaning/cli.py --dataset-lineage (store-direct training)",
    "scripts/sweep_temporal_context.py (R016 temporal-context screen/repeat/lock/evaluate)",
    "src/moma_lineage_analysis/trainset_review/cli.py (R018 app launcher)",
    "src/moma_lineage_analysis/trainset_review/streamlit_app.py (R018 review UI)",
    "src/moma_lineage_analysis/trainset_review/store.py (R018 guarded review backend; R019 dataset-version removal)",
    "scripts/remove_dataset_versions.py (R019 duplicate dataset-version removal)",
    "src/moma_lineage_analysis/model_registry.py (shared selectable checkpoint registry)"
  ],
  "inputs": [
    "datasets/verified_20260916/dataset_manifest.json",
    "datasets/verified_20260916/splits_no_longruns.json",
    "datasets/verified_20260916/longrun_excluded_trajectories.json",
    "datasets/verified_20261013/dataset_manifest.json",
    "datasets/verified_20261016/dataset_manifest.json",
    "datasets/verified_merged_20261016/merge_manifest.json",
    "datasets/verified_20261017/dataset_manifest.json",
    "datasets/verified_merged_20261017/merge_manifest.json",
    "datasets/verified_20261029/dataset_manifest.json",
    "datasets/verified_20261029/qc_report.json",
    "datasets/verified_20261029/staging_report.json",
    "datasets/verified_20261029/rename_map.json",
    "datasets/verified_merged_20261029/merge_manifest.json",
    "runs/merged_20261029/cnn/splits.json",
    "runs/context_20261029/protocol.json",
    "runs/context_20261029/environment.json",
    "runs/context_20261029/preservation_snapshot.json",
    "datasets/store/lineage.sqlite3",
    "datasets/store/SCHEMA.md",
    "datasets/store/retired_csv_manifest_20261017.json",
    "datasets/manifest.json",
    "docs/PLAN_DATASET_STORE.md",
    "docs/TRAINSET_REVIEW.md",
    "src/moma_lineage_analysis/trainset_review/cli.py",
    "src/moma_lineage_analysis/trainset_review/streamlit_app.py",
    "src/moma_lineage_analysis/trainset_review/store.py",
    "src/moma_lineage_analysis/model_registry.py",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/merged_20261029/cnn/release_manifest.json",
    "runs/trainset_review_implementation/pytest_full.txt",
    "runs/trainset_review_implementation/launcher_help.txt",
    "runs/trainset_review_implementation/app_tests.txt",
    "runs/trainset_review_implementation/integration_tests.txt",
    "runs/store_cleanup_20261031/before_state.json",
    "runs/store_cleanup_20261031/after_removal_state.json",
    "runs/store_cleanup_20261031/finalize_results.json",
    "runs/store_cleanup_20261031/training_frame.json",
    "runs/store_cleanup_20261031/verify_after_finalize.json",
    "runs/reviewed_20261031/cnn/metrics.json",
    "runs/reviewed_20261031/cnn/splits.json",
    "runs/reviewed_20261031/resnet/metrics.json",
    "runs/reviewed_20261031/lstm/metrics.json",
    "runs/reviewed_20261031/model_comparison.json",
    "runs/reviewed_20261031/production_comparison.json",
    "runs/reviewed_20261031/training_summary.json"
  ],
  "analysis_unit": "observation (one cell at one time point); division event for event-level metrics",
  "independent_unit": "trajectory (trajectory_key); known limitation: not FOV-held-out",
  "populations": {
    "train": "838 trajectories / 279,548 rows (24,538 division, 1,376 excluded positives)",
    "validation": "179 trajectories / 59,643 rows (5,220 division, 325 excluded positives)",
    "test": "180 trajectories / 59,977 rows (5,274 division, 321 excluded positives)",
    "note": "R015 and R016 use identical frozen lineage/split; 399,168 rows / 1,197 trajectories, eight sources, no new data or split in R016. Historical R010 804/172/173 split remains in its round sections.",
    "r016_validation": "179 trajectories / 59,643 rows / 325 exclusions; 74 trajectories <197 frames (41.3%); common interior radius 98 = 28,096 rows / 147 exclusions",
    "r016_evaluation": "Planned after lock: full R015 test 180 trajectories, 103-key common historical test, seven new-batch test trajectories; previously reported, descriptive not fresh external holdout",
    "r018_review": "No new training corpus or model metrics. Existing frozen lineage/split remain the data source; app mutations change labels/membership only under explicit user authority. Newly finalized review datasets are new lineage IDs and must be verified before training.",
    "r019_train": "806 trajectories / 269,699 rows (23,499 division, 1,196 excluded positives)",
    "r019_validation": "173 trajectories / 58,007 rows (5,029 division, 270 excluded positives)",
    "r019_test": "173 trajectories / 58,109 rows (5,137 division, 274 excluded positives)",
    "r019_note": "Composition of the five reviewed_20261031_* frozen lineages: 385,815 rows / 1,152 trajectories / 33,665 divisions / 1,740 exclusions / 49 legacy overlap rows, eight sources, zero duplicate observation rows. The reviewed population differs from R015 (45 trajectories permanently deleted, many labels edited) so per-round metrics are not comparable."
  },
  "metrics": {
    "POINT_F1": "point-level precision/recall/F1 per label at calibrated threshold",
    "POINT_AP": "average precision per label",
    "DIVISION_EVENT_F1": "event-level P/R/F1, tolerance +/-1 frame",
    "DIVISION_EVENT_MAE": "mean absolute frame error over matched events",
    "DIVISION_RECALL_TAIL": "division recall in last trajectory quintile (rel. pos. >= 0.8)",
    "DIVISION_RECALL_HEAD": "division recall in first trajectory quintile (rel. pos. <= 0.2)",
    "LAST_EVENT_MISS_RATE": "fraction of trajectories whose last division event is missed"
  },
  "parameters": {
    "MODEL_CONFIG": "input 5, output 2, hidden 64, layers 4, kernel 5, dropout 0.15 (pre-R002 checkpoints: input 2)",
    "TRAIN_CONFIG": "epochs 100, batch 16, lr 1e-3, wd 1e-4, gamma 1.5, max_pos_weight 50, patience 15, seed 42",
    "DIVISION_TOLERANCE": 1,
    "THRESHOLD_GRID": "np.linspace(0.05, 0.95, 91), validation-only",
    "SPLIT_FRACTIONS": "validation 0.15, test 0.15, seed 42",
    "NORMALIZER_SCALE": "clip 12, time_step_scale 100, min_length_scale 0.05, MAD 1.4826, feature_compatibility extended/legacy",
    "DIVISION_DROP_RATIO": 0.75,
    "EXCLUSION_RUN_FILTER": "drop trajectories whose longest contiguous exclusion run >= 5 frames (R004)",
    "R002_LABEL_POLICY": "coalesced_previous labels dropped; edp-only rows kept; needs_review/shifted_previous kept",
    "EFFECTIVE_LABEL_POLICY": "division_point=effective_division_point for curated batches; raw division/exclusion overlaps become zero; resolver audit statuses retained",
    "R016_CONTEXT": "normalized-input dense RF 17/33/49/65/97/121/161/197; four-entry schedules; CNN k5 63,938 params; primary ResNet k3 100,354; old ResNet k5 RF121 165,890 separate control; optional conv_dilations backward compatible; hidden 64 fixed",
    "R016_DIAGNOSTICS": "common interior radius 98 vs boundary; trajectories <197 vs >=197; finalists matched on common coverage; all-run validation diagnostics planned post-lock",
    "R018_MUTATION": "label edits autosave globally across referencing versions; time_h/spine_length immutable; permanent whole-trajectory deletion with exact-key confirmation and impact display; per-mutation online backup in review_backups; strict global CAS",
    "R018_FINALIZE": "new dataset ID; R004 membership-only filter; human attestation; source-lineage provenance; byte-exact native export sha256; core verify allowed only for matching finalize audit/provenance/hashes",
    "AI_MODEL_REGISTRY": "R019 production default (sha256 816a8b39...); R015 selectable non-production (sha256 975329...) in shared registry and review app",
    "R019_REMOVAL": "remove whole dataset versions only when every trajectory stays referenced by another version; review-revision CAS; online snapshot + remove_dataset_versions audit event; never delete observations/trajectories",
    "R019_FINALIZE": "five per-batch finalizations with user_attested_2026-10-31; R004 membership filter applied to each new version; observed dropped keys baseline_0_10h::FOV_28_CH_2, curated_20261016::FOV_53_CH_11, curated_20261017::FOV_14_CH_14",
    "R019_TRAINING": "unchanged TrainConfig/ModelConfig/seed-42 per-source split; five repeated --dataset-lineage composition; cnn/resnet/lstm epochs 31/20/29 (best 16/5/14); validation winner CNN mean F1 0.901205"
  },
  "statistics": {
    "model_selection": "maximize mean of validation division-event F1 and validation exclusion F1; test never used for selection",
    "uncertainty": "R015 single-seed descriptive results; no significance test or bootstrap; common historical panel 103 trajectories, new-batch panel 7 / 13 exclusions; R010-vs-R008 historical bootstrap remains separate",
    "seed": 42,
    "r016_selection": "top3 per family on seed42 validation; repeat selected contexts seeds43/44; lock one by three-seed mean before test; report all3 winner seeds, no best seed; 29 runs completed; locked RF17 for CNN (0.885582 +/- 0.004732) and primary ResNet (0.887547 +/- 0.006512); best among screened/shortlisted schedules, not universal optimum",
    "r016_uncertainty": "seed42 shortlist reused in final mean -> selection optimism; three-seed SD = optimization noise, not data/FOV uncertainty; irregular grid no smooth interpolated optimum; original controls single-seed descriptive; no significance/generalization claim",
    "r018_selection": "No model training or architecture selection. Review preview uses released checkpoints; R010 remains production default and R015 is selectable but non-production. Evaluation badges describe recorded split membership only.",
    "r019_selection": "maximize mean of validation division-event F1 and validation exclusion F1; test never used for selection; CNN winner 0.901205 vs BiLSTM 0.884530 vs ResNet 0.860075",
    "r019_comparison": "descriptive same-row panels with own thresholds: common test unseen in both rounds 35 trajectories / 10,317 rows / 44 exclusions (R019 CNN exclusion F1/AP 0.8395/0.9034 vs production 0.8250/0.8618); new 2026-10-29 batch 7 trajectories / 14 exclusions (0.7586/0.8654 vs 0.3636/0.4804); full R019 test contaminated for production (130/173 keys in its train/validation). Single seed, no bootstrap or significance test; no promotion"
  },
  "plots": {
    "LOSS_CURVES": "training_loss_curves.png",
    "F1_COMPARISON": "model_f1_comparison.png",
    "LABEL_BALANCE": "dataset_label_balance.png"
  },
  "outputs": [
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/verified_20260916/cnn/",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/verified_20260916/resnet/",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/verified_20260916/lstm/",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/verified_20260916/r002_model_comparison.json",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/verified_20260916/r002_heldout_full_newdataset.json",
    "tests/trajectory_cleaning/R002_ROUND_REPORT.md",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt (promoted R010 CNN 2026-10-17; sha256 6e7c5a58...; previous R008 CNN backed up under promotion_backups/20261017T124420.980675Z)",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion.json",
    "datasets/verified_20261013/verified_training_data.csv (R006 batch; retired in R013, restore via store export)",
    "datasets/verified_20261013/staging_report.json",
    "datasets/verified_20261013/qc_report.json",
    "datasets/verified_20261016/verified_training_data.csv (R007 batch; retired in R013, restore via store export)",
    "datasets/verified_20261016/staging_report.json",
    "runs/merged_20261016/cnn/ (R008)",
    "runs/merged_20261016/resnet/ (R008)",
    "runs/merged_20261016/lstm/ (R008)",
    "runs/merged_20261016/r008_model_comparison.json",
    "datasets/verified_20261017/verified_training_data.csv (R009 batch; retired in R013, restore via store export)",
    "datasets/verified_20261017/staging_report.json",
    "datasets/verified_20261017/qc_report.json",
    "runs/merged_20261017/cnn/ (R010)",
    "runs/merged_20261017/resnet/ (R010)",
    "runs/merged_20261017/lstm/ (R010)",
    "runs/merged_20261017/r010_model_comparison.json",
    "runs/merged_20261017/figures/",
    "runs/merged_20261017_store_repro/{cnn,resnet,lstm}/ (R012 reproduction)",
    "runs/merged_20261017_store_repro/reproduction_report.json",
    "datasets/store/lineage.sqlite3 (eight frozen dataset versions after R014/R015 ingestion)",
    "datasets/verified_20261029/ (R014 provenance and staged files)",
    "datasets/verified_merged_20261029/ (R015 merge provenance)",
    "runs/audit_20261029/preflight.json",
    "runs/audit_20261029/build_report.json",
    "runs/audit_20261029/raw_independent_audit.json",
    "runs/audit_20261029/retirement_report.json",
    "runs/merged_20261029/verification_report.json",
    "runs/merged_20261029/store_verification.json",
    "runs/merged_20261029/store_backup.json",
    "runs/merged_20261029/pytest_full.txt",
    "runs/merged_20261029/{cnn,resnet,lstm}/ (R015)",
    "runs/merged_20261029/model_comparison.json",
    "runs/merged_20261029/cross_eval_vs_production.json",
    "runs/merged_20261029/figures/",
    "datasets/store/backups/lineage_20261017T132240.874489Z.sqlite3.gz",
    "datasets/store/backups/lineage_20261028T175033.995401Z.sqlite3.gz (final corrected R014/R015 snapshot; sha256 388246cfa0299e3f1a935f0a66e4d786f383bfd41538d23b206c0d646bdbc10c; 7,054,400 bytes)",
    "runs/context_20261029/protocol.json (R016 registered before training)",
    "runs/context_20261029/screen_results.json",
    "runs/context_20261029/repeat_results.json",
    "runs/context_20261029/selection_lock.json",
    "runs/context_20261029/evaluation.json",
    "runs/context_20261029/summary.json",
    "runs/context_20261029/validation_diagnostics.json",
    "runs/context_20261029/control_evaluation.json",
    "src/moma_lineage_analysis/trainset_review/ (R018 app source)",
    "docs/TRAINSET_REVIEW.md (R018 user guide)",
    "src/moma_lineage_analysis/model_registry.py (R010 production + selectable R015)",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/merged_20261029/cnn/ (packaged R015 release)",
    "runs/trainset_review_implementation/ (app tests, launcher help, full-suite evidence)",
    "scripts/remove_dataset_versions.py (R019 duplicate-version admin removal)",
    "datasets/store/lineage.sqlite3 (ten versions after R019: five working originals + five reviewed_20261031_* training-ready)",
    "datasets/store/backups/lineage_20261030T181226.223866Z.sqlite3.gz (pre-removal backup; sha256 bad95fa0...)",
    "runs/store_cleanup_20261031/ (before/after states, finalize results, composed training frame, store verification)",
    "runs/reviewed_20261031/{cnn,resnet,lstm}/ (R019)",
    "runs/reviewed_20261031/model_comparison.json",
    "runs/reviewed_20261031/production_comparison.json",
    "runs/reviewed_20261031/training_summary.json",
    "runs/reviewed_20261031/figures/",
    "runs/reviewed_20261031/verification/ (independent verification)",
    "runs/reviewed_20261031/production_promotion.json (R020 promotion record)",
    "runs/reviewed_20261031/promotion_previous_R010.json (archived R010 promotion manifest)",
    "src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261030T184217.243155Z/ (R010 CNN rollback artifacts + archived promotion.json)"
  ],
  "validation_gates": [
    "86-test unittest battery in tests/trajectory_cleaning (run 2026-09-17: OK)",
    "validate_methods_file.py on this contract: PASS",
    "R002 retraining on corpus v2: division tail-quintile recall 0.988-0.994, last-event miss rate 0.000-0.012 (test partition)",
    "R002 backward compatibility: pre-R002 checkpoint loads in legacy mode and reproduces original predictions",
    "R009 batch: raw QC 16/16 files pass; sha256 pin verified; 1-epoch CPU training smoke with --require-human-verified OK; pytest 70 passed / 2 skipped",
    "R010 retraining on the 4-source merged corpus: cnn/resnet/lstm trained on GPU via packaged CLI; independent verifier corpus 11/11 PASS and training artifacts 8/8 PASS; canonical splits reproduced; source hashes match",
    "R011 promotion of the R010 CNN: validation-only rule re-applied; hash-verified copy of runs/merged_20261017/cnn/best_model.pt (sha256 6e7c5a58...); previous R008 CNN + artifacts backed up; promotion.json rewritten; load-checkpoint + predict smoke produced finite outputs; contract-integrity hash pin updated and PASS (6 passed / 2 skipped)",
    "R012 dataset store: phase-1 independent audit PASS (schema v2, 375,348 observations / 1,152 trajectories, six versions; counts, canonical hashes, legacy CSV sha256 and byte-identical exports all PASS; backup restore and rebuild determinism PASS); store-direct training records dataset_lineage; R010 reproduction has byte-identical splits for all three models, bit-identical resnet/lstm weights and metrics, cnn thresholds/discrete metrics identical with AP drift <=8.95e-5 matched by a CSV-path control run (GPU nondeterminism); no corpus CSV deleted",
    "R015 model comparison: CNN/ResNet/BiLSTM epochs 40/21/23 (best 25/6/8), validation mean F1 0.888577/0.860210/0.832268 selects CNN; test division-event F1 0.995439/0.995236/0.992091, exclusion F1 0.802568/0.766871/0.720247, AP 0.877942/0.852725/0.723745; common panel production/R015cnn exclusion F1 0.8232/0.8099 and AP 0.8999/0.8791; seven-key new-batch exclusion F1 0.3750/0.4242 and AP 0.4883/0.6273; no significance/bootstrap and no replacement support",
    "R015 artifact verification: final verifier PASS_WITH_DOCUMENTED_CAVEATS; 441 recorded checks (317 data / 35 retirement / 65 training and cross-evaluation / 24 final comparison and integrity), failures empty; additional packaged retirement verify 18 checks; production R010 CNN unchanged",
    "R016 completed: 29 runs on fixed R015 lineage/split at hidden64; validation-only lock selected RF17 for CNN and primary ResNet k3 before any test use; locked three-seed test means (CNN 0.7972 F1 / 0.8659 AP; ResNet 0.8043 / 0.8759) remain below the R010 production control (0.8183 / 0.8960; common panel 0.8999); matched radius-98 interior beats boundary for every seed; nominal frames map to unequal physical spans (RF17 ~0.803 h full coverage vs RF197 ~9.831 h / 47.1% centers); per-source exclusion support 104 down to 2; Lead audit 438 checks PASS after an audit-only RF formula fix; no promotion",
    "R016 verification: PASS_WITH_DOCUMENTED_CAVEATS, 21/21 checks, no failures, 183 replay checks per GPU; GPU0 replay exact on all discrete metrics; one CNN RF17 seed42 validation probability within 1e-5 of the 0.87 exclusion threshold flips cross-device (precision -0.002738, F1 -0.001255) and AP ties differ <=1.6e-4 across devices; production preserved with promotion_performed=false",
    "R018 review app: final pytest 388 passed / 4 skipped / 59 subtests (81.96s, pytest_full_final.txt); independent verification PASS with zero open defects (backend 54, registry 25, scoped helper 86, AppTest 4, opaque-ID/cache 13); installed launcher --help exit 0; live DB sha 2315a601... and production 6e7c5a58... byte-identical; fresh TEMP browser retest confirmed wrong-key rejection and exact-key deletion with snapshot after UI-DELETE-CONFIRM-001 fix; navigation/finalize/stale-view and auto/CUDA device fixes re-tested",
    "R019 store cleanup and retraining: three duplicate merged versions removed at revision 562 (3,345 memberships, zero orphaned trajectories, observations/trajectories unchanged at 387,217/1,155); five reviewed lineages finalized at revisions 563-567 (385,815 rows / 1,152 trajectories, three R004 membership drops, database-native export hashes); store verify PASS with 5 WORKING + 5 PASS; split 806/173/173 identical across models and reconstructed exactly; validation winner CNN 0.901205; test div-event/exclusion F1 0.9966/0.7976 (AP 0.8783); descriptive same-row panels do not support a promotion claim; production R010 CNN unchanged; full pytest 429 passed / 4 skipped / 59 subtests (runs/reviewed_20261031/pytest_full.txt); independent verification PASS 29/29 checks, 0 failures, 3 documentation-level discrepancies",
    "R020 promotion (user decision 2026-10-31): promote_best_model re-applied the validation-only rule (cnn 0.901205 / BiLSTM 0.884530 / ResNet 0.860075), required identical split manifests, copied and hash-verified the four R019 CNN artifacts (checkpoint sha256 816a8b39...); previous R010 CNN artifacts (6e7c5a58...) plus its archived promotion.json backed up under promotion_backups/20261030T184217.243155Z; load-checkpoint/predict smoke finite; registry, contract-integrity pin, AI-panel label and docs updated; no data/model/metric change; post-promotion full pytest 430 passed / 3 skipped / 59 subtests (runs/reviewed_20261031/pytest_full_post_promotion.txt)",
    "R014 data and R015 store phase: raw audit PASS_CURRENT_BATCH_WITH_WARNINGS (42 needs_review sources suppressed by effective policy; separate >2**53 audit-decision ID precision issue outside current batch); 317 independent data/store checks PASS; store 401,076 observations / 1,200 keys / 8 versions / 4,680 memberships / 21 ingest provenance rows / 14,458,880 bytes; two new corpus CSVs retired after verification; eight retirement ledger entries; post-retirement verify 18 checks PASS; final corrected backup sha256 388246cfa0..., 7,054,400 bytes; full pytest 230 passed / 4 skipped / 59 subtests",
    "R013 retirement: user sign-off after phase 2; strict store verify PASS with all six legacy CSVs present; the six pinned corpus CSVs were then deleted (only those files); normal store verify PASS with legacy_csv_state=retired for all six versions and exports still byte-identical to the pinned sha256; provenance JSONs, per-corpus manifests, staged provenance, datasets/manifest.json and the store backup remain; retirement record with per-file sha256 and restore commands at datasets/store/retired_csv_manifest_20261017.json; full pytest tests -q 96 passed / 3 skipped (the third skip is the standalone training-experiment script tests/test_cell_lineage_sperate.py, which now module-level skips under pytest instead of training and blocking on plt.show())"
  ]
}
```
<!-- METHOD_MANIFEST_END -->

## Layer 3 — Analysis Rounds and Reconciliation

### Analysis-round registry

| Round | Date | Type | Question or change | Intended method version | Status | Primary artifacts |
|---|---|---|---|---|---|---|
| `R001` | 2026-09-16 | Retrospective documentation | Initialize the living Methods contract; consolidate and re-run test suite | 0.1.0 | Closed | `METHODS_TRAJECTORY_CLEANING_MODEL.md`, `AUDIT_REPORT.md`, 78-test battery OK |
| `R002` | 2026-09-16 | Model improvement | Fix collapsed division recall at trajectory ends on the 21.6 h RDM-Gluc dataset; extend features and retrain on merged corpus v2 | 0.2.0 | Closed | corpus `verified_20260916/`, retrained cnn/resnet/lstm, this contract |
| `R003` | 2026-09-16 | Robustness | Train only on the RDM-Gluc dataset to test whether condition-matched training beats the mixed corpus | 0.2.0 | Closed | `rdm_only_20260916/`, `r003_model_comparison.json` |
| `R004` | 2026-09-17 | Model improvement | Remove trajectories with long continuous exclusion runs (analysis-driven cell-cycle removal, not growth noise) from training; measure exclusion F1 recovery | 0.3.0 | Closed | filtered corpus `verified_training_data_no_longruns.csv`, retrained cnn/resnet/lstm in `no_longruns_20260916/`, promoted R004 CNN |
| `R005` | 2026-10-13 | Repository transfer (no methodological change) | Relocate code, R004 corpus, promoted checkpoint, and test suite from the legacy `Moma_cellPose` workspace into the standalone `moma_lineage_analysis` package (src layout, editable install); reconcile declared paths | 0.3.1 | Closed | `src/moma_lineage_analysis/trajectory_cleaning/cli.py`, `datasets/verified_20260916/`, `tests/trajectory_cleaning/`, this contract |
| `R006` | 2026-10-13 | Data addition (no methodological change) | Verify 35 newly curated single-trajectory exports; stage with `effective_division_point` as the final true division label (user decision); rename 11 duplicate keys with a `20261013_` prefix (user decision); build a new attested batch corpus; merging into the production corpus deferred | 0.3.2 | Closed | `datasets/verified_20261013/`, `datasets/verified_20261013/qc_report.json`, `scripts/qc_raw_train_data.py`, `scripts/stage_verified_labels.py`, this contract |
| `R007` | 2026-10-16 | Data addition (no methodological change) | Verify 48 newly curated single-trajectory exports from the rotated `raw_train_data/` (previous batch's raw exports removed by user; its QC report archived into `verified_20261013/`); same pipeline: effective division labels, `20261016_` prefix for 18 keys duplicating R004-corpus trajectories, attested batch corpus; merging deferred | 0.3.3 | Closed | `datasets/verified_20261016/`, this contract |
| `R008` | 2026-10-16 | Retraining (model improvement) | Merge R004 production corpus with both curated batches (R004 filter applied to batches: 2 trajectories dropped); retrain cnn/resnet/lstm with unchanged config on GPU via packaged CLI; validation-only rule selected BiLSTM (mean F1 0.8987); user overrode and promoted the CNN (best test exclusion F1 0.8452, tail recall 0.9959, smallest/fastest) | 0.4.0 | Closed | `datasets/verified_merged_20261016/`, `runs/merged_20261016/`, `r008_model_comparison.json`, promoted `best_model.pt` (sha256 `3e796c9e…`), this contract |
| `R009` | 2026-10-17 | Data addition (no methodological change) | Audit 16 newly added single-trajectory exports from `datasets/raw_train_data/`; stage with `effective_division_point` as the final division label (529 divisions, overlap 0); rename 9 keys duplicating R004-corpus trajectories with a `20261017_` prefix; build an attested batch corpus; merging into the production corpus deferred | 0.4.1 | Closed | `datasets/verified_20261017/`, this contract |
| `R010` | 2026-10-17 | Retraining (model improvement) | Clean the temporary analysis files, then merge all four verified sources into `verified_merged_20261017` (R004 filter applied to the three batches: 3 trajectories dropped) and retrain cnn/resnet/lstm with unchanged config on GPU via the packaged CLI; validation-only rule selects the cnn (mean F1 0.8987); no promotion performed at round close (the R010 CNN was promoted on 2026-10-17 in round R011) | 0.5.0 | Closed | `datasets/verified_merged_20261017/`, `runs/merged_20261017/`, `r010_model_comparison.json`, figures, `verification_report.json`, this contract |
| `R011` | 2026-10-17 | Deployment (promotion; no methodological change) | Promote the R010 CNN to production: the tool re-applies the validation-only rule, backs up the previous R008 CNN, hash-verifies the four new artifacts, rewrites `promotion.json` and updates the checkpoint pins/labels; no model, metric, parameter, or corpus changed | 0.5.1 | Closed | `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt` (sha256 `6e7c5a58…`), `promotion.json`, `promotion_backups/20261017T124420.980675Z`, this contract |
| `R012` | 2026-10-17 | Data layer (no model/metric change) | Build and verify the local SQLite dataset store (six frozen versions, de-duplicated observations, canonical + legacy hashes, byte-identical exports), switch the packaged training CLI to read the store directly by dataset lineage id, and record lineage ids/content hashes in `metrics.json`; reproduce R010 from `verified_merged_20261017@v1`; delete nothing yet (legacy CSVs retained for phase 3) | 0.6.0 | Closed | `datasets/store/lineage.sqlite3`, `datasets/store/SCHEMA.md`, `docs/PLAN_DATASET_STORE.md`, `runs/merged_20261017_store_repro/`, this contract |
| `R013` | 2026-10-17 | Retirement (no model/metric change) | Retire the six legacy corpus CSVs after explicit user sign-off on the phase-2 report; strict store verification first, then delete only `datasets/verified_*/verified_training_data*.csv`, then verify again (all versions `legacy_csv_state: retired`, exports still byte-identical to the pins); record a per-file retirement manifest with restore commands; the store becomes the sole training source | 0.7.0 | Closed | `datasets/store/retired_csv_manifest_20261017.json`, `datasets/README.md`, `AGENTS.md`, `PLAN_DATASET_STORE.md`, this contract |
| `R014` | 2026-10-29 | Data addition | Audit 48 exports; effective-label staging (2,562 divisions / 98 exclusions / overlap 0); date-prefix eight reused identifiers without inferring biological identity; preserve all 48 (max exclusion run 3); attest and ingest new batch | 0.8.0 | Closed (raw audit warnings documented) | `datasets/verified_20261029/`, `runs/audit_20261029/` |
| `R015` | 2026-10-29 | Retraining | Extend frozen R010 lineage with R014 (399,168 rows / 1,197 trajectories); retrain CNN/ResNet/BiLSTM with unchanged configuration; select by validation only, compare on paired populations, verify/retire redundant CSV materializations | 0.8.0 | Closed (no promotion) | `datasets/verified_merged_20261029/`, `runs/merged_20261029/` |
| `R016` | 2026-10-29 | Exploratory temporal-context tuning | Screen eight dense dilation schedules per family at fixed hidden64 on frozen R015 split; repeat shortlisted schedules across seeds42/43/44; lock contexts before test and report all winner seeds | 0.9.0 | Closed (RF17 locked for CNN and primary ResNet; verification PASS_WITH_DOCUMENTED_CAVEATS; no promotion) | `runs/context_20261029/` |
| `R018` | 2026-10-29 | Operational app / data-editing integration | Add the independent `moma-trainset-review` Streamlit app for guarded label review, permanent trajectory deletion, working-version invalidation and new-ID finalization; expose packaged selectable R015 alongside unchanged production R010 | 0.10.0 | PASS (final suite 388 passed / 4 skipped / 59 subtests; independent verification PASS, zero open defects; no promotion) | `src/moma_lineage_analysis/trainset_review/`, `docs/TRAINSET_REVIEW.md`, `runs/trainset_review_implementation/` |
| `R019` | 2026-10-31 | User-requested store cleanup + retraining | Remove the three duplicate `verified_merged_*` dataset versions (every trajectory is still held by the five batch versions), finalize the five reviewed batches as `reviewed_20261031_*` training-ready lineages with the R004 membership filter, and retrain cnn/resnet/lstm on their composition with unchanged configuration; compare with production R010 on identical rows | 0.11.0 | PASS (independent verifier 29/29 checks, 0 failures, 3 documentation-level discrepancies; promotion followed in R020) | `runs/store_cleanup_20261031/`, `runs/reviewed_20261031/`, `datasets/store/lineage.sqlite3`, this contract |
| `R020` | 2026-10-31 | Deployment (promotion; user decision) | Promote the R019 CNN to production: the tool re-applies the validation-only rule, backs up the previous R010 artifacts and its promotion manifest, hash-verifies the four new artifacts, and the registry/pins/labels/docs are updated | 0.11.1 | Closed (validation-selected cnn; previous R010 CNN backed up; no data/model/metric change) | `src/moma_lineage_analysis/models/trajectory_cleaning_run/` (+ `promotion.json`, `promotion_backups/20261030T184217.243155Z/`), `runs/reviewed_20261031/production_promotion.json`, this contract |

### Current round plan

Request (R020): promote the newly trained R019 CNN to the main app (user:
"promote new cnn model to the main app"). R020 is deployment only.

- Candidate: `runs/reviewed_20261031/cnn/` (validation mean F1 0.901205; test
  division-event F1 0.996583, exclusion F1 0.797642, AP 0.878254).
- Promotion: `promote_best_model` re-applies the validation-only rule
  (cnn > BiLSTM > ResNet), requires identical split manifests, copies and
  hash-verifies the four artifacts, and backs up the previous R010 CNN
  (checkpoint `6e7c5a58...`) plus its archived `promotion.json` under
  `promotion_backups/20261030T184217.243155Z/`.
- New production checkpoint sha256 `816a8b39...`; `promotion.json` records the
  hash-verified copy and scores. Registry labels, Streamlit AI-panel test,
  contract-integrity pin, README, AGENTS.md, `datasets/manifest.json` and this
  contract are updated; load-checkpoint/predict smoke and the full suite are run.
- No corpus, dataset version, split, threshold, training run or metric changes.

**R019 plan (closed):**

Request (R019): remove the three duplicate `verified_merged_*` dataset versions
from the training store and retrain the three architectures on the five
reviewed batch versions (user: "3 verified_merged_xxx data set are duplicate
trajectories from the no merged dataset, remove them from database; remain 5
dataset are re-cheched, use them to train new model weights").

- Cleanup: precondition proof that merged membership is a strict subset of the
  five batch memberships; single-transaction removal of three versions and 3,345
  memberships via `remove_dataset_versions` at review revision 562 with snapshot
  and audit event; observations/trajectories unchanged; manifest entries marked
  `store_removed`; pre-removal backup retained.
- Training inputs: five `reviewed_20261031_*` frozen lineages finalized with
  `user_attested_2026-10-31` and the R004 membership filter (three dropped keys,
  nothing deleted); composition 385,815 rows / 1,152 trajectories, eight sources,
  zero duplicate observation rows.
- Training: unchanged TrainConfig/ModelConfig; repeated `--dataset-lineage`
  composition; seed-42 per-source split 806/173/173; sequential cnn/resnet/lstm
  on GPU; validation-only selection; test reported for acceptance only.
- Comparison: descriptive same-row panels versus production R010 with each
  checkpoint's own thresholds; panels are small (44 and 14 exclusion positives),
  no bootstrap/significance, and production-visible overlap is counted rather
  than hidden.
- Deployment: no promotion; production remains the R010 CNN (sha256
  `6e7c5a58...`). Independent verification of cleanup, finalization, splits and
  metrics is required before this round is closed.

**R018 plan (closed, historical):**

Request (R018): productize an independent training-dataset review app for
human label review, explicit permanent deletion and new-ID finalization, without
changing model training, model selection or production deployment. R017
latest-only outputs remain unregistered and are not used here. R001-R016 history
and the lineage-construction contract remain unchanged.

- Source and launch: nine Python modules under
  `src/moma_lineage_analysis/trainset_review/`; console command
  `moma-trainset-review` plus `python -m moma_lineage_analysis.trainset_review`.
  Default store `datasets/store/lineage.sqlite3`, overridden by
  `MOMA_TRAINSET_STORE`; environment Scripts is not on PATH. Read-only opening
  must not create review tables.
- Mutations: label edits autosave globally across referencing versions;
  measurements `time_h`/`spine_length` remain immutable. Permanent whole-
  trajectory deletion removes observations/all memberships and has no in-app
  restore; exact-key confirmation and impact display are mandatory. Every
  mutation is global-revision CAS-guarded and pre-backed up under
  `review_backups/` beside the database.
- Working/finalized data: reviewed frozen versions become working (`frozen=0`)
  with original pins preserved in review state; training resolution refuses them.
  Finalization creates a new dataset ID with human attestation, R004 membership
  filter only, source-lineage provenance and a byte-exact native export hash; core
  verify accepts only matching finalize audit/provenance/hashes. Historical CSV
  import is refused after review so deleted data cannot resurrect.
- Models: R010 CNN remains default production; packaged R015 CNN is selectable
  but non-production. Preview thresholds default to calibrated values and Apply
  is explicit. Evaluation badges describe recorded split membership; agreement
  is not a fresh holdout result. Preview cache includes frame content,
  checkpoint bytes, device, thresholds and revision.
- Acceptance: no live DB mutation during implementation; final full suite
  388 passed / 4 skipped / 59 subtests in 81.96 s; independent backend 54,
  registry 25, scoped helper 86, AppTest 4 and opaque-ID/cache 13 checks pass;
  launcher help exits 0. Independent verification is PASS with zero open defects.
  A fresh TEMP browser retest confirmed wrong-key rejection and exact-key
  deletion with snapshot after the UI-DELETE-CONFIRM-001 form fix. Navigation,
  finalize selection, stale-view protection and auto/CUDA device resolution were
  also re-tested. Live DB/production hashes remained unchanged.
- Deployment: no training, promotion, model-registry production change or
  historical-lineage byte-identity claim.


### Declared-versus-actual reconciliation

| Manifest path or method step | Declared | Actual evidence | Status | Resolution | Affected outputs |
|---|---|---|---|---|---|
| Methods contract initialized | Three-layer contract | File created from governed template | Match | Populated; strictly validated | `METHODS_TRAJECTORY_CLEANING_MODEL.md` |
| Test consolidation | Six test modules under `tests/trajectory_cleaning/` | Moved; REPOSITORY_ROOT depth fixed; `__init__.py` added | Match | 78 tests pass (unittest, 2026-09-16) | `tests/trajectory_cleaning/` |
| Docs test references | Docs point at moved modules | `docs/trajectory_cleaning_model.md` lines updated to `tests.trajectory_cleaning.*` | Match | -- | `docs/trajectory_cleaning_model.md` |
| Declared thresholds vs promoted checkpoint | division 0.82 / excluded 0.87 | `metrics.json` excluded threshold 0.8700000047683716 (float32) | Match | float32 representation documented | `trajectory_cleaning_run/verified_20260718/cnn/metrics.json` |
| R002 corpus construction | 4 sources + supersession, RDM-Gluc reconciled labels | `verified_20260916` manifest: 1,070 trajectories / 318,745 rows; supersession omitted 83 trajectories; 78 invalid feature rows dropped | Match | label policy documented in Study design | `trajectory_cleaning_run/verified_20260916/` |
| R002 feature extension | 5-channel extended layout, legacy compatibility | `TrajectoryNormalizer().transform` returns (n, 5); pre-R002 checkpoint loads with `feature_compatibility='legacy'` and reproduces original predictions | Match | backward compatibility verified | `trajectory_cleaning/data.py` |
| R002 position metrics | tail/head recall + last-event miss rate in `evaluate_model` | metrics computed on all three retrained architectures and on the old promoted checkpoint | Match | -- | `trajectory_cleaning/training.py` |
| R002 failure-mode resolution | late-trajectory division recall restored | last-quintile recall 0.086 -> 0.988 (cnn) / 0.987 (resnet) / 0.994 (lstm); last-event miss rate 0.573 -> 0.006 / 0.007 / 0.000 | Match | acceptance target met on full new dataset | `trajectory_cleaning_run/verified_20260916/` |
| R004 corpus filtering | drop trajectories with exclusion run 5+ frames | 17 of 1,070 trajectories removed (6,428 rows); kept 1,053 trajectories / 312,317 rows; flagged keys in `longrun_excluded_trajectories.json` | Match | filter rule frozen in parameter registry | `trajectory_cleaning_run/verified_20260916/verified_training_data_no_longruns.csv` |
| R004 exclusion F1 recovery | improved exclusion detection | original v2 test: exclusion F1 0.557 (R002 cnn) -> 0.728 (R004 cnn); filtered test: 0.818 (cnn) / 0.782 (resnet) / 0.805 (lstm); division event F1 0.995-0.996 maintained | Match | exclusion-label quality target met | `trajectory_cleaning_run/no_longruns_20260916/` |
| R004 promotion | best architecture promoted by validation metrics only | `promote_best_model.py` selected cnn (validation mean F1 0.923); root `best_model.pt` metadata `training_round: R004`, thresholds 0.81/0.89, feature compatibility extended | Match | promotion manifest written | `trajectory_cleaning_run/promotion.json` |
| R005 corpus transfer | R004 corpus shipped byte-identical at `datasets/verified_20260916/` | sha256 `a2b3793d…` identical to legacy source; 312,317 rows / 1,053 trajectories (pandas recount) | Match | Frozen corpus treated as the trainset; upstream R002 corpus and raw exports remain archived in the legacy workspace | `datasets/manifest.json` |
| R005 promoted checkpoint transfer | Deployed checkpoint shipped byte-identical | Packaged `best_model.pt` sha256 equals promotion pin `3943bd8f…` (contract-integrity test) | Match | Data-structure table fingerprint corrected from the stale R001-era hash to the R004 hash | `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt` |
| R005 CLI relocation | General train/predict CLI packaged as `trajectory_cleaning/cli.py` + `moma-trajectory-cleaning` console script | `--help` runs from installed package; imports switched to intra-package | Match | Unresolved-item relocation executed with user approval; round drivers (`r002/r003/r004_train`) remain archived in the legacy workspace | `src/moma_lineage_analysis/trajectory_cleaning/cli.py` |
| R005 test-suite transfer | Six test modules rewired to the package layout | sys.path hacks removed; `AppTest` targets the packaged app; contract-integrity paths point at `datasets/`, `docs/`, packaged models | Match | Full battery re-run in validation gates | `tests/trajectory_cleaning/` |
| R006 raw-export QC | 35 files pass all hard gates | 0 failures on schema, boolean labels, missing values, finite positive features, duplicate observation keys, filename/key consistency, cross-file collisions; 22,890 rows / 35 trajectories, 1,948 raw division / 229 excluded labels | Match | Report frozen at `datasets/verified_20261013/qc_report.json` (canonical; the transient raw-inbox copy was rotated away) | `scripts/qc_raw_train_data.py` |
| R006 label staging (final: effective semantics) | `effective_division_point` declared the final true division label (user decision) | Staged with `--division-mode effective`: 1,932 divisions = 1,926 unchanged + 6 shifted targets; 22 raw labels superseded (7 coalesced, 6 shifted sources, 9 needs_review whose flanking cycles merge); division/exclusion overlap 0 | Match | Supersedes the first R002-policy build (1,947 divisions, sha256 `b0476ffc…`); counts reconcile exactly with resolver statuses | `datasets/verified_20261013/staging_report.json` |
| R006 duplicate-key renaming | 11 keys duplicating R004-corpus trajectories renamed with `20261013_` prefix (user decision) | Overlap check of rebuilt corpus against R004 corpus keys: NONE; mapping frozen in `datasets/verified_20261013/rename_map.json`; raw exports untouched | Match | Renamed trajectories are distinct corpus members; biological re-curation overlap is documented rather than hidden | `datasets/verified_20261013/rename_map.json` |
| R006 batch corpus build | Attested corpus under one batch dataset_id | 22,890 rows retained / 35 trajectories, keys namespaced `curated_20261013::<key>`, zero omissions; per-source stratified split 25/5/5 verified; sha256 `15721ab7…` | Match | First build with 35 per-file dataset IDs rejected (split needs >= 3 trajectories per source); single batch ID matches batch semantics and future merge needs | `datasets/verified_20261013/verified_training_data.csv` |
| R007 raw-export QC | 48 rotated-in files pass all hard gates | 0 failures; 31,384 rows / 48 trajectories, 2,728 raw division / 300 excluded labels; resolver statuses 2,691 unchanged / 13 coalesced / 7 shifted / 17 needs_review | Match | The transient raw-inbox report was rotated away and no durable R007 copy remains; the R006 report is archived at `verified_20261013/qc_report.json` | `scripts/qc_raw_train_data.py` |
| R007 staging + renaming | Same effective-label semantics; date prefix for duplicates | 2,698 effective divisions (37 superseded + 7 added); 18 keys renamed `20261016_*`; exact-key overlap vs R004 corpus and vs the 20261013 batch: NONE | Match | Normalized (prefix-stripped) overlap vs R004 is exactly the 18 documented re-curations | `datasets/verified_20261016/staging_report.json`, `rename_map.json` |
| R007 batch corpus build | Attested corpus under batch dataset_id | 31,384 rows / 48 trajectories namespaced `curated_20261016::<key>`; split 34/7/7 verified; 1-epoch CPU train smoke with `--require-human-verified` produced checkpoint + metrics; sha256 `e4fced1a…` | Match | Pipeline identical to R006 | `datasets/verified_20261016/verified_training_data.csv` |
| R008 corpus merge | R004 corpus + both batches, R004 filter on batches | 365,283 rows / 1,134 trajectories; dropped exactly `curated_20261013::FOV_121_CH_7` and `curated_20261016::FOV_53_CH_11`; uniqueness check passed (1,134 = 1,053 + 34 + 47); sha256 `4e198bfb…` | Match | Filter outcome matched the pre-registered predictions from R006/R007 staging reports | `datasets/verified_merged_20261016/merge_manifest.json` |
| R008 split | per-source stratified 70/15/15, seed 42 | 793/170/171 trajectories; every one of the 6 source datasets present in every partition | Match | -- | CLI `split_trajectory_keys` output |
| R008 training + selection | unchanged TrainConfig; selection by validation mean of division-event F1 and exclusion F1 | CNN 26 epochs (val div-event F1 0.9950, excl 0.7880, mean 0.8915); ResNet 28 epochs (0.9970/0.7844/0.8907); BiLSTM 38 epochs (0.9961/0.8014/**0.8987**) -> rule winner BiLSTM; test div-event F1 0.9968-0.9971, test exclusion F1 0.7951-0.8452, test tail recall 0.9959-0.9969, last-event miss 0.0058 for all | Match | Rule outcome recorded; promotion decided separately (next row) | `runs/merged_20261016/r008_model_comparison.json` |
| R008 promotion (user override) | User decision: promote CNN instead of rule winner BiLSTM | Hash-verified copy of `runs/merged_20261016/cnn/` artifacts to the package models root; previous R004 CNN + artifacts backed up (`promotion_backups/20261016T145403.716334Z`); override and rationale recorded in `promotion.json`; production checkpoint sha256 `3e796c9e…`; `load_checkpoint` + predict smoke OK (7/7 divisions on a sample trajectory) | Match (override documented) | Contract-integrity hash pin and app registry label updated to the R008 CNN; deviation from the validation-only rule is explicit and attributed to the user | `src/moma_lineage_analysis/models/trajectory_cleaning_run/` |
| R009 raw-export QC | 16 new single-trajectory exports pass all hard gates | 0 failures on schema, boolean labels, missing values, finite positive features, duplicate observation keys, filename/key consistency, cross-file collisions; 8,757 rows / 16 trajectories, 553 raw division / 138 excluded labels; warnings: 25 rows `division_point != effective_division_point` + 1 effective-only row | Match | Report frozen at `datasets/verified_20261017/qc_report.json` (canonical; the transient raw-inbox copy was rotated away); `FOV_12_CH_17` (599 rows) and `FOV_15_CH_2` (598 rows) are complete-duration exports with internal gaps, not truncated files | `scripts/qc_raw_train_data.py` |
| R009 staging + renaming | Same effective-label semantics as R006/R007; date prefix for duplicates | 529 divisions = 553 raw - 25 superseded + 1 effective-only (528 unchanged / 3 `coalesced_previous` / 1 `shifted_previous` / 21 `needs_review`); division/exclusion overlap 0; 9 keys renamed `20261017_*`; exact-key overlap vs R004 corpus NONE; normalized (prefix-stripped) overlap is exactly the 9 documented re-curations; `effective_division_point` equals the packaged resolver bitwise and staging reproduced the export labels exactly (0 mismatches, 0 rows missing) | Match | `curated_20261017::FOV_14_CH_14` (16-frame run, rows 569-584) is the only trajectory the R004 filter would drop; `curated_20261017::20261017_FOV_13_CH_19` keeps 23 divisions after 13 `needs_review` supersessions (longest run 4) | `datasets/verified_20261017/staging_report.json`, `rename_map.json` |
| R009 batch corpus build | Attested corpus under batch dataset_id `curated_20261017` | 8,757 rows / 16 trajectories namespaced `curated_20261017::<key>`; source `staged/combined_curated_20261017.csv`, zero omissions; sha256 `247da098…`; split feasibility 12/2/2 trajectories (train 6,861 / validation 1,032 / test 864 rows) | Match | Pipeline identical to R006/R007; batch additive and unmerged; 16 trajectories are below a standalone training population | `datasets/verified_20261017/verified_training_data.csv` |
| R009 independent verification | Separate verifier re-checks the frozen batch read-only | sha256 pin matches the manifest; 11 documented columns; namespace and `source_dataset` consistent; staged labels reproduce the packaged resolver (0 mismatches, 0 rows missing); duplication checks pass; R004 filter matches the staging prediction; 1-epoch CPU smoke with `--require-human-verified` completed; `pytest tests/trajectory_cleaning` 70 passed / 2 skipped | Match | All checks PASS; no corpus, checkpoint, metric, parameter, or model changed. The temporary smoke-run artifacts (under `runs/_audit_20261017/`) were removed during the pre-retraining cleanup; the verified batch corpus remains the durable evidence | `datasets/verified_20261017/` |
| R010 corpus merge | R004 corpus + all three curated batches, R004 filter applied to the batches | 373,440 rows / 1,149 trajectories; dropped exactly `curated_20261013::FOV_121_CH_7`, `curated_20261016::FOV_53_CH_11`, `curated_20261017::FOV_14_CH_14`; retained batch rows 22,236 / 30,730 / 8,157; 7 source datasets (baseline_0_10h 477, baseline_0_8h 19, rdm_gluc_toggle_20260905 546, starvation_20251111 11, curated_20261013 34, curated_20261016 47, curated_20261017 15); longest exclusion run 4; sha256 `7ecc9753…`; merge provenance `94a54d91…` | Match | Filter outcome matched the R009 staging prediction (`FOV_14_CH_14`); the 295 overlap rows are documented legacy R002 residue inside the frozen R004 corpus, not a defect | `datasets/verified_merged_20261017/merge_manifest.json` |
| R010 split | per-source stratified 70/15/15, seed 42 | 804/172/173 trajectories; 261,377 / 55,866 / 56,197 rows; 22,689 / 4,842 / 4,939 division and 1,304 / 314 / 306 excluded positives; every source present in every partition; splits identical to the canonical `split_trajectory_keys` recomputation and identical across the three runs | Match | -- | `runs/merged_20261017/cnn/splits.json` |
| R010 training + selection | unchanged TrainConfig; selection by validation mean of division-event F1 and exclusion F1; test never used | cnn 43 epochs (val div-event F1 0.9953, exclusion F1 0.8020, mean **0.8987**); ResNet 25 epochs (0.9962 / 0.7896 / 0.8929); BiLSTM 25 epochs (0.9946 / 0.7339 / 0.8642) -> rule winner cnn; test division-event F1 0.9933-0.9964, test exclusion F1 0.7973-0.8512, test tail recall 0.9929-0.9949, last-event miss 0.0058-0.0116; wall times 0.61 / 0.55 / 10.27 min | Match | Rule outcome recorded; **no promotion performed at round close**, so the production checkpoint and app default were unchanged at that point; the R010 CNN was promoted 2026-10-17 in round R011 | `runs/merged_20261017/r010_model_comparison.json` |
| R010 independent verification | Separate verifier re-checks the corpus and the three training runs read-only | corpus checks 11/11 PASS (sha256 pin, schema, counts, source composition, R004-filter drops, duplicates, run rule over all 1,149 trajectories, legacy-only overlap, row identity, manifest consistency, namespace); training-artifact checks 8/8 PASS (splits disjoint/complete, canonical recomputation match, every source in every partition, identical splits across models, checkpoint load + thresholds, provenance metadata, metrics blocks, histories); 0 blocking issues | Match | Only pending item was the `datasets/manifest.json` entry (D1), closed by this round's docs update; D2 (295 legacy overlap) and D3/D4 (documented duplicate-key and parser semantics) are non-blocking and recorded | `runs/merged_20261017/verification_report.json` |
| R011 promotion (user decision) | Promote the R010 CNN; the tool re-applies the validation-only rule and hash-verifies the copied artifacts | `promote_best_model.py` selected cnn (validation mean F1 0.898668 vs resnet 0.892905 / lstm 0.864244; test metrics never read); new `best_model.pt` sha256 `6e7c5a58…` byte-identical to `runs/merged_20261017/cnn/best_model.pt`; previous R008 CNN (`3e796c9e…`) + histories/metrics/splits backed up under `promotion_backups/20261017T124420.980675Z`; `promotion.json` rewritten; checkpoint loads and predicts finite outputs; contract-integrity SHA pin updated and PASS | Match | The tool backs up only the four model artifacts and replaces `promotion.json` without archiving it, so the R008 promotion manifest was replaced; the R008 decision remains in this contract (R008 registry/reconciliation/change-log rows) and `runs/merged_20261016/r008_model_comparison.json`, and the R004-era `promotion.json` survives in the `20261016T145403.716334Z` backup | `src/moma_lineage_analysis/models/trajectory_cleaning_run/` |
| R012 store training path (user-approved phase 2) | Build a verified SQLite dataset store; training must read it directly and record lineage ids; no model/metric/corpus change | Six frozen versions imported and verified (counts, canonical hashes, legacy CSV sha256, byte-identical exports, backup restore, rebuild determinism); independent phase-1 auditor PASS and schema-2 metadata fixes re-verified; store frame dtype-exact equal to the legacy CSV read with byte-identical split keys; R010 reproduction: resnet/lstm bit-identical weights and metrics, cnn thresholds/discrete metrics identical with AP drift <=8.95e-5 matched by a CSV-only control run (GPU nondeterminism); lineage ids/content hashes recorded in `metrics.json` | Match | Legacy CSVs retained; deletion deferred to phase 3 after user sign-off | `datasets/store/`, `runs/merged_20261017_store_repro/`, `trajectory_cleaning/dataset_store.py`, `trajectory_cleaning/cli.py` |
| R013 CSV retirement (user sign-off) | Delete the six legacy corpus CSVs after phase-2 sign-off; the store must remain the sole, verifiable training source | Strict verify PASS before deletion (`--require-legacy-csv`, 14 checks); six CSVs deleted (338.9 MB); normal verify PASS after deletion (all six `legacy_csv_state: retired`, export sha256 still equals each stored legacy pin); retirement manifest records per-file sha256/size/lineage id and a `dataset_store.py export` restore command; no provenance JSON, manifest, staged file, backup, model, metric or checkpoint touched | Match | The store export remains the rollback path; `verify --require-legacy-csv` fails until a corpus is restored, by design | `datasets/store/retired_csv_manifest_20261017.json`, `datasets/store/lineage.sqlite3` |
| R014 labels and naming | Effective targets; rename reused identifiers; no fabricated biological identity | 25,728 rows / 48 trajectories; 2,562 = 2,609 - 51 + 4 divisions, 98 exclusions, overlap 0; statuses 2,558 unchanged / 5 coalesced / 4 shifted / 42 needs_review; eight date-prefixed keys; exported effective labels match resolver bitwise; max exclusion run 3 | Match | Reused identifiers are naming collisions; identity across acquisitions unestablished; no exact new measurement/payload duplicates found | `datasets/verified_20261029/staging_report.json`, `rename_map.json`, `runs/audit_20261029/preflight.json` |
| R015 split | Per-source trajectory-grouped 70/15/15 seed 42; identical splits across all three models confirmed by independent replay | 838 train / 179 validation / 180 test; 279,548 / 59,643 / 59,977 rows; division 24,538 / 5,220 / 5,274; exclusion 1,376 / 325 / 321; every source in every partition | Match | Cross-round test-key overlap handled by descriptive paired panels, not by raw per-round score subtraction | `runs/merged_20261029/cnn/metrics.json`, `splits.json` |
| R015 merge | Add all R014 trajectories to frozen R010; retain mixed historical policy | 399,168 rows / 1,197 trajectories, 35,032 divisions / 2,022 exclusions / 295 legacy overlaps; no new R004 drops; store canonical `8decab478c80…`, CSV `d8a288a1…` | Match | Historical identical-content pair remains frozen; no new exact payload duplication found | `datasets/verified_merged_20261029/merge_manifest.json`, `runs/audit_20261029/build_report.json` |
| R015 training + selection | Unchanged seed-42 TrainConfig; maximize validation mean F1; no test selection | CNN 40 epochs (best 25), val division-event/exclusion F1 0.9951 / 0.7821, mean 0.8886; ResNet 21 (6), 0.9954 / 0.7250 / 0.8602; BiLSTM 23 (8), 0.9927 / 0.6719 / 0.8323. CNN validation winner; test div-event 0.9954 / 0.9952 / 0.9921, exclusion F1 0.8026 / 0.7669 / 0.7202, exclusion AP 0.8779 / 0.8527 / 0.7237; times 33.9 / 29.7 / 541.4 s | Match | Single seed; validation split differs from R010; no promotion | `runs/merged_20261029/model_comparison.json` |
| R015 paired evaluation | Same rows and own stored thresholds for production and R015 candidates | Common historical test: 103 trajectories / 26,092 rows / 1,957 division / 212 exclusion positives; production exclusion F1/AP 0.8232 / 0.8999 versus R015 CNN 0.8099 / 0.8791 (lower); division-event F1 0.9920 versus 0.9931. New batch: 7 / 3,780 / 388 / 13; production exclusion F1/AP 0.3750 / 0.4883 versus R015 CNN 0.4242 / 0.6273 (higher) | Descriptive | Common panel content overlap with old/new train/validation is empty; small new batch; no bootstrap/significance. Does not support replacing production | `runs/merged_20261029/cross_eval_vs_production.json` |
| R014/R015 store and retirement | Add two immutable lineages; preserve history; retire only redundant new CSVs after data verification | 401,076 observations / 1,200 keys / 8 versions / 4,680 memberships / 21 ingest-provenance / 14,458,880 bytes; 317 independent data checks PASS; post-retirement 18 checks PASS; eight retirement ledger records; corrected final backup sha256 `388246cfa0…`, 7,054,400 bytes; full pytest 230 passed / 4 skipped / 59 subtests | Match | Original six-version statistics and R013 retirement remain historical; staged files/raw exports/reports retained | `runs/merged_20261029/verification_report.json`, `runs/audit_20261029/retirement_report.json`, `runs/merged_20261029/store_backup.json`, `runs/merged_20261029/pytest_full.txt` |
| R015 CSV retirement | Retire only two redundant materialized CSVs after independent data verification | 317 data checks PASS before deletion; post-retirement verify 18 checks PASS; eight retirement ledger records; staged/raw/report provenance retained; corrected final backup sha256 `388246cfa0…`, 7,054,400 bytes | Match | Raw exports, staged files and reports retained; no promotion | `runs/audit_20261029/retirement_report.json`, `runs/merged_20261029/store_backup.json` |
| R015 final verification | Independent replay of corpus/store/training/selection/retirement and production invariants | PASS_WITH_DOCUMENTED_CAVEATS; 441 checks (317 data / 35 retirement / 65 training-cross-evaluation / 24 final integrity), failures empty; packaged retirement verify 18; production unchanged | Match | Caveats: single seed, no FOV holdout, scarce new-batch exclusions, no significance/bootstrap; external historical original source paths unavailable while frozen hashes pass; separate >2**53 audit-decision ID precision issue does not affect this batch | `runs/merged_20261029/verification_report.json` |
| R016 independent verification | Re-derive implementation, provenance, validation-only selection and locked evaluation; confirm production preserved | PASS_WITH_DOCUMENTED_CAVEATS: 21/21 checks, no failures, 183 replay checks per GPU; GPU0 replay exact on every discrete metric; production `promotion_performed: false` | Match | One CNN RF17 seed42 validation probability sits within 1e-5 of the 0.87 exclusion threshold, so cross-device replay flips it (precision -0.002738, F1 -0.001255); AP ties differ <=1.6e-4 across devices; compare discrete metrics on the original device | `runs/context_20261029/verification_report.json` |
| R016 protocol and context selection | Pre-register before training; freeze data/split/hidden64; lock on validation before test | 29 runs completed; seed42 shortlist CNN17/33/121 and ResNet17/33/97; lock chose RF17 for both (`test_evaluated_before_lock: false`); three-seed validation means 0.885582 +/- 0.004732 and 0.887547 +/- 0.006512; all winner seeds reported | Match | Best among screened/shortlisted schedules only; selection optimism from seed42 reuse; seed SD is optimization noise; no causal RF or universal-optimum claim | `runs/context_20261029/selection_lock.json`, `repeat_results.json` |
| R016 locked evaluation and controls | Evaluate only locked finalists and fixed controls after locking; descriptive panels | Locked three-seed test means CNN RF17 exclusion F1 0.7972 / AP 0.8659 (common panel 0.8771) and ResNet RF17 0.8043 / 0.8759 (0.8745) versus single-seed controls R015 CNN 0.8026 / 0.8779 (0.8791) and R010 production 0.8183 / 0.8960 (0.8999); newly trained original-ResNet k5 control evaluated separately | Descriptive | No locked model surpasses production; no significance/bootstrap; primary ResNet kernel differs from deployed k5; no promotion performed or recommended | `runs/context_20261029/evaluation.json`, `summary.json`, `control_evaluation.json` |
| R016 coverage and stratum diagnostics | Compare finalists on matched radius-98 interior plus length strata and per-source support | Interior (105 trajectories / 28,096 rows / 147 positives) exceeds boundary (179 / 31,547 / 178) for every locked seed; RF17 median span ~0.803 h with full coverage of 179 trajectories versus RF197 ~9.831 h covering 105 (47.1% of row centers); per-source exclusion support 104 (RDM-Gluc) down to 15 and 2 | Match | Frame contexts are not constant physical windows; tiny strata (2-15 positives) are uninterpretable; diagnostics never re-ranked contexts | `runs/context_20261029/validation_diagnostics.json` |
| R018 app integration | Independent Streamlit review app with nine modules, installed launcher, user guide and guarded SQLite backend | Source package, pyproject entry point, launcher help and docs exist; final suite 388 passed / 4 skipped / 59 subtests; no live DB mutation during implementation | Match | Independent verification PASS with zero open defects; browser delete confirmation fix re-tested (wrong-key error; exact-key snapshot + removal); no promotion | `src/moma_lineage_analysis/trainset_review/`, `docs/TRAINSET_REVIEW.md`, `runs/trainset_review_implementation/` |
| R018 mutation authority | User-authorized label edits and permanent whole-trajectory deletion; measurements immutable; backups and CAS guards | Labels autosave across shared observations; deletion removes observations/all memberships/undo payloads and requires exact key + impact; `time_h`/`spine_length` immutable; every mutation online-backup under `review_backups/` and global-revision CAS | Match | This is an explicit app-scoped exception to the prior frozen-corpus immutability rule; no claim original lineage/export bytes survive edits | `trainset_review/store.py`, `streamlit_app.py` |
| R018 working-version handling | Reviewed frozen versions become working; training refs refused; original pins retained | Original pins/counts/frozen flag/legacy export copied to review state; live version `frozen=0` with working hash/counts/order; training resolver rejects; read-only opening creates no review tables | Match | Original frozen versions are not byte-identical after edits; archival snapshots retain pre-change state | `trainset_review/store.py`, `trajectory_cleaning/dataset_store.py` |
| R018 finalization and import refusal | New dataset ID, human attestation, R004 membership-only filter, source provenance and native export hash | New frozen lineage records audit/provenance/canonical/native export hash; core verify accepts only matching finalize evidence; historical CSV import refused after review data exists | Match | Prevents permanent-deletion resurrection; finalized training lineages must pass verifier before training | `trainset_review/store.py`, `dataset_io.py`, `trajectory_cleaning/dataset_store.py` |
| R018 model selection surface | Shared registry exposes production R010 default and selectable non-production R015 | Registry tags, paths and sha256 pins match packaged R010/R015; release manifest records R015 lineage and RF17; no promotion performed | Match | Preview/evaluation selection is not model promotion or new holdout evidence | `model_registry.py`, `merged_20261029/cnn/release_manifest.json` |
| R019 duplicate-version removal | Remove the three `verified_merged_*` versions at the user's request without losing data | Preconditions: merged union 1,152 trajectories ⊆ batch union 1,155, zero unique trajectories, batch rows sum to all 387,217 observations; removal at revision 562 removed 3,345 memberships and left trajectories/observations unchanged; snapshot + audit event recorded; manifest entries marked `store_removed`; verify expects only the five retained lineages | Match | Removal is a new user-authorized data-layer operation implemented in the review backend with orphan refusal and CAS, not a script-side ad-hoc delete | `runs/store_cleanup_20261031/`, `trainset_review/store.py`, `scripts/remove_dataset_versions.py`, `datasets/manifest.json` |
| R019 finalization of reviewed batches | Five training-ready frozen lineages with R004 membership filter and native export hashes | Revisions 563-567; totals 385,815 rows / 1,152 trajectories / 33,665 divisions / 1,740 exclusions / 49 overlap; three R004 drops identified and applied to membership only; verify reports 5 WORKING + 5 PASS with `database_native` export hashes | Match | Reviewed originals stay working; finalized versions are the training inputs and remain reproducible from the store | `runs/store_cleanup_20261031/finalize_results.json`, `verify_after_finalize.json` |
| R019 training and selection | Unchanged configuration; validation-only selection; test only for acceptance | Split 806/173/173 identical across three runs and exactly reconstructed; CNN/ResNet/BiLSTM epochs 31/20/29; validation mean F1 0.901205/0.860075/0.884530 selects CNN; test div-event/exclusion F1 0.9966/0.7976 (AP 0.8783) | Match | New population/labels (review deleted 45 trajectories and edited labels), so not comparable with R015 per-round metrics | `runs/reviewed_20261031/model_comparison.json`, `training_summary.json` |
| R019 production comparison | Descriptive same-row panels; own thresholds; contamination counted | Common panel 35 trajectories / 10,317 rows / 44 exclusions: R019 CNN 0.9983 / 0.8395 / 0.9034 versus production 0.9948 / 0.8250 / 0.8618; new batch 7 / 3,780 / 14: 0.9948 / 0.7586 / 0.8654 versus 0.9834 / 0.3636 / 0.4804; full test panel contaminated for production (130/173 keys) | Descriptive | Small panels; single seed; no bootstrap/significance; R015 CNN contaminated; no promotion decision or recommendation | `runs/reviewed_20261031/production_comparison.json` |
| R020 promotion (user decision) | Promote the R019 CNN; the tool must re-apply the validation-only rule and hash-verify the artifacts | `promote_best_model` selected cnn (validation mean F1 0.901205 vs BiLSTM 0.884530 / ResNet 0.860075; test never read); new `best_model.pt` sha256 `816a8b39...` byte-identical to `runs/reviewed_20261031/cnn/best_model.pt`; previous R010 CNN (`6e7c5a58...`) + history/metrics/splits + archived `promotion.json` under `promotion_backups/20261030T184217.243155Z`; `promotion.json` rewritten; registry AI-panel label and contract-integrity pin updated; load-checkpoint/predict smoke finite | Match | The tool again replaced `promotion.json` without archiving it; the R010 manifest was copied into the backup directory manually and kept at `runs/reviewed_20261031/promotion_previous_R010.json`. No data/model/metric change | `src/moma_lineage_analysis/models/trajectory_cleaning_run/` |

### Method change log

| Method version | Date | Change | Reason | Comparability impact | Affected code and outputs |
|---|---|---|---|---|---|
| 0.1.0 | 2026-09-16 | Initialized contract | Establish durable terminology and method governance | No validated result yet | `METHODS_TRAJECTORY_CLEANING_MODEL.md` |
| 0.2.0 | 2026-09-16 | Extended feature layout (5 channels), RDM-Gluc corpus v2, position-stratified tail metrics | R002 root cause: absolute-time channel extrapolates beyond the training range on long trajectories, collapsing late-division recall | Division recall in the last quintile improved from 0.086 (promoted CNN) to 0.988-0.994; last-event miss rate from 0.573 to 0.000-0.012; not comparable to v0.1.0 tail metrics (new metrics) | `data.py`, `models.py`, `training.py`, corpus `verified_20260916/`, retrained checkpoints |
| 0.3.0 | 2026-09-17 | Corpus curation: drop 17 trajectories whose longest contiguous exclusion run is 5+ frames (analysis-driven cell-cycle removal, not growth noise); retrain on filtered corpus; promote R004 CNN | Exclusion F1 was depressed by non-representative long-run exclusion labels (R002 test exclusion F1 0.54-0.59) | On the original v2 test split, exclusion F1 rose 0.557 -> 0.728 (cnn); on the filtered test split it is 0.782-0.818; division event F1 unchanged (0.995-0.996). Comparable to v0.2.0 division metrics; exclusion metrics not comparable to v0.2.0 filtered-test values (different test population) | `dev_code/r004_train_no_longruns.py`, corpus `verified_training_data_no_longruns.csv`, `no_longruns_20260916/` checkpoints, `trajectory_cleaning_run/best_model.pt` (promoted R004 CNN) |
| 0.3.1 | 2026-10-13 | Repository transfer to the standalone `moma_lineage_analysis` package: training CLI packaged (`trajectory_cleaning/cli.py`, console script `moma-trajectory-cleaning`); R004 corpus + provenance JSONs shipped under `datasets/`; promoted checkpoint + R002/R003 checkpoints shipped under `src/moma_lineage_analysis/models/`; test suite rewired to package imports; declared paths reconciled | User request: standalone package for lineage construction + cleaning; no methodological change | All hashes, corpora sizes, hyperparameters, metrics, and the promoted checkpoint are unchanged; R002 corpus, R001/R004 run dirs, raw exports, and round driver scripts remain archived in the legacy `Moma_cellPose` workspace and are cited here as provenance only | `src/moma_lineage_analysis/trajectory_cleaning/cli.py`, `datasets/`, `tests/trajectory_cleaning/`, this contract |
| 0.3.2 | 2026-10-13 | Data addition: verified batch `verified_20261013` (35 trajectories / 22,890 rows). Label semantics per user decision: `division_point` = `effective_division_point` (recomputed effective boundaries; 1,932 divisions; overlap with exclusion 0). 11 keys duplicating R004-corpus trajectories renamed with a `20261013_` prefix per user decision (batch is duplication-free). New QC (`scripts/qc_raw_train_data.py`) and staging (`scripts/stage_verified_labels.py`, `--division-mode effective` / `--rename-map`) entry points; batch not yet merged into the R004 production corpus | User added newly curated single-trajectory exports, declared effective_division_point the final truth, and requested date-prefixed names for duplicated keys | No existing corpus, checkpoint, metric, or parameter changed; the batch is additive and separately hashed. Label-policy difference vs the historical R002 staging (shifted/needs_review raw labels at excluded rows were kept there) is documented for any future merge. `curated_20261013::FOV_121_CH_7` would be dropped under the R004 filter at merge time | `datasets/verified_20261013/`, `datasets/raw_train_data/`, `scripts/`, this contract |
| 0.3.3 | 2026-10-16 | Data addition: verified batch `verified_20261016` (48 trajectories / 31,384 rows) from rotated `raw_train_data/` contents; identical pipeline and label semantics as 0.3.2; 18 keys renamed with `20261016_` prefix (duplication-free vs both existing corpora); not yet merged | User added more curated exports and requested the same audit/build workflow | No existing corpus, checkpoint, metric, or parameter changed; batches remain additive and separately hashed. `curated_20261016::FOV_53_CH_11` would be dropped under the R004 filter at merge time | `datasets/verified_20261016/`, this contract |
| 0.4.0 | 2026-10-16 | Retraining on the merged corpus `verified_merged_20261016` (1,134 trajectories / 365,283 rows): cnn/resnet/lstm retrained with unchanged config; validation-only rule selected BiLSTM (mean F1 0.8987 vs 0.8915/0.8907); user overrode and promoted the CNN (test exclusion F1 0.8452, tail recall 0.9959, smallest/fastest); production checkpoint replaced with backup | User requested retraining after adding 83 curated trajectories (batches 20261013 + 20261016) and chose the CNN at promotion | Models trained on the merged corpus are not metric-comparable with R002/R004-era runs (new split, mixed label policy documented in Study design). The app's default model is now the R008 CNN; the R004 CNN is retained under `promotion_backups/` | `runs/merged_20261016/`, `datasets/verified_merged_20261016/`, promoted `best_model.pt` (`3e796c9e…`), this contract |
| 0.4.1 | 2026-10-17 | Data addition: verified batch `verified_20261017` (16 trajectories / 8,757 rows), identical pipeline and label semantics as 0.3.2/0.3.3; 9 keys duplicating R004-corpus trajectories renamed with a `20261017_` prefix (duplication-free vs the R004 and merged corpora); not yet merged | User added more curated exports and requested the audit/build workflow | No existing corpus, checkpoint, metric, or parameter changed; the batch is additive and separately hashed. `curated_20261017::FOV_14_CH_14` would be dropped under the R004 filter at merge time | `datasets/verified_20261017/`, this contract |
| 0.5.0 | 2026-10-17 | Retraining on the 4-source merged corpus `verified_merged_20261017` (1,149 trajectories / 373,440 rows): R004 corpus + batches 20261013/20261016/20261017 with the R004 filter applied to the batches (3 trajectories dropped); cnn/resnet/lstm retrained with unchanged config on GPU; validation-only rule selected the cnn (mean F1 0.8987); no promotion at round close (production checkpoint unchanged; the R010 CNN was promoted 2026-10-17 in round R011) | User requested temporary-file cleanup, incorporation of the R009 batch, and a fresh 3-model retraining comparison | Models trained on the merged corpus are not metric-comparable with R008-era runs (new split; mixed label policy documented in Study design). No corpus, checkpoint, metric, or parameter changed; the deployed model is still the R008 CNN | `datasets/verified_merged_20261017/`, `runs/merged_20261017/`, `datasets/README.md`, `datasets/manifest.json`, this contract |
| 0.5.1 | 2026-10-17 | Deployment: R010 CNN promoted to production (sha256 `6e7c5a58…`, byte-identical to `runs/merged_20261017/cnn/best_model.pt`); previous R008 CNN backed up under `promotion_backups/20261017T124420.980675Z`; `promotion.json` rewritten; app default model label updated to the R010 CNN | User decision after the R010 model report recommended promotion (validation-rule winner; paired same-population exclusion-AP improvement over the deployed R008 CNN) | No model, corpus, metric, or parameter changed - only the deployed checkpoint and its registry labels. Metrics are the R010 CNN metrics, not paired-comparable with R008-era per-round values on different splits (the model report carries the same-population comparison). The previous R008 promotion manifest was replaced by tool behavior (not backed up) | `src/moma_lineage_analysis/models/trajectory_cleaning_run/`, `promotion.json`, this contract |
| 0.6.0 | 2026-10-17 | Data layer: verified SQLite dataset store (six frozen versions, de-duplicated observations, canonical + legacy CSV hashes, byte-identical exports) and store-direct training (`--dataset-lineage`, lineage ids/content hashes in `metrics.json`); R010 reproduced from `verified_merged_20261017@v1` with byte-identical splits, bit-identical resnet/lstm and cnn AP drift <=8.95e-5 attributable to GPU nondeterminism | User approved the dataset-store plan (phase 2) to stop duplicating corpora on disk and to make training provenance explicit | No model, metric, parameter, corpus or deployed checkpoint changed; legacy CSVs are retained and store exports stay byte-identical, so existing corpus pins remain valid | `datasets/store/`, `scripts/dataset_store.py`, `trajectory_cleaning/dataset_store.py`, `trajectory_cleaning/cli.py`, `runs/merged_20261017_store_repro/`, this contract |
| 0.7.0 | 2026-10-17 | Retirement: the six legacy corpus CSVs (`datasets/verified_*/verified_training_data*.csv`, 338.9 MB) were deleted after user sign-off; the SQLite store is the sole training source, each export still reproduces the retired CSV byte-for-byte, and a per-file retirement manifest records sha256 + restore commands | The store made the duplicated CSVs redundant (375,348 unique observations vs 1,114,071 CSV rows); phase 3 removes the duplication after phase-2 verification | No model, metric, parameter, corpus content or deployed checkpoint changed; the same frozen rows are read from the store (dtype-exact frame parity proven in R012). Verification commands now treat retired CSVs as expected (`verify`), with `--require-legacy-csv` available to demand the pre-retirement state | `datasets/store/retired_csv_manifest_20261017.json`, `datasets/store/SCHEMA.md`, `scripts/dataset_store.py`, `trajectory_cleaning/dataset_store.py`, this contract |
| 0.8.0 | 2026-10-29 | R014 effective-label batch (48 trajectories / 25,728 rows) and R015 retraining on expanded frozen store lineage (1,197 / 399,168); eight naming collisions date-prefixed; no new R004 drops | User requested audit, additive ingestion and comparable retraining | Defaults and production remain unchanged; CNN wins validation (0.888577), mixed descriptive paired results do not support replacement; no significance/bootstrap; final verification PASS_WITH_DOCUMENTED_CAVEATS (441 checks, no failures) | `datasets/verified_20261029/`, `datasets/verified_merged_20261029/`, `runs/merged_20261029/`, this contract |
| 0.9.0 | 2026-10-29 | R016 context tuning: optional dilation schedules for CNN/primary ResNet at hidden64; 29 runs, validation-only lock selected RF17 for both families, locked models and controls evaluated descriptively | Compare nominal temporal support without conflating width or seed with context | No defaults, production checkpoint or data changed. RF is in normalized input, not a raw end-to-end window; dilation paths/kernels confound causality; shortlist reuse adds selection optimism; seed SD is optimization variation; locked models do not beat production; independent verification PASS_WITH_DOCUMENTED_CAVEATS (21 checks; compare discrete metrics on the original device); no promotion | `runs/context_20261029/`, `models.py`, this contract |
| 0.10.0 | 2026-10-29 | R018: add independent `moma-trainset-review` Streamlit app (nine modules) for guarded label review, permanent whole-trajectory deletion, working-version invalidation and new-ID finalization; shared registry exposes packaged selectable R015 beside unchanged R010 production | User-approved focused review workflow and user authority for explicit app-scoped data edits | No model training, selection, metric or production change. R018 is accepted after independent PASS with zero open defects; browser delete-confirmation and navigation/finalize/auto-device issues were fixed and re-tested. Measurements remain immutable; label edits/permanent deletion are user-authorized only through the review backend; original frozen lineage/export bytes do not remain identical after edits; R010 remains default, R015 selectable non-production. Final suite 388 passed / 4 skipped / 59 subtests. | `src/moma_lineage_analysis/trainset_review/`, `docs/TRAINSET_REVIEW.md`, `model_registry.py`, `runs/trainset_review_implementation/`, this contract |
| 0.11.0 | 2026-10-31 | R019: removed the three duplicate `verified_merged_*` dataset versions from the live store (subset proof, 3,345 memberships, zero orphaned data, revision 562, snapshot + pre-removal backup); finalized the five reviewed batches as `reviewed_20261031_*` training-ready lineages with the R004 membership filter (385,815 rows / 1,152 trajectories); retrained cnn/resnet/lstm with unchanged configuration on their composition (split 806/173/173) and compared them descriptively with production R010 on identical rows | User request: the merged corpora duplicate the five batch datasets; retrain on the re-checked five datasets | No hyperparameter, model-registry or production change. The reviewed population differs from R015 (45 trajectories permanently deleted, labels edited) so per-round metrics are not comparable; R019 CNN wins validation (0.9012) and shows higher descriptive exclusion AP on the small clean common panel (+0.0416 with 44 positives) and the new batch (+0.385 with 14 positives), without significance testing; no promotion performed or recommended; independent verification PASS 29/29 checks with 3 documentation-level discrepancies | `runs/store_cleanup_20261031/`, `runs/reviewed_20261031/`, `datasets/store/lineage.sqlite3`, `datasets/manifest.json`, this contract |
| 0.11.1 | 2026-10-31 | R020: promoted the R019 CNN to production (checkpoint sha256 `816a8b39...`, validation mean F1 0.901205, byte-identical to `runs/reviewed_20261031/cnn/best_model.pt`); previous R010 CNN (`6e7c5a58...`) plus its archived promotion manifest backed up under `promotion_backups/20261030T184217.243155Z`; `promotion.json`, shared registry, Streamlit AI-panel default label, contract-integrity pin, README/AGENTS/datasets documentation updated | User decision: promote the newly trained CNN to the main app | Deployment only: no corpus, dataset version, split, threshold, training run or metric changed. The R019 same-row comparison remains descriptive single-seed context; test metrics are not used for selection. Rollback = copy the four artifacts from the R010 backup and restore the previous pin/docs | `src/moma_lineage_analysis/models/trajectory_cleaning_run/`, `model_registry.py`, `tests/trajectory_cleaning/test_contract_integrity.py`, `tests/trajectory_cleaning/test_streamlit_ai_panel.py`, `tests/trainset_review/test_model_registry.py`, `README.md`, `AGENTS.md`, `datasets/manifest.json`, this contract |

- R016 is complete and independently verified (21/21 checks,
  PASS_WITH_DOCUMENTED_CAVEATS). Its three-seed design quantifies optimization
  noise, not dataset or FOV-generalization uncertainty, and does not resolve the
  FOV-held-out limitation below. Width tuning remains available as a separate
  future experiment; no promotion is authorized.

### Unresolved items and next checks

- R018 is closed after independent verification PASS with zero open defects. The
  initial browser delete-confirmation defect (UI-DELETE-CONFIRM-001) was fixed
  with an atomic confirmation form and re-tested in a fresh TEMP browser: wrong
  key errors; exact key saves a pre-mutation snapshot and removes the trajectory.
  Navigation callbacks, finalize selection, stale-view protection and auto/CUDA
  device resolution were also fixed and re-tested. Permanent deletion has no
  in-app restore; archival `review_backups/` are the recovery source. Do not
  claim original frozen lineage bytes/exports remain unchanged after review.
- The app opens the editable live store. Use a copy for testing; opening and
  browsing must not create review tables, and implementation must not mutate the
  live database.
- ~~Relocate `dev_code/trajectory_cleaning_cli.py` into the package~~ Resolved in
  round R005 (2026-10-13): packaged as
  `src/moma_lineage_analysis/trajectory_cleaning/cli.py` with the
  `moma-trajectory-cleaning` console script.
- FOV-held-out cross-validation and per-source evaluation; add uncertainty
  estimates (trajectory bootstrap) to reported metrics.
- Exclusion-label detection improved after R004 corpus curation (original v2
  test exclusion F1 0.557 -> 0.728 for cnn) but remains the weaker label;
  exclusion positives are scarce (1,570 after filtering). Next round should
  target exclusion-label diversity.
- Old-corpus (short trajectory) point precision dropped for R002 models
  (0.92-0.97 vs 0.994 for the R001 CNN) because thresholds were calibrated on
  the RDM-Gluc-dominated v2 validation split; per-condition thresholding or
  source-weighted calibration is a candidate fix.
- Sensitivity of `division_drop_ratio = 0.75` and `division_tolerance_frames = 1`
  not yet quantified.
- No seed-robustness retrain (all runs seed 42).
- R004 CNN promoted 2026-09-17 (replaces the R001 promoted checkpoint; backup
  under `trajectory_cleaning_run/promotion_backups/`). R002/R003 checkpoints
  remain selectable in the Streamlit model registry.
- The `verified_20261013` / `verified_20261016` / `verified_20261017` batches
  are no longer unmerged: round R010 merged them with the R004 corpus into
  `verified_merged_20261017` (R004 filter applied; `FOV_14_CH_14` dropped)
  and retrained. The R010 CNN was promoted to production on 2026-10-17 (round
  R011); future retraining rounds need their own re-promotion.
- ~~Dataset-store phase 3 (delete the legacy corpus CSVs)~~ Completed 2026-10-17
  (round R013) after user sign-off: the six CSVs were deleted, the store is the
  sole training source, and `datasets/store/retired_csv_manifest_20261017.json`
  records each file's sha256 plus a restore command. The store schema-migration
  path is manual (a scripted migration is a hardening item); only the CNN shows
  run-to-run GPU nondeterminism (see the model report and AGENTS.md caveats).
- R019 adds a user-authorized dataset-version removal capability
  (`remove_dataset_versions`) and five reviewed training-ready lineages. The
  three historical `verified_merged_*` lineage ids no longer resolve in the live
  store; the pre-removal backup
  `store/backups/lineage_20261030T181226.223866Z.sqlite3.gz` retains them. The
  R019 comparison panels are small (44 and 14 exclusion positives) and
  single-seed, so no promotion decision is supported by them; independent
  verification is recorded under `runs/reviewed_20261031/verification/`.
