# Methods: trajectory cleaning model

| Contract field | Value |
|---|---|
| Research task | trajectory cleaning model |
| Analysis ID | `trajectory_cleaning_model` |
| Method version | `0.7.0` |
| Status | Reconciled (R013 retirement: the six legacy corpus CSVs were deleted after user sign-off; the SQLite dataset store is the sole training source and keeps byte-identical exports/restore commands) |
| Created | 2026-09-16 |
| Last reconciled | 2026-10-17 |
| Project root | `G:/python_code/moma_lineage_analysis` |
| Contract owner | Research team |
| Prior method version | `0.6.0` |

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
current production corpus (round R002) is
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
  input; 66,050 with 5-channel input), dilated 1D ResNet (`ResNetLabeler`,
  165,698 params with 2-channel input; 168,770 with 5-channel input),
  bidirectional LSTM (`LSTMLabeler`, 333,058 params with 2-channel input;
  339,330 with 5-channel input). Default `ModelConfig` since R002: input 5,
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
  42; no multiplicity correction (single comparison framework).

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
| deployed checkpoint | `models/trajectory_cleaning_run/best_model.pt` (shipped under `src/moma_lineage_analysis/`) | one file = promoted CNN checkpoint | checkpoint sha256 | model_config, normalizer, thresholds, metadata, state_dict | promotion is verified backup + hash-checked copy | sha256 6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b (R010 CNN, promoted 2026-10-17 by user decision from `runs/merged_20261017/cnn/`; previous R008 CNN `3e796c9e…` backed up under `promotion_backups/20261017T124420.980675Z`) |
| Streamlit trajectory dict | `Moma_lineage_streamlit.py` session state | mapping key = trajectory_key | trajectory key | `trajectory` frame with `fov_cell_index`, `time_h`, `spine_length` | preflight drops invalid rows to validation-rule scoring | TRAJECTORY_STATISTICS_SCHEMA_VERSION = 1 |

### Global analysis constraints

- Do not silently redefine, rename, clip, normalize, interpolate, exclude, or aggregate a registered metric.
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
| `METHODS_CONTRACT` | `docs/METHODS_TRAJECTORY_CLEANING_MODEL.md` | Living Methods contract | 0.3.1 | scientific-analysis-loop | Validated (contract-integrity tests PASS) |
| `R004_MODELS` | `trajectory_cleaning_run/no_longruns_20260916/` (archived in legacy workspace; promoted CNN shipped as `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt`) | R004 cnn/resnet/lstm checkpoints, histories, metrics, comparison | 0.3.0 | `dev_code/r004_train_no_longruns.py` (archived) | 86-test battery OK; R004 CNN promoted via `promotion.json` |
| `R004_CORPUS` | `datasets/verified_20260916/verified_training_data_no_longruns.csv` (retired in R013; restore via store export) | R004 production training corpus (frozen) | 0.3.1 | round R004 filter over R002 corpus | sha256 verified 2026-10-13 (`a2b3793d…`); 312,317 rows / 1,053 trajectories |
| `R009_CORPUS` | `datasets/verified_20261017/verified_training_data.csv` (retired in R013; restore via `scripts/dataset_store.py export`) | R009 batch training corpus (now merged into the R010 retraining corpus) | 0.4.1 | `scripts/qc_raw_train_data.py` + `scripts/stage_verified_labels.py` + `trajectory_cleaning/prepare_verified_training_data.py` | sha256 `247da098…` verified 2026-10-17; 8,757 rows / 16 trajectories (independent verifier PASS) |
| `R010_MODELS` | `runs/merged_20261017/` | R010 cnn/resnet/lstm checkpoints, histories, metrics, splits, comparison, figures, verification report | 0.5.0 | packaged `moma-trajectory-cleaning train` on `verified_merged_20261017` (GPU) | independent verifier PASS (corpus 11/11, artifacts 8/8); validation-selected cnn; promoted 2026-10-17 (R011, see `R011_PROMOTION`) |
| `R011_PROMOTION` | `src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt` (+ `history.csv`, `metrics.json`, `splits.json`, `promotion.json`) | promoted production checkpoint (R010 CNN) | 0.5.1 | `trajectory_cleaning/promote_best_model.py --run-dir runs/merged_20261017` | hash-verified copy of `runs/merged_20261017/cnn/best_model.pt` (sha256 `6e7c5a58…`); previous R008 CNN backed up under `promotion_backups/20261017T124420.980675Z`; contract-integrity pin updated |
| `R012_STORE_TRAINING` | `datasets/store/lineage.sqlite3` (+ `SCHEMA.md`, `backups/`), `runs/merged_20261017_store_repro/` | SQLite dataset store (six frozen versions) + store-direct training reproduction of R010 (cnn/resnet/lstm) | 0.6.0 | `trajectory_cleaning/dataset_store.py`, `scripts/dataset_store.py`, `trajectory_cleaning/cli.py --dataset-lineage` | store verify 6/6 versions PASS (counts, canonical hashes, legacy CSV sha256, byte-identical exports); independent phase-1 auditor PASS; R010 reproduction: byte-identical splits, resnet/lstm bit-identical, cnn AP drift <=8.95e-5 matched by a CSV-only control (GPU nondeterminism); no CSVs deleted |
| `R013_STORE_RETIREMENT` | `datasets/store/retired_csv_manifest_20261017.json` (+ six `datasets/verified_*/` provenance folders) | retirement record for the six legacy corpus CSVs + post-retirement verification evidence | 0.7.0 | `python scripts/dataset_store.py verify` before/after deletion; explicit user sign-off | strict verify PASS before deletion (14 checks); normal verify PASS after deletion with all six versions `legacy_csv_state: retired`; exports still byte-identical to the pinned sha256; `datasets/` 348 MB -> 29 MB; restore command recorded per corpus |

### Declared method manifest

The JSON below is the machine-readable intended contract used for code-method reconciliation.

<!-- METHOD_MANIFEST_START -->
```json
{
  "analysis_id": "trajectory_cleaning_model",
  "method_version": "0.7.0",
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
    "src/moma_lineage_analysis/trajectory_cleaning/cli.py --dataset-lineage (store-direct training)"
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
    "datasets/store/lineage.sqlite3",
    "datasets/store/SCHEMA.md",
    "datasets/store/retired_csv_manifest_20261017.json",
    "datasets/manifest.json",
    "docs/PLAN_DATASET_STORE.md"
  ],
  "analysis_unit": "observation (one cell at one time point); division event for event-level metrics",
  "independent_unit": "trajectory (trajectory_key); known limitation: not FOV-held-out",
  "populations": {
    "train": "804 trajectories / 261,377 rows (22,689 division, 1,304 excluded positives)",
    "validation": "172 trajectories / 55,866 rows (4,842 division, 314 excluded positives)",
    "test": "173 trajectories / 56,197 rows (4,939 division, 306 excluded positives)",
    "note": "R010 populations on verified_merged_20261017 (4 sources; R004 filter applied to the three batches: 3 trajectories dropped, longest exclusion run 4 frames across all 1,149 trajectories)"
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
    "R002_LABEL_POLICY": "coalesced_previous labels dropped; edp-only rows kept; needs_review/shifted_previous kept"
  },
  "statistics": {
    "model_selection": "maximize mean of validation division-event F1 and validation exclusion F1; test never used for selection",
    "uncertainty": "none reported (limitation)",
    "seed": 42
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
    "datasets/store/lineage.sqlite3 (six frozen dataset versions)",
    "datasets/store/backups/lineage_20261017T132240.874489Z.sqlite3.gz"
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

### Current round plan

Request: execute phase 3 of the user-approved dataset-store plan - retire the six
legacy corpus CSVs after the phase-2 sign-off - without changing any model,
metric, parameter, or corpus content.

- Inputs: `datasets/store/lineage.sqlite3` (schema 2, six frozen versions), its
  backup, the six `datasets/verified_*/verified_training_data*.csv` files, and the
  per-corpus provenance JSONs.
- Metrics/parameters/statistics: unchanged; nothing is re-derived in this round.
  Each store export must still reproduce the retired CSV byte-for-byte.
- Validation: (1) strict store verification PASS with all six CSVs present
  (`verify --require-legacy-csv`); (2) delete only the six pinned corpus CSVs;
  (3) normal store verification PASS afterwards with `legacy_csv_state: retired`
  and export sha256 still equal to each stored legacy pin; (4) retirement
  manifest written with per-file sha256/size/lineage id and restore commands;
  (5) full test suite; (6) independent verifier audit of the post-retirement
  state.
- Outputs: `datasets/store/retired_csv_manifest_20261017.json`, updated
  `datasets/README.md`, `AGENTS.md`, `PLAN_DATASET_STORE.md`, `MODEL_REPORT.md`,
  `README.md`, `datasets/manifest.json`, `datasets/store/SCHEMA.md`, and this
  contract.
- Round type justification: a retirement round. No model, metric, parameter,
  corpus content, or deployed checkpoint changed; the store already held the same
  frozen rows and its exports were proven byte-identical in R012.

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

### Unresolved items and next checks

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
