# Round R002 Report — End-of-Trajectory Division Recall

Date: 2026-09-16
Method version: 0.1.0 → 0.2.0
Contract: `METHODS_TRAJECTORY_CLEANING_MODEL.md`

## User-reported failure mode

The promoted model (R001 CNN) misses division points in the last part of long
trajectories. The new 21.6 h RDM-Gluc Toggle-L3 dataset
(`LineageAnalysis/trajectories_data/20260905_RDM-Gluc_Toggle-L3_selected_labeled.csv`,
241,540 rows / 560 trajectories / ~432 frames) exposed this clearly.

## Root cause (temporal extrapolation)

Training corpus (R001) covers at most ~10 h of experiment time, so the
absolute-time feature seen during training maxes out around 1.5–3.0. The new
dataset reaches 4.24 (432 frames × median 3 min sampling). Division recall of
the promoted checkpoint on the new dataset, by time-feature value:

| Time feature | Recall |
|---|---|
| ≤ 1.5 | ≥ 0.997 |
| 2.0–2.5 | 0.987 |
| 2.5–3.0 | 0.928 |
| 3.0–3.5 | 0.450 |
| 3.5–4.0 | 0.044 |

The division signature itself (normalized length drop ≈ −2.4) is intact late in
the trajectory — the network simply never saw such large absolute-time inputs,
and the unclipped extrapolation saturates the first-layer activations.

## Changes (all declared in METHODS v0.2.0)

1. **Boundary-safe feature layout** (`trajectory_cleaning/data.py`,
   `models.py`): extended layout of 5 channels — the v0.1.0 (time, length)
   channels plus `length_delta`, `log_growth_rate`, and
   `time_since_last_drop`, all stationary with respect to absolute experiment
   time. `TrajectoryNormalizer.feature_compatibility` selects the layout;
   pre-R002 checkpoints restore `legacy` (2-channel) automatically, so the
   old promoted checkpoint still loads and scores identically.
2. **Corpus v2** (`trajectory_cleaning_run/verified_20260916/`): merged
   baseline_0_10h, baseline_0_8h retained subset, starvation_20251111, and the
   RDM-Gluc dataset (label policy: `coalesced_previous` labels dropped,
   `effective_division_point`-only rows kept; 1,070 trajectories / 318,745
   rows, sha256 `deff25a9…23cba`).
3. **Position-stratified metrics** (`trajectory_cleaning/training.py`):
   `division_recall_head_quintile`, `division_recall_tail_quintile`, and
   `last_division_event.miss_rate` in `evaluate_model`.
4. **Retrained** CNN / ResNet / BiLSTM on corpus v2 (seed 42, 70/15/15 split,
   thresholds re-calibrated on validation).

## Results

Held-out v2 test partition (161 trajectories):

| Model | Point F1 (div) | Event F1 | Tail-quintile recall | Last-event miss rate |
|---|---|---|---|---|
| CNN | 0.988 | 0.990 | 0.988 | 0.006 |
| ResNet | 0.989 | 0.992 | 0.987 | 0.012 |
| BiLSTM | 0.985 | 0.988 | **0.994** | **0.000** |

Full RDM-Gluc dataset (405 scored trajectories), same models vs the old
promoted checkpoint:

| Model | Event recall | Tail-quintile recall | Last-event miss rate |
|---|---|---|---|
| Promoted R001 CNN | 0.751 | 0.086 | 0.573 |
| R002 CNN | 0.996 | 0.991 | 0.005 |
| R002 ResNet | 0.994 | 0.987 | 0.007 |
| R002 BiLSTM | **0.998** | **0.994** | **0.005** |

The failure mode is resolved: tail recall went from 8.6 % to ~99 % and the
last-division miss rate from 57 % to ≤ 1.2 %.

## Known trade-offs / open items

- Exclusion-point detection is unchanged in difficulty (test F1 0.54–0.59):
  only 1,876 exclusion positives exist in corpus v2. Needs a round with
  exclusion-label diversity.
- R002 thresholds are calibrated on the RDM-Gluc-dominated v2 validation
  split; on short (≤ 10 h) old-corpus trajectories the R002 models trade some
  point precision (0.92–0.97 vs 0.994 for the R001 CNN) while keeping event
  recall ≥ 0.989. Per-condition thresholding is a candidate follow-up.
- Models are trained but **not promoted**; `promote_best_model.py` runs only
  after user review.

## Follow-up R003 — RDM-Gluc-only training (user question)

Same method and features trained ONLY on the new dataset (560 trajectories,
`trajectory_cleaning_run/rdm_only_20260916/`, script `dev_code/r003_train_rdm_only.py`).

RDM-Gluc held-out test (84 trajectories):

| Model | Event F1 | Tail recall | Last-event miss | Exclusion F1 |
|---|---|---|---|---|
| CNN | 0.998 | 0.993 | 0.012 | **0.851** |
| ResNet | 0.996 | **0.997** | **0.000** | 0.562 |
| BiLSTM | 0.997 | 0.995 | 0.012 | 0.661 |

Cross-condition generalization, R001 legacy corpus test split (short ≤ 10 h
trajectories, 77 trajectories) — event F1 / exclusion F1:

| Model | RDM-only (R003) | Mixed v2 (R002) | R001 (trained on it) |
|---|---|---|---|
| CNN | 0.966 / 0.152 | 0.963 / 0.488 | 0.993 / 0.769 |
| ResNet | 0.929 / 0.119 | 0.980 / 0.490 | — |
| BiLSTM | 0.978 / 0.047 | 0.959 / 0.416 | — |

Interpretation: on the RDM-Gluc condition itself, RDM-only training matches or
slightly beats the mixed corpus (division metrics ~equal; exclusion F1 much
better for CNN: 0.851 vs 0.56 — the old corpus's exclusion examples dilute the
new dataset's exclusion signature). On out-of-condition short trajectories the
RDM-only models degrade noticeably in exclusion detection (F1 0.05–0.15) and
somewhat in division precision; division RECALL stays high everywhere
(0.98–0.99). If the production workload is RDM-Gluc-like long experiments,
RDM-only training is the better specialist; if multiple conditions must be
served by one checkpoint, the mixed corpus (R002) is safer.

## Validation evidence

- `validate_methods_file.py` on the v0.2.0 contract: PASS (86-test battery
  incl. contract-integrity: OK).
- Backward compatibility: pre-R002 checkpoint loads in legacy mode; legacy
  transform output shape (n, 2), forward pass OK.
- Edge-case fix during validation: empty `drop_indices` in
  `time_since_last_drop` (`data.py`) — caught by
  `test_trajectory_key_groups_and_sorts_rows`, fixed, full battery re-run OK;
  trained-model metrics re-verified unchanged.

## Artifacts

- Corpus + manifest: `trajectory_cleaning_run/verified_20260916/verified_training_data.csv`, `dataset_manifest.json`
- Resolved-label staging: `trajectory_cleaning_run/verified_20260916/rdm_gluc_toggle_20260905_resolved_labels.csv`
- Per-model runs: `trajectory_cleaning_run/verified_20260916/{cnn,resnet,lstm}/` (`best_model.pt`, `history.csv`, `splits.json`, `metrics.json`)
- Comparison tables: `trajectory_cleaning_run/verified_20260916/r002_model_comparison.json`, `r002_heldout_full_newdataset.json`
- Training script: `dev_code/r002_train.py` (moved from workspace root)
