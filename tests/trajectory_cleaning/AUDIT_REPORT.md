# Trajectory Cleaning Sub-Project — Audit Report

| Field | Value |
|---|---|
| Audit date | 2026-09-16 |
| Scope | `trajectory_cleaning/`, `trajectory_cleaning_run/`, `Moma_lineage_streamlit.py` AI integration, `dev_code/trajectory_cleaning_cli.py`, tests |
| Method contract | `METHODS_TRAJECTORY_CLEANING_MODEL.md` v0.1.0 (round R001, retrospective documentation) |
| Test status after consolidation | 86 unittest cases in `tests/trajectory_cleaning/` — **all pass** (38.5 s) |

## 1. What the sub-project does

A per-observation multi-label sequence classifier for mother-machine lineage
trajectories. Inputs are only `time_h` and `spine_length`; outputs are
`division_point` and `excluded_point` probabilities per row. Three
architectures were trained and compared on identical splits (CNN, dilated
ResNet, BiLSTM); the CNN won on validation mean F1 (0.862) and was promoted to
`trajectory_cleaning_run/best_model.pt`. The Streamlit app loads that
checkpoint, preflights trajectories, scores a chosen scope, resolves
division/exclusion conflicts deterministically, and attaches predictions to
exports — always as human-reviewed decision support.

## 2. Code structure inventory

| Component | File | Lines | Role | Assessment |
|---|---|---|---|---|
| Data layer | `trajectory_cleaning/data.py` | 460 | validation, normalization, dataset, collation, splits | Clean; strict validation; well tested |
| Models | `trajectory_cleaning/models.py` | 204 | CNN / ResNet / LSTM labelers | Clean; receptive fields documented |
| Training | `trajectory_cleaning/training.py` | 568 | focal loss, training loop, thresholds, checkpoint I/O | Solid; reproducibility seeding; early stopping restores best state |
| App API | `trajectory_cleaning/streamlit_api.py` | 1193 | preflight, scoped prediction, apply, conflict resolution, snapshots | Largest file; heavy validation; good audit columns |
| Provenance builder | `trajectory_cleaning/prepare_verified_training_data.py` | 671+ | merge curated exports, supersession, namespacing | Good; has CLI and tests |
| Promotion | `trajectory_cleaning/promote_best_model.py` | 433+ | hash-verified atomic promotion with backups | Good |
| Figures | `trajectory_cleaning/plot_training_results.py` | 281 | report figure regeneration | Fine |
| **Training CLI** | `dev_code/trajectory_cleaning_cli.py` | — | train/predict entry point | **Misplaced** (see F1) |
| Report | `trajectory_cleaning/MODEL_REPORT.md` | 632 | architecture comparison, results | Accurate; comparability notes preserved |
| Docs | `docs/trajectory_cleaning_model.md` | — | user/dev documentation | Path drift (see F2); test references now updated |

Run artifacts in `trajectory_cleaning_run/` are well organized: promoted
checkpoint + history/splits/metrics at root, per-run directories
(`verified_20260717`, `verified_20260718`), timestamped promotion backups.

## 3. Findings

### F1 — Training CLI lives in `dev_code/` (medium)
`docs/trajectory_cleaning_model.md` examples invoke `trajectory_cleaning_cli.py`
at the repo root; the actual file is `dev_code/trajectory_cleaning_cli.py`.
The training entry point is production-critical (it enforces manifest validation
and `--require-human-verified`) and should live in `trajectory_cleaning/` or
be exposed as `python -m trajectory_cleaning.cli`. **Action deferred** —
moving it is a breaking change for existing invocations; flagged in the contract
unresolved items for user decision.

### F2 — Documentation path drift (low, fixed)
The docs' "Focused tests" block referenced the old flat `tests.*` module
paths. **Fixed**: updated to `tests.trajectory_cleaning.*` in this round.

### F3 — Split is not FOV-held-out (high, scientific)
`split_trajectory_keys` partitions trajectories but one FOV can contribute to
multiple partitions, so acquisition-specific patterns leak between train and
test. The report's held-out metrics are upper bounds. Mitigation planned: FOV
held-out cross-validation (needs a `fov_id` column in the prepared corpus).

### F4 — Starvation-condition exclusion weakness (high, data)
Test exclusion F1 = 0.273 on the starvation condition (16 positives; only 2
starvation trajectories per held-out partition). The exclusion label is
under-diverse. **This is the primary target for the incoming new dataset.**

### F5 — No uncertainty quantification (medium)
All metrics are point estimates from a single seed (42) and a single split.
Trajectory-level bootstrap CIs and a second-seed retrain are cheap and should
accompany the next training round.

### F6 — Calibration split reuse (medium)
Model selection and threshold calibration both consume the validation split;
there is no held-out calibration set. Acceptable at current corpus size; note
when reporting probabilities as confidence.

### F7 — Default checkpoint path hardcoded (low)
`Moma_lineage_streamlit.py` defaults `DEFAULT_AI_CHECKPOINT` to
`trajectory_cleaning_run/best_model.pt` (overridable via
`MOMA_AI_MODEL_PATH`). Reasonable, but a packaged default path would survive
repo relocation better.

### F8 — Non-causal normalization (low, by design)
`TrajectoryNormalizer` uses whole-trajectory median/MAD — correct for offline
cleaning, but must never be reused for online prediction. Documented in the
contract interpretation boundary.

### F9 — Report/architecture messaging mismatch (low, informational)
`MODEL_REPORT.md` calls the ResNet the "recommended default" architecturally,
yet the promoted checkpoint is the CNN (metric-based selection). The selection
rule (validation mean F1) is the authority; consider rewording the report at the
next retrain so readers do not expect a ResNet deployment.

## 4. Test consolidation performed in this round

- Created `tests/trajectory_cleaning/` (mirrors module name; precedent:
  `tests/omnishim/`) with `__init__.py`.
- Moved 6 modules: `test_trajectory_cleaning`, `test_streamlit_ai_cleaning`,
  `test_streamlit_ai_panel`, `test_streamlit_division_resolution_panel`,
  `test_promote_best_model`, `test_prepare_verified_training_data`.
- Fixed `REPOSITORY_ROOT` depth (+1 level) in all 6 files; added
  `__init__.py` so unittest package imports work. **78 moved + 8 new contract-integrity tests = 86 tests, all pass.**
- Added `test_contract_integrity.py` (new): guards the governance surface —
  promoted checkpoint SHA-256, checkpoint schema/consistency, metrics/threshold
  agreement, split disjointness, dataset manifest schema, Methods manifest
  parseability and entry-point existence.

## 5. Improvement backlog for the next round (new dataset)

1. **Ingestion**: run the new curated export through
   `prepare_verified_training_data.py` (bump run id, e.g. `verified_202609XX`);
   target exclusion-label diversity, especially starvation-like conditions.
2. **Method update**: MINOR version bump of the contract; extend the split to
   FOV-held-out CV if `fov_id` is available; add bootstrap CIs.
3. **Training**: retrain all three architectures on the enlarged corpus, keep
   identical `ModelConfig`/`TrainConfig` for comparability, then re-select on
   validation mean F1.
4. **Robustness**: second-seed retrain; threshold-tolerance sensitivity
   (0/1/2 frames); `division_drop_ratio` sensitivity.
5. **Promotion**: only via `promote_best_model.py` (hash-verified), update the
   declared manifest thresholds, regenerate figures + MODEL_REPORT, re-run the
   full 78+ test battery and both skill validators.
6. **Structural** (needs approval): move `dev_code/trajectory_cleaning_cli.py`
   into the package.
