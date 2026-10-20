# tests/trajectory_cleaning/

Test suite and method governance for the trajectory-cleaning sequence-model
sub-project (`trajectory_cleaning/` package, `trajectory_cleaning_run/`
artifacts, and the AI cleaning panel in `Moma_lineage_streamlit.py`).

## Contents

| File | Role |
|---|---|
| `test_trajectory_cleaning.py` | Normalization, dataset, collate, split, loss, training smoke test, thresholds, checkpoint I/O, event clustering |
| `test_streamlit_ai_cleaning.py` | AI cleaning panel integration (preflight, apply, undo/snapshot) |
| `test_streamlit_ai_panel.py` | AI panel state handling in the Streamlit app |
| `test_streamlit_division_resolution_panel.py` | Division/exclusion conflict resolution panel |
| `test_promote_best_model.py` | Atomic checkpoint promotion with hash verification |
| `test_prepare_verified_training_data.py` | Verified-corpus preparation, supersession, namespacing |
| `test_contract_integrity.py` | **Added 2026-09-16** — pins promoted checkpoint SHA-256, checkpoint schema, metrics/threshold agreement, split disjointness, dataset manifest schema, and Methods manifest parseability |
| `METHODS_TRAJECTORY_CLEANING_MODEL.md` | Living Methods contract v0.1.0 (scientific-analysis-loop skill; round R001 = retrospective documentation) |
| `METHODS_TRAJECTORY_CLEANING_MODEL_ACTUAL_MANIFEST.json` | Actual run manifest for reconciliation |
| `METHODS_TRAJECTORY_CLEANING_MODEL_ALIGNMENT.csv` | Declared-vs-actual alignment report |
| `AUDIT_REPORT.md` | Code-structure audit and improvement backlog |

## Run

```powershell
G:\miniconda3\envs\Moma_cellpose\python.exe -m unittest discover -s tests.trajectory_cleaning -t . -v
```

or the explicit module list shown in `docs/trajectory_cleaning_model.md`.
Status at consolidation (2026-09-16): 86 tests, all pass.

## Governance

Per `AGENTS.md`: before changing analysis code or interpreting a metric, read
`METHODS_TRAJECTORY_CLEANING_MODEL.md` completely; after each analysis round,
reconcile prose, registries, manifests, code, and outputs, and re-run the
validators:

```powershell
G:\miniconda3\envs\Moma_cellpose\python.exe .agents\skills\scientific-analysis-loop\scripts\validate_methods_file.py tests\trajectory_cleaning\METHODS_TRAJECTORY_CLEANING_MODEL.md
```
