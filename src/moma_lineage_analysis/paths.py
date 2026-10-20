"""Package-anchored filesystem locations.

Model checkpoints ship inside the package tree so they resolve identically in
editable installs, regular installs, and regardless of the caller's working
directory.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# Directory holding the released trajectory-cleaning checkpoints in the exact
# sub-layout the Streamlit app's AI_MODEL_REGISTRY expects
# (best_model.pt, verified_20260916/<arch>/best_model.pt,
# rdm_only_20260916/<arch>/best_model.pt).
PACKAGE_MODELS_RUN_DIR = PACKAGE_ROOT / "models" / "trajectory_cleaning_run"
