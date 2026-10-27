import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import copy
import hashlib
import re
from pathlib import Path
from scipy import signal as sg
from typing import Optional, Iterable, Tuple, Union
from warnings import warn
from joblib import Parallel, delayed
import torch
from moma_lineage_analysis import moma_logic
from moma_lineage_analysis.moma_logic import (
    get_one_trajectory,
    compute_traj_stats,
    process_channel_worker,
    COLNAME_NOT_FLU_CHANNEL
)
from moma_lineage_analysis.paths import PACKAGE_MODELS_RUN_DIR
from moma_lineage_analysis import curation_store as _curation_store_module
import importlib

# Upgrade an already-running session without losing unsaved labels or the
# pending initial-save context. Existing coordinator functions share these globals.
if (not getattr(_curation_store_module, "NETWORK_PATHS_SUPPORTED", False)
        or getattr(_curation_store_module, "_SCHEMA_VERSION", 0) < 3):
    importlib.reload(_curation_store_module)
import moma_lineage_analysis.app.curation_session as _curation_session_module
if getattr(_curation_session_module, "COMPACT_CURATED_SCHEMA", 0) < 3:
    importlib.reload(_curation_session_module)
from moma_lineage_analysis.curation_store import (
    DB_FILENAME, CurationStoreError, fingerprint_sources, find_dataset,
    list_datasets, load_dataset,
)
from moma_lineage_analysis.app.curation_session import (
    attach_saved_dataset, flush_curation, initialize_curation, save_recovery_copy,
    save_current_session_to_curation,
)
from moma_lineage_analysis.image_review import ImageReviewError, make_review_snapshot
import moma_lineage_analysis.app.image_review_panel as _image_review_panel_module
if getattr(_image_review_panel_module, "PAGE_STATE_VERSION", 0) < 2:
    importlib.reload(_image_review_panel_module)
from moma_lineage_analysis.app.image_review_panel import (
    close_image_review,
    open_image_review,
    render_image_review_if_open,
)
from moma_lineage_analysis.trajectory_cleaning import load_checkpoint
from moma_lineage_analysis.trajectory_cleaning.training import resolve_device
from moma_lineage_analysis.trajectory_cleaning.streamlit_api import (
    apply_ai_predictions,
    preflight_ai_trajectory_inputs,
    predict_trajectory_scope,
    resolve_division_exclusion_conflicts,
    restore_cleaning_snapshot,
    suggested_indices,
)

# --- Streamlit Application ---

st.set_page_config(page_title="MoMa Lineage Explorer (Plotly)", layout="wide")
st.sidebar.info(f"Logic Version: {moma_logic.LOGIC_VERSION}")


DEFAULT_AI_CHECKPOINT = os.environ.get(
    "MOMA_AI_MODEL_PATH",
    str(PACKAGE_MODELS_RUN_DIR / "best_model.pt"),
)

# Registry of released trajectory-cleaning checkpoints with usage guidance.
# Key = display name tag; value = (checkpoint path, usage suggestion).
# Paths are anchored to the package-shipped models directory
# (moma_lineage_analysis.paths.PACKAGE_MODELS_RUN_DIR), independent of the
# working directory the app is launched from.
AI_MODEL_REGISTRY = {
    "R010 CNN (merged corpus · production)": (
        str(PACKAGE_MODELS_RUN_DIR / "best_model.pt"),
        "Current production model (promoted 2026-10-17): CNN trained on the "
        "merged verified corpus (1,149 trajectories, all conditions incl. the "
        "2026-10 curated batches). Test division-event F1 0.996, exclusion F1 "
        "0.851, tail recall 0.995. Validation-only rule winner; paired evaluation "
        "shows significantly better exclusion AP than the previous R008 CNN on "
        "identical test sets.",
    ),
    "R002 CNN (mixed corpus · all conditions)": (
        str(PACKAGE_MODELS_RUN_DIR / "verified_20260916" / "cnn" / "best_model.pt"),
        "Trained on all conditions incl. the 21.6 h RDM-Gluc dataset. Safe "
        "default for mixed workloads: tail division recall 0.99, last-event "
        "miss rate <1%; slightly lower precision on short trajectories.",
    ),
    "R002 ResNet (mixed corpus · all conditions)": (
        str(PACKAGE_MODELS_RUN_DIR / "verified_20260916" / "resnet" / "best_model.pt"),
        "Mixed-corpus dilated ResNet; same scope as R002 CNN with a wider "
        "receptive field (121 frames).",
    ),
    "R002 BiLSTM (mixed corpus · all conditions)": (
        str(PACKAGE_MODELS_RUN_DIR / "verified_20260916" / "lstm" / "best_model.pt"),
        "Mixed-corpus bidirectional LSTM; best last-division miss rate on "
        "the mixed test split (0.0%), slower to train/predict.",
    ),
    "R003 CNN (RDM-Gluc only · long experiments)": (
        str(PACKAGE_MODELS_RUN_DIR / "rdm_only_20260916" / "cnn" / "best_model.pt"),
        "Specialist for long RDM-Gluc-like experiments: best exclusion "
        "detection (F1 0.85) and tail recall 0.99. Do NOT use for short "
        "≤10 h trajectories from other conditions.",
    ),
    "R003 ResNet (RDM-Gluc only · long experiments)": (
        str(PACKAGE_MODELS_RUN_DIR / "rdm_only_20260916" / "resnet" / "best_model.pt"),
        "RDM-only specialist; zero last-division misses on the RDM-Gluc "
        "test split. Not for other conditions.",
    ),
    "R003 BiLSTM (RDM-Gluc only · long experiments)": (
        str(PACKAGE_MODELS_RUN_DIR / "rdm_only_20260916" / "lstm" / "best_model.pt"),
        "RDM-only specialist; balanced division/exclusion behavior on long "
        "experiments. Not for other conditions.",
    ),
}
AI_MODEL_CUSTOM_TAG = "Custom path…"


def resolve_ai_model_tag(tag):
    """Return (checkpoint_path, suggestion) for a registry tag or custom path."""
    if tag in AI_MODEL_REGISTRY:
        return AI_MODEL_REGISTRY[tag]
    return st.session_state.get("ai_checkpoint_custom_path", ""), ""


def current_ai_model_tag():
    """Registry tag whose path matches the configured checkpoint, if any."""
    configured = st.session_state.get("ai_checkpoint_path", DEFAULT_AI_CHECKPOINT)
    try:
        configured_resolved = str(Path(configured).expanduser().resolve())
    except Exception:
        configured_resolved = configured
    for tag, (path, _suggestion) in AI_MODEL_REGISTRY.items():
        if str(Path(path).resolve()) == configured_resolved:
            return tag
    return AI_MODEL_CUSTOM_TAG


def apply_ai_model_tag():
    """Sync the registry dropdown selection into the configured checkpoint path."""
    tag = st.session_state.get("ai_model_tag", AI_MODEL_CUSTOM_TAG)
    if tag in AI_MODEL_REGISTRY:
        st.session_state.ai_checkpoint_path = AI_MODEL_REGISTRY[tag][0]
    invalidate_ai_model_selection()

# A division conflict is shifted to the previous valid observation only when
# the next/previous valid spine-length ratio provides evidence of a real drop.
DIVISION_CONFLICT_DROP_RATIO = 0.75

# Bump this whenever the conflict-resolution or statistics schema changes.
# It is part of every trajectory cache signature, so older Streamlit sessions
# are migrated automatically without requiring users to reload their data.
TRAJECTORY_STATISTICS_SCHEMA_VERSION = 2


# High-contrast palette for cell-cycle segments (Tol "bright" extended with
# distinct hues). Chosen so neighboring cycles stay distinguishable for
# color-vision-deficient users and never repeat within one trajectory.
TRAJECTORY_TRACE_COLORS = (
    '#0077BB',  # blue
    '#EE7733',  # orange
    '#009988',  # teal
    '#CC3311',  # vermillion
    '#33BBEE',  # cyan
    '#EE3377',  # magenta
    '#BBBB00',  # olive-yellow
    '#882255',  # wine
    '#44AA99',  # blue-teal
    '#117733',  # green
    '#AA4499',  # purple
    '#661100',  # brown
    '#6699CC',  # pale blue
    '#DDCC77',  # sand
    '#999933',  # olive
    '#332288',  # indigo
)

# Semantic marker colors/sizes for review overlays. Exclusion markers stay
# SMALL so they cannot cover neighboring observations of long trajectories.
APPLIED_EXCLUSION_MARKER = dict(symbol='x', size=7, color='#dc2626', line=dict(width=1.5))
AI_EXCLUSION_MARKER = dict(symbol='x-open', size=6, color='#f97316', line=dict(width=1))
AI_DIVISION_MARKER = dict(symbol='diamond-open', size=9, color='#7c3aed', line=dict(width=1.5))


AI_DEVICE_OPTIONS = (
    "auto",
    "cpu",
    *tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())),
)


@st.cache_data(show_spinner=False)
def checkpoint_sha256(checkpoint_path, modified_time_ns, file_size):
    """Fingerprint model bytes once per file version for cache/provenance."""
    del modified_time_ns, file_size
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_ai_device(device_selection):
    resolved = resolve_device(device_selection)
    if resolved.type == "cuda" and resolved.index is None:
        return "cuda:0"
    return str(resolved)


def configured_ai_checkpoint_signature():
    """Return a content-sensitive cache key for the configured model.

    Cached in session_state: every rerun rebuilds the app script top to
    bottom, so without the cache each rerun pays a Path.stat + canonical
    device resolution; the cached tuple is invalidated only when the
    configured path changes (which also resets ai_model_signature).
    """
    checkpoint_path = Path(
        st.session_state.get('ai_checkpoint_path', DEFAULT_AI_CHECKPOINT)
    ).expanduser()
    if not checkpoint_path.is_file():
        return None
    resolved_checkpoint = str(checkpoint_path.resolve())
    # Key the per-rerun cache on the file version too, so overwriting or
    # touching the same checkpoint path (retraining, os.utime) still
    # invalidates the cached signature -- only the sha256 digest is skipped
    # on a hit, which is the expensive part for large checkpoints.
    file_stat = checkpoint_path.stat()
    cache = st.session_state.get('_ai_checkpoint_signature_cache')
    if (
        cache is not None
        and cache[0] == resolved_checkpoint
        and cache[1] == file_stat.st_mtime_ns
        and cache[2] == file_stat.st_size
    ):
        return cache[3]
    file_digest = checkpoint_sha256(
        resolved_checkpoint,
        file_stat.st_mtime_ns,
        file_stat.st_size,
    )
    signature = (
        resolved_checkpoint,
        file_stat.st_mtime_ns,
        file_stat.st_size,
        file_digest,
        canonical_ai_device(st.session_state.get('ai_device', 'auto')),
    )
    st.session_state['_ai_checkpoint_signature_cache'] = (
        resolved_checkpoint,
        file_stat.st_mtime_ns,
        file_stat.st_size,
        signature,
    )
    return signature


@st.cache_resource(show_spinner=False)
def load_ai_checkpoint_cached(
    checkpoint_path,
    modified_time_ns,
    file_size,
    file_digest,
    device,
):
    """Load one checkpoint per path/version/device across Streamlit reruns."""
    del modified_time_ns, file_size, file_digest  # Cache-key components.
    return load_checkpoint(checkpoint_path, device=device)


def clear_ai_predictions():
    """Discard model previews without changing applied-label undo history."""
    st.session_state.ai_predictions = {}
    st.session_state.ai_predictions_revision = int(
        st.session_state.get('ai_predictions_revision', 0)
    ) + 1
    st.session_state.ai_preview_count_cache = {}
    st.session_state.ai_prediction_signature = None
    st.session_state.ai_prediction_checkpoint = None


def reset_ai_working_state():
    """Discard all AI state that cannot be reused for newly processed data."""
    clear_ai_predictions()
    st.session_state.ai_undo_snapshot = None
    st.session_state.ai_last_message = None
    st.session_state.ai_autopilot_request = None
    st.session_state.ai_autopilot_last_status = None


def invalidate_ai_model_selection():
    """Prevent stale model predictions after checkpoint/device changes."""
    st.session_state.pop('_ai_checkpoint_signature_cache', None)
    st.session_state.ai_model_signature = None
    st.session_state.ai_model_metadata = {}
    st.session_state.ai_model_error = None
    clear_ai_predictions()
    st.session_state.ai_last_message = None
    st.session_state.ai_autopilot_enabled = False
    st.session_state.ai_autopilot_request = None


def handle_ai_cleaning_toggle():
    """Disarm automatic application whenever AI assistance is disabled."""
    if not st.session_state.get('ai_cleaning_enabled', False):
        st.session_state.ai_autopilot_enabled = False
        st.session_state.ai_autopilot_request = None


def invalidate_ai_undo_after_manual_edit():
    """A one-level AI snapshot is unsafe after subsequent manual edits."""
    st.session_state.ai_undo_snapshot = None


def _trajectory_index_signature(indices, *, sort_values=False):
    """Return a stable, hashable representation of annotation indices."""
    values = []
    for value in indices:
        if isinstance(value, np.generic):
            value = value.item()
        try:
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            value = str(value)
        values.append(value)
    if sort_values:
        values.sort(key=lambda value: (type(value).__name__, repr(value)))
    return tuple(values)


def trajectory_statistics_signature(trajectory_data):
    """Describe every input that can change derived trajectory statistics."""
    excluded = set(trajectory_data.get('excluded_indices', set()))
    return (
        TRAJECTORY_STATISTICS_SCHEMA_VERSION,
        float(DIVISION_CONFLICT_DROP_RATIO),
        int(trajectory_data.get('_trajectory_data_revision', 0)),
        _trajectory_index_signature(trajectory_data.get('div_indices', [])),
        _trajectory_index_signature(excluded, sort_values=True),
    )


def mark_trajectory_data_changed(trajectory_data):
    """Invalidate derived state after replacing or editing raw trajectory data."""
    trajectory_data['_trajectory_data_revision'] = int(
        trajectory_data.get('_trajectory_data_revision', 0)
    ) + 1
    trajectory_data.pop('_trajectory_stats_signature', None)


def ensure_trajectory_statistics(trajectory_data, *, force=False):
    """Synchronize derived state only when raw labels or data have changed."""
    trajectory_data['excluded_indices'] = set(
        trajectory_data.get('excluded_indices', set())
    )
    signature = trajectory_statistics_signature(trajectory_data)
    required_keys = (
        'effective_div_indices',
        'division_resolution',
        'division_resolution_summary',
        'segments',
        'division_time',
        'growth_rate',
        '_trajectory_stats_resolution',
    )
    cache_is_current = (
        not force
        and trajectory_data.get('_trajectory_stats_signature') == signature
        and trajectory_data.get('_trajectory_stats_schema')
        == TRAJECTORY_STATISTICS_SCHEMA_VERSION
        and all(key in trajectory_data for key in required_keys)
    )
    if cache_is_current:
        return trajectory_data['_trajectory_stats_resolution']

    trajectory = trajectory_data['trajectory']
    excluded = set(trajectory_data.get('excluded_indices', set()))
    resolution = resolve_division_exclusion_conflicts(
        trajectory,
        trajectory_data.get('div_indices', []),
        excluded,
        division_drop_ratio=DIVISION_CONFLICT_DROP_RATIO,
    )
    trajectory_data['effective_div_indices'] = list(
        resolution.effective_division_indices
    )
    trajectory_data['division_resolution'] = resolution.decisions.copy(deep=True)
    trajectory_data['division_resolution_summary'] = resolution.summary_counts
    remaining = trajectory[
        ~trajectory['fov_cell_index'].isin(excluded)
    ].copy()
    remaining_time = pd.to_numeric(remaining['time_h'], errors='coerce')
    remaining_length = pd.to_numeric(remaining['spine_length'], errors='coerce')
    remaining_time_values = remaining_time.to_numpy(dtype=float)
    remaining_length_values = remaining_length.to_numpy(dtype=float)
    statistics_valid = (
        np.isfinite(remaining_time_values)
        & np.isfinite(remaining_length_values)
        & (remaining_length_values > 0.0)
    )
    remaining = remaining.loc[statistics_valid].copy()
    remaining['time_h'] = remaining_time.loc[statistics_valid].to_numpy(dtype=float)
    remaining['spine_length'] = remaining_length.loc[
        statistics_valid
    ].to_numpy(dtype=float)
    remaining = remaining.sort_values('time_h', kind='stable')
    segments, division_time, growth_rate = compute_traj_stats(
        remaining,
        trajectory_data['effective_div_indices'],
    )
    trajectory_data['segments'] = segments
    trajectory_data['division_time'] = division_time
    trajectory_data['growth_rate'] = growth_rate
    trajectory_data['_trajectory_stats_resolution'] = resolution
    trajectory_data['_trajectory_stats_signature'] = signature
    trajectory_data['_trajectory_stats_schema'] = (
        TRAJECTORY_STATISTICS_SCHEMA_VERSION
    )
    trajectory_data['_trajectory_stats_recompute_count'] = int(
        trajectory_data.get('_trajectory_stats_recompute_count', 0)
    ) + 1
    return resolution


def recompute_trajectory_statistics(trajectory_data):
    """Compatibility wrapper for callers that explicitly require a refresh."""
    return ensure_trajectory_statistics(trajectory_data, force=True)


def selected_fov_cell_indices(plot_state, allowed_indices=None):
    """Extract unique cell IDs from any selectable Plotly trace schema."""
    selected_indices = set()
    if not plot_state:
        return selected_indices
    selection = plot_state.get("selection", {})
    for point in selection.get("points", []):
        customdata = point.get("customdata")
        if customdata is None:
            continue
        if isinstance(customdata, (list, tuple, np.ndarray)):
            if len(customdata) == 0:
                continue
            customdata = customdata[0]
        try:
            selected_indices.add(int(customdata))
        except (TypeError, ValueError, OverflowError):
            continue
    if allowed_indices is not None:
        selected_indices.intersection_update(set(allowed_indices))
    return selected_indices


def trajectory_plot_runs(
    trajectory,
    excluded_indices,
    boundary_indices=(),
    *,
    presorted=False,
):
    """Return ``(group, frame)`` runs split at exclusions and boundaries."""
    ordered = (
        trajectory
        if presorted
        else trajectory.sort_values('time_h', kind='stable')
    )
    excluded = set(excluded_indices)
    boundaries = set(boundary_indices)
    runs = []
    run_start = None
    boundary_group = 0

    for position, cell_id in enumerate(ordered['fov_cell_index'].tolist()):
        if cell_id in excluded:
            if run_start is not None:
                runs.append((
                    boundary_group,
                    ordered.iloc[run_start:position].copy(),
                ))
                run_start = None
            continue

        if run_start is None:
            run_start = position
        if cell_id in boundaries:
            runs.append((
                boundary_group,
                ordered.iloc[run_start:position + 1].copy(),
            ))
            run_start = None
            boundary_group += 1

    if run_start is not None:
        runs.append((boundary_group, ordered.iloc[run_start:].copy()))
    return runs


def ai_scope_keys(scope, ordered_keys, current_key):
    if scope == "Current":
        return [current_key]
    if scope == "Selected":
        return [key for key in ordered_keys if key in st.session_state.selected_keys]
    return list(ordered_keys)


def ai_preview_scope_counts(scope_keys):
    """Count AI suggestions once per preview/threshold/exclusion revision."""
    threshold_signature = (
        float(st.session_state.ai_division_threshold),
        float(st.session_state.ai_excluded_threshold),
    )
    exclusion_signature = tuple(
        (
            key,
            _trajectory_index_signature(
                st.session_state.trajectory_dict[key].get(
                    'excluded_indices', set()
                ),
                sort_values=True,
            ),
        )
        for key in scope_keys
    )
    signature = (
        int(st.session_state.get('ai_predictions_revision', 0)),
        tuple(scope_keys),
        threshold_signature,
        exclusion_signature,
    )
    cached = st.session_state.get('ai_preview_count_cache', {})
    if cached.get('signature') == signature:
        return cached['counts']

    division_count = 0
    excluded_count = 0
    for key in scope_keys:
        suggested_divisions, suggested_exclusions = suggested_indices(
            st.session_state.ai_predictions[key],
            division_threshold=threshold_signature[0],
            excluded_threshold=threshold_signature[1],
        )
        division_count += len(suggested_divisions)
        applied_exclusions = set(
            st.session_state.trajectory_dict[key].get(
                'excluded_indices', set()
            )
        )
        excluded_count += len(suggested_exclusions - applied_exclusions)

    counts = (division_count, excluded_count)
    st.session_state.ai_preview_count_cache = {
        'signature': signature,
        'counts': counts,
    }
    return counts


def store_ai_predictions(predictions, checkpoint_signature):
    """Store one model-versioned prediction batch and invalidate count caches."""
    if st.session_state.ai_prediction_signature != checkpoint_signature:
        st.session_state.ai_predictions = {}
    st.session_state.ai_predictions.update(predictions)
    st.session_state.ai_predictions_revision = int(
        st.session_state.get('ai_predictions_revision', 0)
    ) + 1
    st.session_state.ai_preview_count_cache = {}
    st.session_state.ai_prediction_signature = checkpoint_signature
    st.session_state.ai_prediction_checkpoint = checkpoint_signature[0]


def apply_ai_input_preflight(trajectory_keys):
    """Force objectively invalid model inputs into the exclusion set.

    Validation is completed for the whole requested scope before any state is
    changed. Invalid rows remain in the raw trajectory and plot data, while the
    inference adapter omits them from model normalization and scoring.
    """

    keys = list(trajectory_keys)
    results = {}
    signatures = {}
    for key in keys:
        if key not in st.session_state.trajectory_dict:
            raise KeyError(f"Unknown trajectory: {key}")
        entry = st.session_state.trajectory_dict[key]
        trajectory = entry['trajectory']
        result = preflight_ai_trajectory_inputs(trajectory)
        results[key] = result
        signatures[key] = tuple(
            zip(
                trajectory['fov_cell_index'].map(repr).tolist(),
                trajectory['time_h'].map(repr).tolist(),
                trajectory['spine_length'].map(repr).tolist(),
                result.decisions['ai_input_valid'].tolist(),
                result.decisions['ai_input_invalid_reason'].tolist(),
            )
        )

    newly_excluded = 0
    invalid_total = 0
    cleared_prediction = False
    for key in keys:
        entry = st.session_state.trajectory_dict[key]
        result = results[key]
        invalid = set(result.invalid_indices)
        current_exclusions = set(entry.get('excluded_indices', set()))
        additions = invalid.difference(current_exclusions)
        previous_signature = entry.get('_ai_input_preflight_signature')

        entry['ai_input_validation'] = result.decisions.copy(deep=True)
        entry['ai_invalid_indices'] = invalid
        entry['ai_input_validation_summary'] = {
            'valid': result.valid_count,
            'invalid': result.invalid_count,
        }
        entry['_ai_input_preflight_signature'] = signatures[key]
        entry['excluded_indices'] = current_exclusions | invalid
        invalid_total += result.invalid_count
        newly_excluded += len(additions)

        if invalid:
            ensure_trajectory_statistics(entry)
        if (
            previous_signature != signatures[key]
            and key in st.session_state.ai_predictions
        ):
            st.session_state.ai_predictions.pop(key, None)
            cleared_prediction = True

    if cleared_prediction:
        st.session_state.ai_predictions_revision = int(
            st.session_state.get('ai_predictions_revision', 0)
        ) + 1
        st.session_state.ai_preview_count_cache = {}
        if not st.session_state.ai_predictions:
            st.session_state.ai_prediction_signature = None
            st.session_state.ai_prediction_checkpoint = None
    if not save_curation():
        raise CurationStoreError(st.session_state.curation_error)
    return results, invalid_total, newly_excluded


def apply_ai_autopilot_to_trajectory(
    trajectory_key,
    checkpoint_signature,
    ai_bundle,
):
    """Apply AI to an unreviewed destination; reviewed labels are protected."""
    entry = st.session_state.trajectory_dict.get(trajectory_key)
    if entry is None:
        raise KeyError(f"Unknown trajectory: {trajectory_key}")
    if entry.get("_review_status") == "reviewed":
        st.session_state.ai_autopilot_last_key = trajectory_key
        st.session_state.ai_autopilot_last_status = "skipped_reviewed"
        st.session_state.ai_last_message = f"Auto pilot skipped {trajectory_key}: already reviewed."
        st.session_state.ai_model_error = None
        return None
    if not st.session_state.get('ai_cleaning_enabled', False):
        raise RuntimeError("AI-assisted cleaning is disabled")
    if checkpoint_signature is None or ai_bundle is None:
        raise RuntimeError("No loaded AI model is available")
    if st.session_state.ai_model_signature != checkpoint_signature:
        raise RuntimeError("The configured AI model must be loaded again")
    if not (
        st.session_state.ai_apply_divisions
        or st.session_state.ai_apply_exclusions
    ):
        raise RuntimeError("Enable division and/or exclusion application")
    if trajectory_key not in st.session_state.trajectory_dict:
        raise KeyError(f"Unknown trajectory: {trajectory_key}")

    _, invalid_input_count, newly_excluded_count = apply_ai_input_preflight(
        [trajectory_key]
    )

    prediction_is_current = (
        st.session_state.ai_prediction_signature == checkpoint_signature
        and trajectory_key in st.session_state.ai_predictions
    )
    prediction_source = "cached"
    if not prediction_is_current:
        model, normalizer, checkpoint_thresholds, _ = ai_bundle
        predictions = predict_trajectory_scope(
            model,
            normalizer,
            checkpoint_thresholds,
            st.session_state.trajectory_dict,
            [trajectory_key],
            device=checkpoint_signature[-1],
        )
        store_ai_predictions(predictions, checkpoint_signature)
        prediction_source = "new"

    prediction = st.session_state.ai_predictions[trajectory_key]
    entry = st.session_state.trajectory_dict[trajectory_key]
    before_labels = (
        frozenset(entry.get('div_indices', [])),
        frozenset(entry.get('excluded_indices', set())),
    )
    previous_undo_snapshot = st.session_state.ai_undo_snapshot
    snapshot = None
    try:
        snapshot = apply_ai_predictions(
            st.session_state.trajectory_dict,
            {trajectory_key: prediction},
            [trajectory_key],
            division_threshold=st.session_state.ai_division_threshold,
            excluded_threshold=st.session_state.ai_excluded_threshold,
            apply_divisions=st.session_state.ai_apply_divisions,
            apply_exclusions=st.session_state.ai_apply_exclusions,
            policy=st.session_state.ai_apply_policy.lower(),
        )
        resolution = ensure_trajectory_statistics(entry)
    except Exception:
        if snapshot is not None:
            restore_cleaning_snapshot(
                st.session_state.trajectory_dict,
                snapshot,
            )
            ensure_trajectory_statistics(entry)
        raise

    after_labels = (
        frozenset(entry.get('div_indices', [])),
        frozenset(entry.get('excluded_indices', set())),
    )
    if after_labels != before_labels:
        st.session_state.ai_undo_snapshot = snapshot
    else:
        st.session_state.ai_undo_snapshot = previous_undo_snapshot

    suggested_divisions, suggested_exclusions = suggested_indices(
        prediction,
        division_threshold=st.session_state.ai_division_threshold,
        excluded_threshold=st.session_state.ai_excluded_threshold,
    )
    summary = resolution.summary_counts
    st.session_state.ai_autopilot_completed_count = int(
        st.session_state.get('ai_autopilot_completed_count', 0)
    ) + 1
    st.session_state.ai_autopilot_last_key = trajectory_key
    st.session_state.ai_autopilot_last_status = "success"
    st.session_state.ai_model_error = None
    st.session_state.ai_last_message = (
        f"Auto pilot applied {prediction_source} predictions to "
        f"{trajectory_key}: {len(suggested_divisions)} division and "
        f"{len(suggested_exclusions)} exclusion suggestions; "
        f"{summary['shifted_previous']} shifted, "
        f"{summary['coalesced_previous']} coalesced, and "
        f"{summary['needs_review']} need review; "
        f"{invalid_input_count} invalid input point(s) excluded "
        f"({newly_excluded_count} newly added)."
    )
    if not save_curation():
        raise CurationStoreError(st.session_state.curation_error)
    return resolution


def request_next_trajectory():
    """Confirm/save the current review, then navigate or apply destination AI."""
    keys = list(st.session_state.get('trajectory_dict', {}))
    if not keys:
        return
    source_index = int(st.session_state.get('current_idx', 0)) % len(keys)
    source_key = keys[source_index]
    if not save_curation(mark_reviewed=source_key):
        return
    # Undo must not restore older AI labels over a completed review.
    undo = st.session_state.get('ai_undo_snapshot')
    if undo is not None and source_key in undo:
        remaining = {key: value for key, value in undo.items() if key != source_key}
        st.session_state.ai_undo_snapshot = remaining or None
    if not st.session_state.get('ai_autopilot_enabled', False):
        navigate_trajectory(1)
        return

    target_index = (source_index + 1) % len(keys)
    if st.session_state.trajectory_dict[keys[target_index]].get('_review_status') == 'reviewed':
        apply_ai_autopilot_to_trajectory(keys[target_index], None, None)
        st.session_state.ai_autopilot_request = None
        navigate_trajectory(1)
        return
    if target_index < source_index:
        # Auto pilot will wrap past the last trajectory back to the first.
        st.session_state.review_completed_notice_pending = True
    request_id = int(st.session_state.get('ai_autopilot_request_id', 0)) + 1
    st.session_state.ai_autopilot_request_id = request_id
    st.session_state.ai_autopilot_request = {
        'request_id': request_id,
        'source_key': keys[source_index],
        'target_key': keys[target_index],
        'target_index': target_index,
        'model_signature': st.session_state.get('ai_model_signature'),
    }


def toggle_current_trajectory_selection():
    """Toggle export/statistics membership without invalidating the plot."""
    keys = list(st.session_state.get('trajectory_dict', {}))
    if not keys:
        return
    current_index = int(st.session_state.get('current_idx', 0)) % len(keys)
    current_key = keys[current_index]
    selected = set(st.session_state.get('selected_keys', set()))
    if current_key in selected:
        selected.remove(current_key)
    else:
        selected.add(current_key)
    st.session_state.selected_keys = selected
    save_curation()
    if st.session_state.get('show_selected_statistics', False):
        st.session_state.selected_statistics_refresh_requested = True


def trajectory_keys_for_fov(trajectory_keys, fov_index):
    """Match the numeric FOV in loaded keys, including zero-padded indices."""
    matches = set()
    for key in trajectory_keys:
        match = re.fullmatch(r"FOV_([0-9]+)_CH_.+", str(key))
        if match and int(match.group(1)) == fov_index:
            matches.add(key)
    return matches


def process_ai_autopilot_request(
    request,
    checkpoint_signature,
    ai_bundle,
):
    """Apply a queued destination request and return whether it was handled."""
    if not request:
        return False
    keys = list(st.session_state.get('trajectory_dict', {}))
    try:
        source_index = int(st.session_state.get('current_idx', 0))
        if (
            not keys
            or not 0 <= source_index < len(keys)
            or keys[source_index] != request.get('source_key')
        ):
            raise RuntimeError("The source trajectory changed before auto pilot ran")
        target_index = int(request['target_index'])
        if not keys or not 0 <= target_index < len(keys):
            raise RuntimeError("The trajectory list changed before auto pilot ran")
        target_key = request['target_key']
        if keys[target_index] != target_key:
            raise RuntimeError("The destination trajectory changed before auto pilot ran")
        if (st.session_state.trajectory_dict[target_key].get('_review_status') != 'reviewed'
                and request.get('model_signature') != checkpoint_signature):
            raise RuntimeError("The AI model changed before auto pilot ran")

        apply_ai_autopilot_to_trajectory(
            target_key,
            checkpoint_signature,
            ai_bundle,
        )
        if not save_curation(current_key=target_key):
            raise RuntimeError(st.session_state.curation_error)
        st.session_state.current_idx = target_index
        advance_trajectory_plot_epoch()
    except Exception as exc:
        target_key = request.get('target_key', 'next trajectory')
        st.session_state.ai_autopilot_last_key = target_key
        st.session_state.ai_autopilot_last_status = "error"
        st.session_state.ai_model_error = (
            f"Auto pilot failed for {target_key}: {exc}"
        )
        st.session_state.ai_last_message = st.session_state.ai_model_error
    return True


def advance_trajectory_plot_epoch():
    """Reset Plotly interaction state so a lasso cannot leak to another view."""
    st.session_state.trajectory_plot_epoch = int(
        st.session_state.get('trajectory_plot_epoch', 0)
    ) + 1
    st.session_state.reset_trajectory_plot_state = True


def navigate_trajectory(step):
    """Move the session-wide viewer cursor without rerunning the full app."""
    keys = list(st.session_state.get('trajectory_dict', {}))
    if not keys:
        return
    previous_idx = int(st.session_state.get('current_idx', 0)) % len(keys)
    new_idx = (previous_idx + int(step)) % len(keys)
    if not save_curation(current_key=keys[new_idx]):
        return
    st.session_state.current_idx = new_idx
    if int(step) > 0 and new_idx < previous_idx:
        # Forward navigation wrapped past the last trajectory back to the first.
        st.session_state.review_completed_notice_pending = True
    advance_trajectory_plot_epoch()


def jump_to_trajectory(widget_key):
    """Jump to a 1-based trajectory number without applying AI or wrapping."""
    keys = list(st.session_state.get('trajectory_dict', {}))
    if not keys:
        return
    target_index = min(
        max(int(st.session_state[widget_key]) - 1, 0), len(keys) - 1
    )
    st.session_state.ai_autopilot_request = None
    if target_index != st.session_state.get('current_idx', 0):
        if not save_curation(current_key=keys[target_index]):
            return
        st.session_state.current_idx = target_index
        advance_trajectory_plot_epoch()


def canonical_labeled_export(trajectory_dict, builder):
    """Match SQL view row order and full dataset measurement-column union."""

    selected = [
        builder(entry, key) for key, entry in trajectory_dict.items()
        if key in st.session_state.selected_keys
    ]
    label_columns = ["trajectory_key", "division_point", "excluded_point",
                     "effective_division_point", "cell_cycle_index"]
    measurement_columns = []
    for entry in trajectory_dict.values():
        measurement_columns.extend(
            column for column in entry['trajectory'].columns
            if column not in measurement_columns
        )
    if not selected:
        return pd.DataFrame(columns=measurement_columns + label_columns)
    return pd.concat(selected, ignore_index=True).reindex(columns=measurement_columns + label_columns)


def build_labeled_trajectory_export(trajectory_data, trajectory_key):
    """Return one trajectory's rows with the canonical curation labels.

    This is the single schema used by both "Export with Labels" (batch) and
    "Export Current Trajectory" (single). Rows are sorted by ``time_h``;
    excluded rows are kept and flagged, never deleted. An effective division
    point is the final revised last observation of a cell cycle; the row after
    it starts the next ``cell_cycle_index``.
    """
    resolution = ensure_trajectory_statistics(trajectory_data)
    df = trajectory_data['trajectory'].copy()
    df['trajectory_key'] = trajectory_key

    # 1. division_point (raw labels, as marked)
    df['division_point'] = df['fov_cell_index'].isin(
        trajectory_data['div_indices']
    )

    # 2. excluded_point
    excluded = trajectory_data.get('excluded_indices')
    df['excluded_point'] = (
        df['fov_cell_index'].isin(excluded) if excluded is not None else False
    )

    # 3. Effective boundaries after division/exclusion conflict resolution.
    # Raw labels remain untouched even when a safe boundary was moved to the
    # previous valid measurement.
    df = df.sort_values('time_h', kind='stable')
    df['effective_division_point'] = df['fov_cell_index'].isin(
        resolution.effective_division_indices
    )
    df['cell_cycle_index'] = (
        df['effective_division_point']
        .shift(1, fill_value=False)
        .cumsum()
        .astype(int)
    )
    return df


def save_curation(*, current_key=None, mark_reviewed=None):
    """Commit edits before a rerun/navigation; leave failed changes in memory."""
    success = flush_curation(
        st.session_state, build_labeled_trajectory_export,
        current_key=current_key, mark_reviewed=mark_reviewed,
    )
    if not success:
        st.session_state.ai_autopilot_enabled = False
        st.session_state.ai_autopilot_request = None
    return success


def rerun_with_curation():
    save_curation()
    st.rerun(scope="app")


def save_current_session_to_database():
    """Widget callback: preserve a session created before autosave was enabled."""
    if st.session_state.get("curation_dataset_id"):
        return
    try:
        save_current_session_to_curation(
            st.session_state, data_path, build_labeled_trajectory_export,
        )
        close_image_review()
        st.session_state.trajectory_figure_cache = {}
        reset_ai_working_state()
        st.session_state.curation_migration_message = (
            f"Current session saved as dataset {st.session_state.curation_dataset_id}."
        )
    except Exception as exc:
        st.session_state.curation_error = f"Current-session save failed: {exc}"


def recover_after_save_conflict():
    """Keep a durable local copy before accepting the database's newer revision."""
    try:
        recovery_path = save_recovery_copy(st.session_state, build_labeled_trajectory_export)
        db_path = st.session_state.curation_db_path
        payload = load_dataset(
            db_path, st.session_state.curation_dataset_id, source_dir=Path(db_path).parent,
        )
        attach_saved_dataset(st.session_state, payload, db_path)
        close_image_review()
        st.session_state.trajectory_figure_cache = {}
        advance_trajectory_plot_epoch()
        reset_ai_working_state()
        st.session_state.curation_recovery_message = str(recovery_path)
    except Exception as exc:
        st.session_state.curation_error = f"Recovery copy/reload failed: {exc}"


def show_curation_save_status():
    if st.session_state.get("curation_migration_message"):
        st.success(st.session_state.pop("curation_migration_message"))
    if st.session_state.get("curation_recovery_message"):
        st.success("Local edits preserved in " + st.session_state.pop("curation_recovery_message"))
    if st.session_state.get('curation_error'):
        st.error(
            "Autosave failed. Changes remain in this session; navigation is paused. "
            + st.session_state.curation_error
        )
        if st.button("Retry autosave", key="retry_curation_save"):
            save_curation()
            rerun_with_curation()
        if st.session_state.get("curation_dataset_id"):
            st.button(
                "Save recovery copy & reload database", key="curation_conflict_recover",
                on_click=recover_after_save_conflict,
                help="Save all local edits to a separate SQLite file, then load the latest database revision.",
            )
    elif st.session_state.get("curation_error") is None and not st.session_state.get("curation_dataset_id") and st.session_state.get("trajectory_dict"):
        st.button(
            "Save current session to SQLite", key="migrate_curation_session",
            on_click=save_current_session_to_database,
            help="Preserve this pre-autosave session as a new version, without overwriting older reviews.",
        )
    elif st.session_state.get('curation_dataset_id'):
        st.caption(
            f"Saved · dataset {st.session_state.curation_dataset_id} · "
            f"revision {st.session_state.curation_revision} · "
            f"{st.session_state.get('curation_last_saved', '')}"
        )


def restore_curation_version(db_path, dataset_id):
    """Run as a widget callback, before restored parameter widgets render."""
    if not save_curation():
        return
    try:
        info = next(version for version in list_datasets(db_path) if version["dataset_id"] == dataset_id)
        files = [Path(db_path).parent / source['filename'] for source in info['sources']]
        sources = fingerprint_sources(files)
        compatible = find_dataset(
            db_path, sources=sources, settings=info['settings'],
            logic_version=str(moma_logic.LOGIC_VERSION),
        )
        if compatible is None or compatible['dataset_id'] != dataset_id:
            raise CurationStoreError(
                "Source CSVs or app logic changed. Use Process Data to create a new "
                "dataset version; the saved review is preserved."
            )
        payload = load_dataset(db_path, dataset_id, source_dir=Path(db_path).parent)
        attach_saved_dataset(st.session_state, payload, db_path)
        for name, value in payload['settings'].items():
            widget_key = CURATION_PARAMETER_KEYS.get(name)
            if widget_key and value is not None:
                st.session_state[widget_key] = value
        close_image_review()
        st.session_state.trajectory_figure_cache = {}
        advance_trajectory_plot_epoch()
        reset_ai_working_state()
        st.session_state.curation_resume_message = f"Resumed dataset {dataset_id}."
    except Exception as exc:
        st.session_state.curation_resume_error = str(exc)


CURATION_PARAMETER_KEYS = {
    'time_unit': 'processing_time_unit', 'div_thresh': 'processing_div_threshold',
    'mode': 'processing_tracking_mode', 'min_duration': 'processing_min_duration',
    'exclude_mode': 'processing_exclude_mode', 'min_spine': 'min_spine_length',
    'enable_start': 'processing_enable_start', 'start_time': 'processing_start_time',
    'enable_stop': 'processing_enable_stop', 'stop_time': 'processing_stop_time',
}

if 'trajectory_dict' not in st.session_state: st.session_state.trajectory_dict = {}
if 'selected_keys' not in st.session_state: st.session_state.selected_keys = set()
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'trajectory_plot_epoch' not in st.session_state: st.session_state.trajectory_plot_epoch = 0
if 'reset_trajectory_plot_state' not in st.session_state: st.session_state.reset_trajectory_plot_state = False
if 'trajectory_figure_cache' not in st.session_state: st.session_state.trajectory_figure_cache = {}
if 'excluded_points' not in st.session_state: st.session_state.excluded_points = {} # key: list of indices

AI_STATE_DEFAULTS = {
    'ai_cleaning_enabled': False,
    'ai_checkpoint_path': DEFAULT_AI_CHECKPOINT,
    'ai_device': 'auto',
    'ai_scope': 'Current',
    'ai_division_threshold': 0.5,
    'ai_excluded_threshold': 0.5,
    'ai_apply_divisions': True,
    'ai_apply_exclusions': True,
    'ai_apply_policy': 'Merge',
    'ai_autopilot_enabled': False,
    'ai_autopilot_request': None,
    'ai_autopilot_request_id': 0,
    'ai_autopilot_completed_count': 0,
    'ai_autopilot_last_key': None,
    'ai_autopilot_last_status': None,
    'ai_predictions': {},
    'ai_predictions_revision': 0,
    'ai_preview_count_cache': {},
    'ai_model_signature': None,
    'ai_model_metadata': {},
    'ai_model_error': None,
    'ai_prediction_signature': None,
    'ai_prediction_checkpoint': None,
    'ai_undo_snapshot': None,
    'ai_last_message': None,
}
for state_key, default_value in AI_STATE_DEFAULTS.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = copy.deepcopy(default_value)

# Streamlit removes widget state when a conditional widget is hidden or an
# earlier manual-action button reruns before the AI controls are rendered.
# Reassigning these keys first makes the complete AI panel durable across both
# paths and guarantees that plot overlays can read its settings earlier.
AI_CONDITIONAL_WIDGET_KEYS = (
    'ai_cleaning_enabled',
    'ai_checkpoint_path',
    'ai_device',
    'ai_scope',
    'ai_division_threshold',
    'ai_excluded_threshold',
    'ai_apply_divisions',
    'ai_apply_exclusions',
    'ai_apply_policy',
    'ai_autopilot_enabled',
)
for state_key in AI_CONDITIONAL_WIDGET_KEYS:
    st.session_state[state_key] = st.session_state[state_key]

if st.session_state.ai_device not in AI_DEVICE_OPTIONS:
    st.session_state.ai_device = 'auto'
if st.session_state.ai_scope not in ('Current', 'Selected', 'All'):
    st.session_state.ai_scope = 'Current'
if st.session_state.ai_apply_policy not in ('Merge', 'Replace'):
    st.session_state.ai_apply_policy = 'Merge'

st.sidebar.title("Parameters")
data_path = st.sidebar.text_input("Data Directory Path", "")
curation_db_path = Path(data_path).expanduser() / DB_FILENAME if data_path else None
if curation_db_path is not None and curation_db_path.is_file():
    try:
        saved_versions = list_datasets(curation_db_path)
    except Exception as exc:
        saved_versions = []
        st.sidebar.error(f"Cannot open saved curation: {exc}")
    if saved_versions:
        versions_by_id = {version['dataset_id']: version for version in saved_versions}
        resume_id = st.sidebar.selectbox(
            "Saved dataset version", options=list(versions_by_id),
            key="curation_resume_version",
            format_func=lambda value: (
                f"Version {value} · {versions_by_id[value].get('reviewed_count', 0)} reviewed · "
                f"{versions_by_id[value].get('trajectory_count', 0)} trajectories"
            ),
        )
        st.sidebar.button(
            "Resume saved review", key="resume_curation", width="stretch",
            on_click=restore_curation_version, args=(str(curation_db_path), resume_id),
        )
if st.session_state.get('curation_resume_error'):
    st.sidebar.error(st.session_state.pop('curation_resume_error'))
if st.session_state.get('curation_resume_message'):
    st.sidebar.success(st.session_state.pop('curation_resume_message'))
# Preserve values restored before conditional parameter widgets render.
for processing_key in CURATION_PARAMETER_KEYS.values():
    if processing_key in st.session_state:
        st.session_state[processing_key] = st.session_state[processing_key]
time_unit = st.sidebar.selectbox("Time Unit", ['s', 'min', 'h', 'd'], index=0, key="processing_time_unit",
                                 help="Unit of the input time_s values. Moma pipeline exports use seconds (s).")
div_thresh = st.sidebar.number_input("Division Threshold", 0.1, 5.0, 1.2, key="processing_div_threshold")
mode = st.sidebar.selectbox("Tracking Mode", ['longest_lineage', 'most_division'], key="processing_tracking_mode")
min_duration = st.sidebar.number_input("Min Duration (h)", 0.0, 100.0, 10.0, key="processing_min_duration")
exclude_mode = st.sidebar.toggle("Exclude Mode", value=True, key="processing_exclude_mode", help="If ON, all trajectories are kept by default and you click 'Drop' to remove them.")

# --- Cell Filter (applied BEFORE division-point finding) ---
st.sidebar.subheader("Cell Filter (pre-division)")
col_smin, col_smax = st.sidebar.columns(2)
with col_smin:
    spine_filter_min = st.number_input("Slider Min (µm)", value=0.0, step=0.5, format="%.2f",
                                       key="spine_filter_slider_min",
                                       help="Lower bound of the Min Spine Length slider.")
with col_smax:
    spine_filter_max = st.number_input("Slider Max (µm)", value=10.0, step=0.5, format="%.2f",
                                       key="spine_filter_slider_max",
                                       help="Upper bound of the Min Spine Length slider.")
if spine_filter_min >= spine_filter_max:
    st.sidebar.warning("Cell Filter: Slider Min must be smaller than Slider Max; using Min + 1.0 as Max.")
    spine_filter_max = spine_filter_min + 1.0
# Keep the slider value inside the (possibly updated) bounds before the widget is built.
if "min_spine_length" in st.session_state:
    st.session_state.min_spine_length = float(
        min(max(st.session_state.min_spine_length, spine_filter_min), spine_filter_max)
    )
min_spine = st.sidebar.slider(
    "Min Spine Length (µm)",
    min_value=float(spine_filter_min),
    max_value=float(spine_filter_max),
    value=1.0,
    step=0.1,
    format="%.2f",
    key="min_spine_length",
    help=("Cells with spine_length below this threshold are removed BEFORE lineage "
          "tracing / division-point finding. 0 disables the filter. "
          "Takes effect on the next 'Process Data' run."),
)

# --- Time Range Filter ---
st.sidebar.subheader("Time Range Filter")
col_start, col_stop = st.sidebar.columns(2)
with col_start:
    enable_start = st.checkbox("Enable Start", value=False, key="processing_enable_start")
    start_time = st.number_input("Start (h)", value=0.0, key="processing_start_time") if enable_start else None
with col_stop:
    enable_stop = st.checkbox("Enable Stop", value=False, key="processing_enable_stop")
    stop_time = st.number_input("Stop (h)", value=100.0, key="processing_stop_time") if enable_stop else None

num_cores = st.sidebar.slider("Number of Cores", 1, os.cpu_count(), 4)

if st.sidebar.button("Process Data") and save_curation():
    if os.path.isdir(data_path):
        fov_files = sorted(
            str(path) for path in Path(data_path).glob('*.csv')
            if 'selected' not in path.name
        )
        settings = {
            'time_unit': time_unit, 'div_thresh': div_thresh, 'mode': mode,
            'min_duration': min_duration, 'exclude_mode': exclude_mode,
            'min_spine': min_spine, 'enable_start': enable_start,
            'start_time': start_time, 'enable_stop': enable_stop, 'stop_time': stop_time,
        }
        try:
            if not fov_files:
                raise CurationStoreError("No source CSV files found in this directory.")
            sources = fingerprint_sources(fov_files)
            db_path = Path(data_path) / DB_FILENAME
            existing = find_dataset(
                db_path, sources=sources, settings=settings,
                logic_version=str(moma_logic.LOGIC_VERSION),
            )
            if existing is not None:
                payload = load_dataset(db_path, existing['dataset_id'], source_dir=data_path)
                attach_saved_dataset(st.session_state, payload, db_path)
                st.success(f"Resumed saved dataset {existing['dataset_id']}.")
            else:
                tasks = []
                for f_path in fov_files:
                    df = pd.read_csv(f_path)
                    name = os.path.basename(f_path).split('_')[1] if '_' in os.path.basename(f_path) else os.path.basename(f_path)
                    df['fov_cell_index'] = np.arange(len(df))
                    for c in df['channels'].unique():
                        tasks.append((name, c, df[df['channels'] == c].copy(), str(Path(f_path).resolve())))
                traj_dict = {}
                pb = st.progress(0.0, text="Processing...")
                with Parallel(n_jobs=num_cores, return_as="generator") as parallel:
                    gen = parallel(delayed(process_channel_worker)(
                        t[0], t[1], t[2], div_thresh, time_unit, mode, min_duration, start_time, stop_time,
                        min_spine_length=min_spine,
                    ) for t in tasks)
                    for i, res in enumerate(gen):
                        if res:
                            if res[0] in traj_dict:
                                raise CurationStoreError(f"Duplicate trajectory key: {res[0]}")
                            res[1]['source_csv_path'] = tasks[i][3]
                            traj_dict[res[0]] = res[1]
                        pb.progress((i + 1) / len(tasks))
                if fingerprint_sources(fov_files) != sources:
                    raise CurationStoreError("Source CSVs changed while processing; retry Process Data.")
                if not traj_dict:
                    raise CurationStoreError("No trajectories passed the processing filters.")
                for entry in traj_dict.values():
                    entry.setdefault('excluded_indices', set())
                    entry.setdefault('original_div_indices', list(entry['div_indices']))
                    entry['_review_status'] = 'unreviewed'
                    entry['_reviewed_at'] = None
                    entry['trajectory'] = entry['trajectory'].sort_values('time_h', kind='stable').copy()
                    mark_trajectory_data_changed(entry)
                    ensure_trajectory_statistics(entry)
                # Detach the previous version before installing newly reconstructed data.
                for state_key in ('curation_dataset_id', 'curation_db_path', 'curation_error'):
                    st.session_state.pop(state_key, None)
                st.session_state.trajectory_dict = traj_dict
                st.session_state.selected_keys = set(traj_dict) if exclude_mode else set()
                st.session_state.current_idx = 0
                try:
                    initialize_curation(
                        st.session_state, db_path, sources=sources, settings=settings,
                        logic_version=str(moma_logic.LOGIC_VERSION), builder=build_labeled_trajectory_export,
                    )
                except Exception as exc:
                    st.session_state.curation_error = str(exc)
                    st.error(f"Initial database save failed. Trajectories remain in memory: {exc}")
                else:
                    st.success(f"Loaded and saved {len(traj_dict)} trajectories from {len(tasks)} channels.")
            close_image_review()
            st.session_state.trajectory_figure_cache = {}
            advance_trajectory_plot_epoch()
            reset_ai_working_state()
        except Exception as exc:
            st.sidebar.error(f"Process Data failed: {exc}")
    else:
        st.sidebar.error("Choose an existing data directory.")

# Sidebar actions rerun the whole app, so update selection before the viewer
# and aggregate statistics are rendered.
st.sidebar.subheader("Drop trajectories by FOV")
st.sidebar.caption(
    "Drop every channel in a FOV from selected exports and statistics."
)
fov_to_drop = st.sidebar.number_input(
    "FOV index (0-based)",
    min_value=0,
    value=0,
    step=1,
    key="drop_fov_index",
    disabled=not st.session_state.trajectory_dict,
    help="Enter the FOV number in the trajectory name, e.g. 0 for FOV_0_CH_16.",
)
fov_trajectory_keys = trajectory_keys_for_fov(
    st.session_state.trajectory_dict, fov_to_drop
)
if st.sidebar.button(
    "Drop all trajectories in FOV",
    key="drop_fov_trajectories",
    width="stretch",
    disabled=not st.session_state.trajectory_dict,
    help="Mark all trajectories in this FOV as dropped. You can keep them individually afterward.",
):
    selected = set(st.session_state.selected_keys)
    dropped_count = len(selected & fov_trajectory_keys)
    if not fov_trajectory_keys:
        st.sidebar.warning(f"No loaded trajectories match FOV {fov_to_drop}.")
    elif not dropped_count:
        st.sidebar.info(f"All trajectories in FOV {fov_to_drop} are already dropped.")
    else:
        st.session_state.selected_keys = selected - fov_trajectory_keys
        if save_curation():
            st.sidebar.success(f"Dropped {dropped_count} trajectories from FOV {fov_to_drop}.")
        else:
            st.sidebar.error("FOV drop is not saved yet. Retry autosave before navigating.")
if st.session_state.trajectory_dict:
    st.sidebar.caption(
        f"FOV {fov_to_drop}: {len(fov_trajectory_keys)} trajectories loaded."
    )

st.session_state.curation_current_parameters = {
    'time_unit': time_unit, 'div_thresh': div_thresh, 'mode': mode,
    'min_duration': min_duration, 'exclude_mode': exclude_mode,
    'min_spine': min_spine, 'enable_start': enable_start,
    'start_time': start_time, 'enable_stop': enable_stop, 'stop_time': stop_time,
}
st.session_state.curation_current_logic_version = str(moma_logic.LOGIC_VERSION)
# AI configuration is session-wide rather than trajectory-specific. Rendering
# these widgets in the sidebar before any navigation button can trigger a rerun
# keeps their state attached for every trajectory in the current data set.
checkpoint_signature = None
ai_bundle = None
model_ready = False
with st.sidebar.expander(
    "🤖 AI-assisted cleaning",
    expanded=bool(st.session_state.ai_cleaning_enabled),
):
    ai_enabled = st.toggle(
        "Enable AI-assisted cleaning",
        key="ai_cleaning_enabled",
        on_change=handle_ai_cleaning_toggle,
        help=(
            "Experimental offline assistance. Predictions are previews until "
            "you click Apply AI Suggestions. This setting applies to every "
            "trajectory."
        ),
    )
    st.caption("The enable state and settings persist across all trajectories.")

    if ai_enabled:
        st.caption("Model configuration")
        model_tag_options = list(AI_MODEL_REGISTRY) + [AI_MODEL_CUSTOM_TAG]
        current_tag = current_ai_model_tag()
        st.selectbox(
            "Model checkpoint",
            options=model_tag_options,
            index=model_tag_options.index(current_tag),
            key="ai_model_tag",
            on_change=apply_ai_model_tag,
            help=(
                "Released checkpoints with usage suggestions, or a custom "
                "path. Pick the model trained for your experiment type."
            ),
        )
        selected_tag = st.session_state.get("ai_model_tag", current_tag)
        _selected_path, selected_suggestion = resolve_ai_model_tag(selected_tag)
        if selected_tag == AI_MODEL_CUSTOM_TAG:
            st.text_input(
                "Custom checkpoint path",
                key="ai_checkpoint_custom_path",
                value=st.session_state.get(
                    "ai_checkpoint_custom_path",
                    st.session_state.get("ai_checkpoint_path", ""),
                ),
                help="Path to a trajectory-cleaning .pt checkpoint.",
                on_change=invalidate_ai_model_selection,
            )
            st.session_state.ai_checkpoint_path = _selected_path
        else:
            st.session_state.ai_checkpoint_path = _selected_path
            if selected_suggestion:
                st.info(selected_suggestion)
            if not Path(_selected_path).is_file():
                st.warning(f"Checkpoint file missing: {_selected_path}")
        st.selectbox(
            "Device",
            options=AI_DEVICE_OPTIONS,
            key="ai_device",
            on_change=invalidate_ai_model_selection,
        )

        checkpoint_path = Path(st.session_state.ai_checkpoint_path).expanduser()
        try:
            checkpoint_signature = configured_ai_checkpoint_signature()
        except Exception as exc:
            st.session_state.ai_model_error = str(exc)

        if st.button(
            "📦 Load / Reload Model",
            use_container_width=True,
            key="ai_load_model",
        ):
            try:
                if checkpoint_signature is None:
                    raise FileNotFoundError(
                        f"Checkpoint not found: {checkpoint_path}"
                    )
                _, _, checkpoint_thresholds, model_metadata = (
                    load_ai_checkpoint_cached(*checkpoint_signature)
                )
                st.session_state.ai_model_signature = checkpoint_signature
                st.session_state.ai_model_metadata = model_metadata
                st.session_state.ai_model_error = None
                clear_ai_predictions()
                st.session_state.ai_division_threshold = float(
                    checkpoint_thresholds[0]
                )
                st.session_state.ai_excluded_threshold = float(
                    checkpoint_thresholds[1]
                )
                st.session_state.ai_last_message = (
                    f"Loaded {Path(checkpoint_signature[0]).name}."
                )
                st.rerun()
            except Exception as exc:
                st.session_state.ai_model_signature = None
                st.session_state.ai_model_error = str(exc)
                clear_ai_predictions()

        model_ready = (
            checkpoint_signature is not None
            and st.session_state.ai_model_signature == checkpoint_signature
        )
        if model_ready:
            try:
                ai_bundle = load_ai_checkpoint_cached(*checkpoint_signature)
            except Exception as exc:
                st.session_state.ai_model_error = str(exc)
                st.session_state.ai_model_signature = None
                model_ready = False

        if model_ready and ai_bundle is not None:
            model_type = (
                ai_bundle[3]
                .get('model_config', {})
                .get('model_type', 'unknown')
            )
            st.success(f"Model ready: {model_type}")
        elif checkpoint_signature is None:
            st.info("Choose a trained checkpoint to enable prediction.")
        else:
            st.warning("Click Load / Reload Model before prediction.")

        if st.session_state.ai_model_error:
            st.error(st.session_state.ai_model_error)

        st.divider()
        st.caption("Suggestion settings")
        st.selectbox(
            "Scope",
            options=("Current", "Selected", "All"),
            key="ai_scope",
            help="Trajectories included when previewing or applying AI labels.",
        )
        st.slider(
            "Division threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="ai_division_threshold",
        )
        st.slider(
            "Noise threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="ai_excluded_threshold",
        )
        st.checkbox("Apply divisions", key="ai_apply_divisions")
        st.checkbox("Apply exclusions", key="ai_apply_exclusions")
        st.radio(
            "Apply policy",
            options=("Merge", "Replace"),
            key="ai_apply_policy",
            horizontal=True,
            help=(
                "Merge adds AI suggestions to current labels. Replace uses only "
                "AI suggestions for the selected label type and scope."
            ),
        )
        autopilot_available = (
            model_ready
            and ai_bundle is not None
            and (
                st.session_state.ai_apply_divisions
                or st.session_state.ai_apply_exclusions
            )
        )
        if not autopilot_available:
            st.session_state.ai_autopilot_enabled = False
        st.toggle(
            "Auto pilot on Next",
            key="ai_autopilot_enabled",
            disabled=not autopilot_available,
            help=(
                "Next confirms the current review, then applies the current "
                "thresholds, label types, and Merge/Replace policy only to an "
                "unreviewed destination. Reviewed trajectories are skipped. Prev never applies AI."
            ),
        )
        if st.session_state.ai_autopilot_enabled:
            st.caption(
                "Auto pilot is active. Reviewed trajectories are protected; "
                "Merge/Replace applies only to unreviewed destinations."
            )

st.markdown(
    "**Keyboard Shortcuts:**  \n"
    "`←` Prev | `→` Next | `k` Keep/Drop Trajectory | "
    "`e` Exclude Points | `r` Restore Selected Exclusions | "
    "`d` Mark Division | `m` Merge Cycles"
)
st.title("🧬 MoMa Cell Lineage Analysis")

if not st.session_state.trajectory_dict:
    st.info("Load data to start.")
else:
    available_fluorescence_channels = sorted(
        {
            column
            for trajectory_data in st.session_state.trajectory_dict.values()
            for column in trajectory_data['trajectory'].columns
            if column not in COLNAME_NOT_FLU_CHANNEL
        },
        key=str,
    )
    with st.sidebar.expander("📊 Display Settings"):
        show_flu = st.toggle(
            "Show Fluorescence",
            value=False,
            key="show_fluorescence",
        )
        show_psd = st.toggle("Show PSD", value=False, key="show_psd")
        st.divider()
        st.caption("Display Channels")
        selected_fluorescence_channels = [
            channel
            for channel in available_fluorescence_channels
            if st.checkbox(channel, value=True, key=f"cb_{channel}")
        ]

    @st.fragment
    def render_trajectory_viewer():
        if st.session_state.pop(
            'selected_statistics_refresh_requested', False
        ):
            rerun_with_curation()

        autopilot_request = st.session_state.pop(
            'ai_autopilot_request', None
        )
        if autopilot_request is not None:
            with st.spinner(
                f"Auto pilot cleaning {autopilot_request.get('target_key', 'next trajectory')}..."
            ):
                request_was_handled = process_ai_autopilot_request(
                    autopilot_request,
                    checkpoint_signature,
                    ai_bundle,
                )
            if request_was_handled:
                rerun_with_curation()

        if st.session_state.pop('reset_trajectory_plot_state', False):
            st.session_state.pop('traj_plot', None)
            st.session_state.pop('selection', None)
        keys = list(st.session_state.trajectory_dict.keys())
        st.session_state.current_idx %= len(keys)
        curr_key = keys[st.session_state.current_idx]
        data = st.session_state.trajectory_dict[curr_key]

        is_sel = curr_key in st.session_state.selected_keys
        review_completed_notice = st.session_state.pop(
            'review_completed_notice_pending', False
        )
        # Keep/Drop styling and the review-complete notice render on EVERY run,
        # before the navigation columns, inside one constant-height block. The
        # button column itself holds only the button, so toggling Keep/Drop (or
        # showing the notice) can never shift any widget's position.
        keep_button_css = "" if is_sel else """
            .st-key-toggle_trajectory_selection_btn button {
                background-color: #ff69b4;
                border-color: #ff1493;
                color: #ffffff;
            }
            .st-key-toggle_trajectory_selection_btn button:hover,
            .st-key-toggle_trajectory_selection_btn button:focus:not(:active) {
                background-color: #ff85c1;
                border-color: #ff1493;
                color: #ffffff;
            }
        """
        # st.toast only renders in a screen corner, so use a centered overlay
        # banner that fades out on its own instead. The banner is fixed-position
        # and therefore does not affect layout when it appears.
        notice_html = (
            '<div class="review-complete-notice">🎉 All trajectories have been '
            'reviewed — back to the first one!</div>'
            if review_completed_notice
            else ""
        )
        st.markdown(
            f"""
            <style>
            {keep_button_css}
            .review-complete-notice {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 1000000;
                background-color: #ffffff;
                color: #d6336c;
                border: 2px solid #ff69b4;
                border-radius: 12px;
                padding: 24px 36px;
                font-size: 1.15rem;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 8px 30px rgba(214, 51, 108, 0.35);
                pointer-events: none;
                animation: review-complete-fade 4s ease-in-out forwards;
            }}
            @keyframes review-complete-fade {{
                0%   {{ opacity: 0; transform: translate(-50%, -50%) scale(0.9); }}
                10%  {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }}
                80%  {{ opacity: 1; }}
                100% {{ opacity: 0; visibility: hidden; }}
            }}
            </style>
            {notice_html}
            """,
            unsafe_allow_html=True,
        )

        # Selection & Navigation
        save_curation()
        show_curation_save_status()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.button(
                "⬅️ Prev",
                on_click=navigate_trajectory,
                args=(-1,),
            )
        with c3:
            st.button(
                "Next ➡️",
                on_click=request_next_trajectory,
                help="Mark this trajectory reviewed, save it, and open the next. Auto pilot skips reviewed destinations.",
            )
        with c2:
            st.button(
                f"{'❌ Drop' if is_sel else '✅ Keep'} Trajectory",
                width='stretch',
                key="toggle_trajectory_selection_btn",
                on_click=toggle_current_trajectory_selection,
            )

        # Synchronize both jump controls before rendering, including after
        # Prev/Next, AI auto pilot, or loading a new set of trajectories.
        trajectory_number = st.session_state.current_idx + 1
        st.session_state.trajectory_jump_slider = trajectory_number
        st.session_state.trajectory_jump_number = trajectory_number
        slider_col, number_col = st.columns([3, 1])
        with slider_col:
            st.select_slider(
                "Trajectory",
                # The browser slider requires two positions even when disabled.
                # Duplicate the sole label to keep the one-trajectory view valid.
                options=range(1, len(keys) + 1) if len(keys) > 1 else (1, 1),
                key="trajectory_jump_slider",
                on_change=jump_to_trajectory,
                args=("trajectory_jump_slider",),
                disabled=len(keys) == 1,
                help="Drag to jump to a trajectory by its number in the loaded list.",
            )
        with number_col:
            st.number_input(
                "Trajectory number",
                min_value=1,
                max_value=len(keys),
                step=1,
                key="trajectory_jump_number",
                on_change=jump_to_trajectory,
                args=("trajectory_jump_number",),
                disabled=len(keys) == 1,
                help="Type a trajectory number and press Enter to jump to it.",
            )

        st.write(f"**Current:** {curr_key} ({st.session_state.current_idx + 1}/{len(keys)}) | **Selected:** {len(st.session_state.selected_keys)}")
        if st.session_state.get("curation_dataset_id"):
            reviewed_count = sum(entry.get("_review_status") == "reviewed" for entry in st.session_state.trajectory_dict.values())
            st.caption(f"Review: {data.get('_review_status', 'unreviewed').replace('_', ' ')} · {reviewed_count}/{len(keys)} reviewed")

        # Normalize older/injected sessions to the mutable set used by all manual
        # and AI exclusion actions.
        data['excluded_indices'] = set(data.get('excluded_indices', set()))
        # Recompute on every current-trajectory render so older Streamlit sessions
        # and externally injected state receive the same conflict resolution.
        ensure_trajectory_statistics(data)

        # Prepare data
        trajectory = data['trajectory']
        traj_df = (
            trajectory
            if trajectory['time_h'].is_monotonic_increasing
            else trajectory.sort_values('time_h', kind='stable')
        )
        active_ai_signature = checkpoint_signature if ai_enabled else None
        sel_channels = [
            channel
            for channel in selected_fluorescence_channels
            if channel in traj_df.columns
        ]
        figure_signature = (
            curr_key,
            id(trajectory),
            len(trajectory),
            data.get('_trajectory_stats_signature'),
            bool(show_flu),
            bool(show_psd),
            tuple(sel_channels),
            active_ai_signature,
            st.session_state.get('ai_prediction_signature'),
            int(st.session_state.get('ai_predictions_revision', 0)),
            float(st.session_state.get('ai_division_threshold', 0.5)),
            float(st.session_state.get('ai_excluded_threshold', 0.5)),
            int(st.session_state.trajectory_plot_epoch),
        )
        figure_cache = st.session_state.get('trajectory_figure_cache', {})
        # Main area: Plot on the left, interactive tools on the right.
        plot_col, tool_col = st.columns([3, 1])
        if figure_cache.get('signature') == figure_signature:
            fig = figure_cache['figure']
        else:
            visible_mask = ~traj_df['fov_cell_index'].isin(data['excluded_indices'])
            plot_df = traj_df[visible_mask]
            excluded_plot_df = traj_df[~visible_mask].copy()
            original_spine_values = pd.to_numeric(
                traj_df['spine_length'], errors='coerce'
            ).to_numpy(dtype=float)
            positive_spine_values = original_spine_values[
                np.isfinite(original_spine_values) & (original_spine_values > 0)
            ]
            spine_display_fallback = (
                float(positive_spine_values.min() * 0.8)
                if positive_spine_values.size
                else 1.0
            )
            display_spine_values = np.where(
                np.isfinite(original_spine_values) & (original_spine_values > 0),
                original_spine_values,
                spine_display_fallback,
            )
            original_spine_by_id = pd.Series(
                original_spine_values,
                index=traj_df['fov_cell_index'],
            )
            display_spine_by_id = pd.Series(
                display_spine_values,
                index=traj_df['fov_cell_index'],
            )
            spine_line_runs = trajectory_plot_runs(
                traj_df,
                data['excluded_indices'],
                data.get('effective_div_indices', []),
                presorted=True,
            )
            fluorescence_line_runs = trajectory_plot_runs(
                traj_df,
                data['excluded_indices'],
                presorted=True,
            )
            # Plotly Visualization - Dynamic Subplots
            rows = 1
            row_map = {"spine": 1}
            titles = ["Spine Length (Log) - Select points to exclude or restore"]

            if show_flu:
                rows += 1
                row_map["flu"] = rows
                titles.append("Fluorescence")
            if show_psd:
                rows += 1
                row_map["psd"] = rows
                titles.append("PSD")

            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=tuple(titles))

            # 1. Spine Length Plot (Segmented) - Always Row 1. Plot-only runs
            # split at both biological boundaries and excluded frames, preventing
            # Plotly from drawing a false connection across a removed observation.
            for cycle_index, df_plot in spine_line_runs:
                trace_color = TRAJECTORY_TRACE_COLORS[
                    cycle_index % len(TRAJECTORY_TRACE_COLORS)
                ]
                fig.add_trace(go.Scatter(
                    x=df_plot['time_h'],
                    y=df_plot['spine_length'],
                    mode='markers+lines',
                    showlegend=False,
                    line=dict(color=trace_color),
                    marker=dict(color=trace_color),
                    customdata=df_plot['fov_cell_index'].tolist(),
                    hovertemplate="Time: %{x:.2f}h<br>Length: %{y:.2f}um<br>ID: %{customdata}<extra></extra>"
                ), row=1, col=1)

            # Applied exclusions are never deleted from the trajectory. They remain
            # selectable as a canonical marker trace regardless of whether their source
            # was a manual action or an applied AI suggestion.
            if not excluded_plot_df.empty:
                original_lengths = excluded_plot_df['fov_cell_index'].map(
                    original_spine_by_id
                ).to_numpy(dtype=float)
                display_lengths = excluded_plot_df['fov_cell_index'].map(
                    display_spine_by_id
                ).to_numpy(dtype=float)
                excluded_customdata = [
                    [
                        int(cell_id),
                        None if not np.isfinite(original_length) else float(original_length),
                        "fallback position" if display_length != original_length else "original position",
                    ]
                    for cell_id, original_length, display_length in zip(
                        excluded_plot_df['fov_cell_index'].tolist(),
                        original_lengths,
                        display_lengths,
                    )
                ]
                fig.add_trace(go.Scatter(
                    x=excluded_plot_df['time_h'],
                    y=display_lengths,
                    mode='markers',
                    name='Applied exclusion',
                    marker=dict(APPLIED_EXCLUSION_MARKER),
                    customdata=excluded_customdata,
                    hovertemplate=(
                        "Applied exclusion<br>Time: %{x:.2f}h"
                        "<br>Original length: %{customdata[1]}um"
                        "<br>ID: %{customdata[0]}<br>%{customdata[2]}"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)

            # AI predictions are preview overlays until the user explicitly applies them.
            ai_preview_df = None
            ai_prediction_is_current = (
                active_ai_signature is not None
                and st.session_state.ai_prediction_signature == active_ai_signature
                and st.session_state.ai_model_signature == active_ai_signature
            )
            if (
                st.session_state.get('ai_cleaning_enabled', False)
                and ai_prediction_is_current
            ):
                ai_preview_df = st.session_state.ai_predictions.get(curr_key)
            if ai_preview_df is not None and not ai_preview_df.empty:
                division_preview_threshold = float(
                    st.session_state.get('ai_division_threshold', 0.5)
                )
                excluded_preview_threshold = float(
                    st.session_state.get('ai_excluded_threshold', 0.5)
                )
                model_scored = (
                    ai_preview_df['ai_input_valid'].astype(bool)
                    if 'ai_input_valid' in ai_preview_df.columns
                    else pd.Series(True, index=ai_preview_df.index)
                )
                division_preview = ai_preview_df[
                    model_scored
                    & (
                        ai_preview_df['pred_division_probability']
                        >= division_preview_threshold
                    )
                ]
                excluded_preview = ai_preview_df[
                    ai_preview_df['pred_excluded_probability']
                    >= excluded_preview_threshold
                ].drop_duplicates('fov_cell_index')
                excluded_preview = excluded_preview[
                    ~excluded_preview['fov_cell_index'].isin(data['excluded_indices'])
                ]

                if not division_preview.empty:
                    division_ids = division_preview['fov_cell_index'].tolist()
                    division_original_lengths = division_preview['fov_cell_index'].map(
                        original_spine_by_id
                    ).to_numpy(dtype=float)
                    division_display_lengths = division_preview['fov_cell_index'].map(
                        display_spine_by_id
                    ).to_numpy(dtype=float)
                    division_custom = [
                        [
                            int(cell_id),
                            float(probability),
                            None if not np.isfinite(original_length) else float(original_length),
                        ]
                        for cell_id, probability, original_length in zip(
                            division_ids,
                            division_preview['pred_division_probability'].tolist(),
                            division_original_lengths,
                        )
                    ]
                    fig.add_trace(go.Scatter(
                        x=division_preview['time_h'],
                        y=division_display_lengths,
                        mode='markers',
                        name='AI division',
                        marker=dict(AI_DIVISION_MARKER),
                        customdata=division_custom,
                        hovertemplate=(
                            "AI division<br>Time: %{x:.2f}h"
                            "<br>Original length: %{customdata[2]}um"
                            "<br>ID: %{customdata[0]}<br>Probability: %{customdata[1]:.3f}"
                            "<extra></extra>"
                        ),
                    ), row=1, col=1)

                if not excluded_preview.empty:
                    excluded_ids = excluded_preview['fov_cell_index'].tolist()
                    excluded_original_lengths = excluded_preview['fov_cell_index'].map(
                        original_spine_by_id
                    ).to_numpy(dtype=float)
                    excluded_display_lengths = excluded_preview['fov_cell_index'].map(
                        display_spine_by_id
                    ).to_numpy(dtype=float)
                    excluded_custom = [
                        [
                            int(cell_id),
                            float(probability),
                            None if not np.isfinite(original_length) else float(original_length),
                        ]
                        for cell_id, probability, original_length in zip(
                            excluded_ids,
                            excluded_preview['pred_excluded_probability'].tolist(),
                            excluded_original_lengths,
                        )
                    ]
                    fig.add_trace(go.Scatter(
                        x=excluded_preview['time_h'],
                        y=excluded_display_lengths,
                        mode='markers',
                        name='AI exclusion',
                        marker=dict(AI_EXCLUSION_MARKER),
                        customdata=excluded_custom,
                        hovertemplate=(
                            "AI exclusion<br>Time: %{x:.2f}h"
                            "<br>Original length: %{customdata[2]}um"
                            "<br>ID: %{customdata[0]}<br>Probability: %{customdata[1]:.3f}"
                            "<extra></extra>"
                        ),
                    ), row=1, col=1)

            # Applied labels keep raw division/exclusion annotations independent. Show
            # where the resolver moved an effective boundary, and keep unresolved
            # conflicts visible even though their noisy source measurement is excluded.
            division_decisions = data.get('division_resolution')
            if isinstance(division_decisions, pd.DataFrame) and not division_decisions.empty:
                shifted_decisions = division_decisions[
                    division_decisions['division_resolution'] == 'shifted_previous'
                ].dropna(subset=['effective_index'])
                if not shifted_decisions.empty:
                    shifted_lookup = (
                        shifted_decisions.drop_duplicates('effective_index')
                        .set_index('effective_index')
                    )
                    shifted_points = traj_df[
                        traj_df['fov_cell_index'].isin(shifted_lookup.index)
                    ].copy()
                    if not shifted_points.empty:
                        shifted_points['division_source_index'] = shifted_points[
                            'fov_cell_index'
                        ].map(shifted_lookup['source_index'])
                        shifted_points['division_length_ratio'] = shifted_points[
                            'fov_cell_index'
                        ].map(shifted_lookup['length_ratio'])
                        shifted_custom = np.column_stack([
                            shifted_points['fov_cell_index'].to_numpy(),
                            shifted_points['division_source_index'].to_numpy(),
                            shifted_points['division_length_ratio'].to_numpy(),
                        ])
                        fig.add_trace(go.Scatter(
                            x=shifted_points['time_h'],
                            y=shifted_points['fov_cell_index'].map(display_spine_by_id),
                            mode='markers',
                            name='Resolved division',
                            marker=dict(
                                symbol='triangle-up-open',
                                size=14,
                                color='#059669',
                                line=dict(width=2),
                            ),
                            customdata=shifted_custom,
                            hovertemplate=(
                                "Resolved division<br>Effective ID: %{customdata[0]:.0f}"
                                "<br>Excluded source ID: %{customdata[1]:.0f}"
                                "<br>Next/previous length: %{customdata[2]:.3f}"
                                "<extra></extra>"
                            ),
                        ), row=1, col=1)
    
                review_decisions = division_decisions[
                    division_decisions['division_resolution'] == 'needs_review'
                ]
                if not review_decisions.empty:
                    review_lookup = review_decisions.set_index('source_index')
                    review_points = traj_df[
                        traj_df['fov_cell_index'].isin(review_lookup.index)
                    ].copy()
                    if not review_points.empty:
                        review_points['division_length_ratio'] = review_points[
                            'fov_cell_index'
                        ].map(review_lookup['length_ratio'])
                        review_original_lengths = review_points['fov_cell_index'].map(
                            original_spine_by_id
                        ).to_numpy(dtype=float)
                        review_custom = [
                            [
                                int(cell_id),
                                None if not np.isfinite(length_ratio) else float(length_ratio),
                                None if not np.isfinite(original_length) else float(original_length),
                            ]
                            for cell_id, length_ratio, original_length in zip(
                                review_points['fov_cell_index'].tolist(),
                                review_points['division_length_ratio'].to_numpy(dtype=float),
                                review_original_lengths,
                            )
                        ]
                        fig.add_trace(go.Scatter(
                            x=review_points['time_h'],
                            y=review_points['fov_cell_index'].map(display_spine_by_id),
                            mode='markers',
                            name='Division needs review',
                            marker=dict(
                                symbol='triangle-down-open',
                                size=14,
                                color='#d97706',
                                line=dict(width=2),
                            ),
                            customdata=review_custom,
                            hovertemplate=(
                                "Division conflict needs review"
                                "<br>Source ID: %{customdata[0]}"
                                "<br>Original length: %{customdata[2]}um"
                                "<br>Next/previous length: %{customdata[1]:.3f}"
                                "<extra></extra>"
                            ),
                        ), row=1, col=1)
    
            # 2. Flu Plot - Conditional
            if show_flu:
                for channel_index, c in enumerate(sel_channels):
                    channel_color = TRAJECTORY_TRACE_COLORS[
                        channel_index % len(TRAJECTORY_TRACE_COLORS)
                    ]
                    for run_index, (_, df_plot) in enumerate(fluorescence_line_runs):
                        fig.add_trace(go.Scatter(
                            x=df_plot['time_h'],
                            y=df_plot[c],
                            name=c,
                            mode='markers+lines',
                            showlegend=run_index == 0,
                            line=dict(color=channel_color),
                            marker=dict(color=channel_color),
                            customdata=df_plot['fov_cell_index'].tolist(),
                            hovertemplate=(
                                "Time: %{x:.2f}h<br>Value: %{y}"
                                "<br>ID: %{customdata}<extra></extra>"
                            ),
                        ), row=row_map["flu"], col=1)

            # 3. PSD Plot - Conditional
            if show_psd and sel_channels:
                sig = plot_df[sel_channels[0]].values
                m = ~np.isnan(sig)
                if m.sum() > 3:
                    fs = 1.0 / np.mean(np.diff(plot_df['time_h'].values[m]))
                    freqs, psd = sg.periodogram(sig[m] - sig[m].mean(), fs=fs)
                    fig.add_trace(go.Scatter(x=freqs, y=psd, name="PSD", line=dict(color='orange')), row=row_map["psd"], col=1)

            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_layout(
                height=300 + 300 * rows,
                hovermode="x unified",
                dragmode="lasso",
                selectionrevision=int(st.session_state.trajectory_plot_epoch),
                uirevision=int(st.session_state.trajectory_plot_epoch),
            )
            st.session_state.trajectory_figure_cache = {
                'signature': figure_signature,
                'figure': fig,
                'build_count': int(figure_cache.get('build_count', 0)) + 1,
            }

        with plot_col:
            plot_key = "traj_plot"
            st.plotly_chart(
                fig,
                width='stretch',
                on_select="rerun",
                key=plot_key,
            )

        with tool_col:
            st.subheader("Data Cleaning")
            if st.button(
                "🖼️ Review Images",
                key="review_images",
                help="Lasso or box select points to inspect their side-channel images and masks.",
                width="stretch",
            ):
                selected_indices = selected_fov_cell_indices(
                    st.session_state.get(plot_key), traj_df['fov_cell_index']
                )
                if not selected_indices:
                    st.info("Select points on the plot first.")
                else:
                    try:
                        snapshot = make_review_snapshot(
                            curr_key, data, selected_indices,
                            data.get('source_csv_path'),
                        )
                        open_image_review(snapshot)
                    except ImageReviewError as exc:
                        st.warning(str(exc))
            render_image_review_if_open(curr_key)
            resolution_summary = data.get('division_resolution_summary', {})
            shifted_count = int(resolution_summary.get('shifted_previous', 0))
            coalesced_count = int(resolution_summary.get('coalesced_previous', 0))
            review_count = int(resolution_summary.get('needs_review', 0))
            if shifted_count or coalesced_count or review_count:
                st.caption(
                    "Division resolution: "
                    f"{shifted_count} shifted · {coalesced_count} coalesced · "
                    f"{review_count} need review"
                )
            if review_count:
                st.warning(
                    f"{review_count} division conflict(s) need manual review. "
                    "They are shown as orange triangles and are not used as cycle "
                    "boundaries."
                )

            input_validation = data.get('ai_input_validation', pd.DataFrame())
            if (
                isinstance(input_validation, pd.DataFrame)
                and not input_validation.empty
                and 'ai_input_valid' in input_validation.columns
            ):
                invalid_input_rows = input_validation[
                    ~input_validation['ai_input_valid'].astype(bool)
                ]
            else:
                invalid_input_rows = pd.DataFrame()
            if not invalid_input_rows.empty:
                invalid_input_ids = invalid_input_rows[
                    'fov_cell_index'
                ].tolist()
                invalid_reason_by_id = invalid_input_rows.set_index(
                    'fov_cell_index'
                )['ai_input_invalid_reason'].to_dict()
                with st.expander(
                    f"Invalid AI inputs ({len(invalid_input_ids)})",
                    expanded=False,
                ):
                    st.caption(
                        "These rows were excluded by deterministic input validation, "
                        "not by model confidence. Rows with invalid time may not have "
                        "a selectable plot marker."
                    )
                    invalid_ids_to_restore = st.multiselect(
                        "Invalid input IDs to restore",
                        options=invalid_input_ids,
                        format_func=lambda cell_id: (
                            f"{cell_id}: {invalid_reason_by_id.get(cell_id, '')}"
                        ),
                        key=f"invalid_ai_input_restore_{curr_key}",
                    )
                    if st.button(
                        "↩️ Restore Chosen Invalid Inputs",
                        disabled=not invalid_ids_to_restore,
                        use_container_width=True,
                        key=f"restore_invalid_ai_inputs_{curr_key}",
                    ):
                        restored_invalid = set(invalid_ids_to_restore) & set(
                            data['excluded_indices']
                        )
                        if restored_invalid:
                            invalidate_ai_undo_after_manual_edit()
                            data['excluded_indices'].difference_update(
                                restored_invalid
                            )
                            ensure_trajectory_statistics(data)
                            st.success(
                                f"Restored {len(restored_invalid)} invalid input "
                                "point(s). They will be excluded again before the "
                                "next AI inference unless their source values change."
                            )
                            advance_trajectory_plot_epoch()
                            rerun_with_curation()
            if st.button("🚫 Exclude Selected", help="Lasso/Box select on the plot first", use_container_width=True):
                plot_state = st.session_state.get(plot_key)
                selected_indices = selected_fov_cell_indices(
                    plot_state,
                    traj_df['fov_cell_index'],
                )
                if selected_indices:
                    indices_to_exclude = (
                        selected_indices - set(data['excluded_indices'])
                    )
                    if indices_to_exclude:
                        invalidate_ai_undo_after_manual_edit()
                        data['excluded_indices'].update(indices_to_exclude)
                        ensure_trajectory_statistics(data)
                        st.success(f"Excluded {len(indices_to_exclude)} point(s).")
                        advance_trajectory_plot_epoch()
                        rerun_with_curation()
                    else:
                        st.info("Every selected point is already excluded.")
                else:
                    st.info("Select points on the plot first.")

            if st.button(
                "↩️ Restore Selected Exclusions",
                help="Select applied red X markers to restore only those observations",
                use_container_width=True,
            ):
                plot_state = st.session_state.get(plot_key)
                selected_indices = selected_fov_cell_indices(
                    plot_state,
                    traj_df['fov_cell_index'],
                )
                indices_to_restore = (
                    selected_indices & set(data['excluded_indices'])
                )
                if indices_to_restore:
                    invalidate_ai_undo_after_manual_edit()
                    data['excluded_indices'].difference_update(indices_to_restore)
                    ensure_trajectory_statistics(data)
                    st.success(f"Restored {len(indices_to_restore)} point(s).")
                    advance_trajectory_plot_epoch()
                    rerun_with_curation()
                elif selected_indices:
                    st.info("The selection contains no applied exclusions.")
                else:
                    st.info("Select applied red X markers first.")

            if st.button(
                "🔄 Reset All Exclusions",
                help="Restore every excluded observation in the current trajectory",
                use_container_width=True,
            ):
                if data['excluded_indices']:
                    invalidate_ai_undo_after_manual_edit()
                    data['excluded_indices'] = set()
                    ensure_trajectory_statistics(data)
                    advance_trajectory_plot_epoch()
                    rerun_with_curation()
                else:
                    st.info("This trajectory has no applied exclusions.")

            st.divider()
            st.subheader("Manual Division")
            if st.button("➕ Mark Division", help="Select point(s); last by time becomes division", use_container_width=True):
                plot_state = st.session_state.get(plot_key)
                if plot_state and "selection" in plot_state and plot_state["selection"]["points"]:
                    selected_points = plot_state["selection"]["points"]
                    last_selected_id = None
                    max_t = -1
                    for p in selected_points:
                        t = p.get("x")
                        cd = p.get("customdata")
                        if t is not None and cd is not None:
                            if t > max_t:
                                max_t = t
                                last_selected_id = cd[0] if isinstance(cd, (list, np.ndarray)) else cd
                    if last_selected_id is not None:
                        cell_id = int(last_selected_id)
                        if cell_id not in data['div_indices']:
                            invalidate_ai_undo_after_manual_edit()
                            data['div_indices'].append(cell_id)
                            ensure_trajectory_statistics(data)
                            st.success(f"Division at {max_t:.2f}h")
                            advance_trajectory_plot_epoch()
                            rerun_with_curation()
                        else:
                            st.warning("Already marked.")
                else:
                    st.info("Select point(s) on plot first.")

            if st.button("🔗 Merge Cycles", help="Select points across a division to merge", use_container_width=True):
                plot_state = st.session_state.get(plot_key)
                if plot_state and "selection" in plot_state and plot_state["selection"]["points"]:
                    selected_points = plot_state["selection"]["points"]
                    selected_by_time = []
                    for selection_order, point in enumerate(selected_points):
                        selected_time = point.get("x")
                        customdata = point.get("customdata")
                        if customdata is None or selected_time is None:
                            continue
                        try:
                            selected_time = float(selected_time)
                        except (TypeError, ValueError, OverflowError):
                            continue
                        if not np.isfinite(selected_time):
                            continue
                        if isinstance(customdata, (list, tuple, np.ndarray)):
                            if len(customdata) == 0:
                                continue
                            customdata = customdata[0]
                        try:
                            selected_id = int(customdata)
                        except (TypeError, ValueError, OverflowError):
                            continue
                        selected_by_time.append(
                            (selected_time, selection_order, selected_id)
                        )
                    if selected_by_time:
                        times = [item[0] for item in selected_by_time]
                        min_t, max_t = min(times), max(times)
                        last_selected_id = max(selected_by_time)[2]
                        divs_to_remove = set()
                        decisions = data.get('division_resolution', pd.DataFrame())
                        protected_sources = set()
                        protected_targets = set()
                        for decision in decisions.itertuples(index=False):
                            effective_idx = decision.effective_index
                            source_matches = decision.source_index == last_selected_id
                            effective_matches = (
                                not pd.isna(effective_idx)
                                and effective_idx == last_selected_id
                            )
                            if source_matches or effective_matches:
                                protected_sources.add(decision.source_index)
                                if not pd.isna(effective_idx):
                                    protected_targets.add(effective_idx)
                        if protected_targets:
                            for decision in decisions.itertuples(index=False):
                                if (
                                    not pd.isna(decision.effective_index)
                                    and decision.effective_index in protected_targets
                                ):
                                    protected_sources.add(decision.source_index)
                        for decision in decisions.itertuples(index=False):
                            boundary_idx = decision.effective_index
                            if pd.isna(boundary_idx):
                                # An unresolved conflict has no effective boundary;
                                # allow selecting its raw source to remove it.
                                boundary_idx = decision.source_index
                            cell_row = traj_df[
                                traj_df['fov_cell_index'] == boundary_idx
                            ]
                            if not cell_row.empty:
                                div_t = cell_row['time_h'].values[0]
                                if min_t <= div_t <= max_t:
                                    # Every raw source coalesced onto the selected
                                    # effective boundary must be removed together.
                                    divs_to_remove.add(decision.source_index)
                        # Merging removes earlier boundaries, but the latest
                        # selected observation remains the end of the merged
                        # cycle when it already represents a raw/effective event.
                        divs_to_remove.difference_update(protected_sources)
                        if divs_to_remove:
                            invalidate_ai_undo_after_manual_edit()
                            data['div_indices'] = [
                                division_index
                                for division_index in data['div_indices']
                                if division_index not in divs_to_remove
                            ]
                            ensure_trajectory_statistics(data)
                            preserved_message = (
                                " Latest selected division was preserved."
                                if protected_sources
                                else ""
                            )
                            st.success(
                                f"Merged {len(divs_to_remove)} division(s)."
                                f"{preserved_message}"
                            )
                            advance_trajectory_plot_epoch()
                            rerun_with_curation()
                        elif protected_sources:
                            st.info("Latest selected division was preserved; no earlier division was removed.")
                        else:
                            st.warning("No divisions in selected range.")
                else:
                    st.info("Select points across the division first.")

            if st.button("🗑️ Reset All", help="Restore original divisions & clear exclusions", use_container_width=True):
                invalidate_ai_undo_after_manual_edit()
                if 'original_div_indices' in data:
                    data['div_indices'] = list(data['original_div_indices'])
                data['excluded_indices'] = set()
                ensure_trajectory_statistics(data)
                advance_trajectory_plot_epoch()
                rerun_with_curation()

            st.divider()
            st.subheader("AI Cleaning")
            if ai_enabled:
                st.caption(
                    "Enabled for every trajectory · model and suggestion settings "
                    "are in the left sidebar."
                )
                if st.session_state.ai_last_message:
                    st.success(st.session_state.ai_last_message)

                requested_scope = ai_scope_keys(
                    st.session_state.ai_scope,
                    keys,
                    curr_key,
                )
                scope_keys = [
                    key for key in requested_scope
                    if st.session_state.trajectory_dict[key].get("_review_status") != "reviewed"
                ]
                reviewed_skipped = len(requested_scope) - len(scope_keys)
                if reviewed_skipped:
                    st.caption(f"{reviewed_skipped} reviewed trajectory(s) protected from AI changes.")
                if not scope_keys:
                    st.info("No unreviewed trajectories in this scope. Reviewed labels are protected.")

                if st.button(
                    "✨ Run AI Preview",
                    use_container_width=True,
                    key="ai_run_preview",
                    disabled=not model_ready or not scope_keys,
                ):
                    try:
                        st.session_state.ai_model_error = None
                        model, normalizer, checkpoint_thresholds, _ = ai_bundle
                        (
                            _,
                            invalid_input_count,
                            newly_excluded_count,
                        ) = apply_ai_input_preflight(scope_keys)
                        with st.spinner(f"Predicting {len(scope_keys)} trajectory(s)..."):
                            predictions = predict_trajectory_scope(
                                model,
                                normalizer,
                                checkpoint_thresholds,
                                st.session_state.trajectory_dict,
                                scope_keys,
                                device=checkpoint_signature[-1],
                            )
                        store_ai_predictions(predictions, checkpoint_signature)
                        st.session_state.ai_last_message = (
                            f"Previewed {len(predictions)} trajectory(s); "
                            f"{invalid_input_count} invalid input point(s) are "
                            f"excluded ({newly_excluded_count} newly added)."
                        )
                        st.session_state.ai_model_error = None
                        advance_trajectory_plot_epoch()
                        rerun_with_curation()
                    except Exception as exc:
                        st.session_state.ai_model_error = f"Prediction failed: {exc}"
                        st.error(st.session_state.ai_model_error)

                preview_ready = (
                    model_ready
                    and st.session_state.ai_prediction_signature == checkpoint_signature
                    and bool(scope_keys)
                    and all(
                        key in st.session_state.ai_predictions for key in scope_keys
                    )
                )
                if preview_ready:
                    division_count, excluded_count = ai_preview_scope_counts(
                        scope_keys
                    )
                    st.caption(
                        f"Preview: {division_count} division and "
                        f"{excluded_count} pending exclusion suggestions."
                    )
                elif scope_keys:
                    st.caption("Run AI Preview for this scope before applying labels.")

                can_apply = (
                    preview_ready
                    and (
                        st.session_state.ai_apply_divisions
                        or st.session_state.ai_apply_exclusions
                    )
                )
                if st.button(
                    "✅ Apply AI Suggestions",
                    use_container_width=True,
                    key="ai_apply_suggestions",
                    disabled=not can_apply,
                ):
                    try:
                        st.session_state.ai_model_error = None
                        snapshot = apply_ai_predictions(
                            st.session_state.trajectory_dict,
                            st.session_state.ai_predictions,
                            scope_keys,
                            division_threshold=st.session_state.ai_division_threshold,
                            excluded_threshold=st.session_state.ai_excluded_threshold,
                            apply_divisions=st.session_state.ai_apply_divisions,
                            apply_exclusions=st.session_state.ai_apply_exclusions,
                            policy=st.session_state.ai_apply_policy.lower(),
                        )
                        st.session_state.ai_undo_snapshot = snapshot
                        resolution_totals = {
                            'shifted_previous': 0,
                            'coalesced_previous': 0,
                            'needs_review': 0,
                        }
                        for key in scope_keys:
                            resolution = ensure_trajectory_statistics(
                                st.session_state.trajectory_dict[key]
                            )
                            for status in resolution_totals:
                                resolution_totals[status] += resolution.summary_counts[
                                    status
                                ]
                        st.session_state.ai_last_message = (
                            f"Applied AI suggestions to {len(scope_keys)} trajectory(s). "
                            f"Resolved divisions: "
                            f"{resolution_totals['shifted_previous']} shifted, "
                            f"{resolution_totals['coalesced_previous']} coalesced, "
                            f"{resolution_totals['needs_review']} need review."
                        )
                        st.session_state.ai_model_error = None
                        advance_trajectory_plot_epoch()
                        rerun_with_curation()
                    except Exception as exc:
                        st.session_state.ai_model_error = f"Apply failed: {exc}"
                        st.error(st.session_state.ai_model_error)

                st.caption(
                    "Manual exclusion and division tools remain available after AI "
                    "Apply. The first later manual edit keeps the AI labels and clears "
                    "the one-step AI undo snapshot."
                )
                st.caption(
                    "If a division source is excluded, its raw labels are preserved. "
                    "The effective boundary moves to the previous valid point only "
                    f"when the next/previous length ratio is below "
                    f"{DIVISION_CONFLICT_DROP_RATIO:.2f}; otherwise it is flagged "
                    "for review."
                )

                if st.button(
                    "↩️ Undo AI Changes",
                    use_container_width=True,
                    key="ai_undo_changes",
                    disabled=st.session_state.ai_undo_snapshot is None,
                ):
                    snapshot = {
                        key: value for key, value in st.session_state.ai_undo_snapshot.items()
                        if st.session_state.trajectory_dict[key].get("_review_status") != "reviewed"
                    }
                    restore_cleaning_snapshot(
                        st.session_state.trajectory_dict,
                        snapshot,
                    )
                    for key in snapshot:
                        ensure_trajectory_statistics(
                            st.session_state.trajectory_dict[key]
                        )
                    st.session_state.ai_undo_snapshot = None
                    st.session_state.ai_last_message = "Restored labels from before AI apply."
                    st.session_state.ai_model_error = None
                    advance_trajectory_plot_epoch()
                    rerun_with_curation()

                if st.button(
                    "🧹 Clear AI Preview",
                    use_container_width=True,
                    key="ai_clear_preview",
                    disabled=not any(
                        key in st.session_state.ai_predictions for key in scope_keys
                    ),
                ):
                    for key in scope_keys:
                        st.session_state.ai_predictions.pop(key, None)
                    st.session_state.ai_predictions_revision = int(
                        st.session_state.get('ai_predictions_revision', 0)
                    ) + 1
                    st.session_state.ai_preview_count_cache = {}
                    if not st.session_state.ai_predictions:
                        st.session_state.ai_prediction_signature = None
                        st.session_state.ai_prediction_checkpoint = None
                    st.session_state.ai_last_message = "Cleared AI preview for this scope."
                    st.session_state.ai_model_error = None
                    advance_trajectory_plot_epoch()
                    rerun_with_curation()
            else:
                st.info(
                    "Enable AI-assisted cleaning in the left sidebar. The setting "
                    "will stay enabled while you move between trajectories."
                )

            st.divider()
            st.subheader("Export")
            if st.button(
                "💾 Export Current Trajectory",
                help=(
                    "Save the trajectory currently shown above, with its "
                    "curation labels (division_point, excluded_point, "
                    "effective_division_point, cell_cycle_index), into the "
                    "input directory as {trajectory_key}_selected_labeled.csv"
                ),
                use_container_width=True,
                key="export_current_trajectory",
            ):
                current_export_df = build_labeled_trajectory_export(
                    data, curr_key
                )
                # The '_selected_' infix keeps this file out of the Process
                # Data loader, which skips any CSV whose name contains
                # 'selected' (same convention as selected_labeled.csv).
                current_out_path = os.path.join(
                    data_path, f"{curr_key}_selected_labeled.csv"
                )
                current_export_df.to_csv(current_out_path, index=False)
                st.success(
                    f"Saved {len(current_export_df)} rows to {current_out_path}"
                )


    render_trajectory_viewer()
    # Statistics Section
    if st.session_state.selected_keys:
        st.divider()
        st.subheader("📊 Statistics (Selected Only)")
        show_selected_statistics = st.toggle(
            "Calculate selected statistics",
            value=False,
            key="show_selected_statistics",
            help=(
                "Keep this off while curating large data sets. Turn it on only "
                "when you want to build an aggregate histogram."
            ),
        )
        if show_selected_statistics:
            sel_list = []
            for selected_key in sorted(st.session_state.selected_keys):
                selected_data = st.session_state.trajectory_dict.get(
                    selected_key
                )
                if selected_data is None:
                    continue
                ensure_trajectory_statistics(selected_data)
                sel_list.append(selected_data)
            statistic_name = st.radio(
                "Statistic",
                ("Growth Rate", "Generation Time"),
                horizontal=True,
                key="selected_statistic_name",
            )
        else:
            st.caption(
                "Aggregate calculation is paused while you navigate and edit "
                "trajectories."
            )

        if show_selected_statistics and statistic_name == "Growth Rate":
            r2_lim = st.slider(
                "R² Filter",
                0.0,
                1.0,
                0.9,
                key="selected_growth_r2_filter",
            )
            grs = []
            for d in sel_list:
                m = d['growth_rate']
                if m.size > 0 and m.shape[0] >= 3: grs.extend(m[1, m[2] >= r2_lim])
            if grs:
                st.plotly_chart(
                    go.Figure(go.Histogram(x=grs, nbinsx=30, name="GR")),
                    key="selected_growth_histogram",
                )
            else:
                st.info("No growth-rate values pass the current R² filter.")
        elif show_selected_statistics:
            gens = []
            for d in sel_list:
                m = d['division_time']
                if m.size > 0: gens.extend(m[1])
            if gens:
                st.plotly_chart(
                    go.Figure(
                        go.Histogram(
                            x=gens,
                            nbinsx=30,
                            name="GenTime",
                            marker_color='salmon',
                        )
                    ),
                    key="selected_generation_histogram",
                )
            else:
                st.info("No generation-time values are available.")

    st.sidebar.divider()
    if st.sidebar.button("💾 Export"):
        if st.session_state.selected_keys:
            final = pd.concat([st.session_state.trajectory_dict[k]['trajectory'] for k in st.session_state.selected_keys])
            final.to_csv(os.path.join(data_path, "selected_plotly.csv"), index=False)
            st.sidebar.success("Saved CSV.")

    if st.sidebar.button("📊 Export with Labels"):
        if st.session_state.selected_keys:
            final_labeled = canonical_labeled_export(st.session_state.trajectory_dict, build_labeled_trajectory_export)
            out_path = os.path.join(data_path, "selected_labeled.csv")
            final_labeled.to_csv(out_path, index=False)
            st.sidebar.success(f"Saved labeled data to {out_path}")
        else:
            st.sidebar.warning("No trajectories selected.")

    # --- Keyboard Shortcuts Logic ---
    import streamlit.components.v1 as components
    
    # JavaScript to listen for key presses and click buttons
    components.html("""
        <script>
        const host = window.parent;
        const doc = window.parent.document;
        const handlerName = '__momaLineageKeyHandler';
        if (host[handlerName]) {
            doc.removeEventListener('keydown', host[handlerName]);
        }
        const keyHandler = function(e) {
            // Image review has its own controls; never edit behind a popup.
            if (doc.querySelector('[role="dialog"]')) return;
            // Let text fields and slider thumbs handle their own arrow keys.
            if (e.repeat || ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)
                || e.target.closest('[role="slider"]')) return;
            
            let btn;
            if (e.key === 'ArrowLeft') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Prev'))?.parentElement;
            } else if (e.key === 'ArrowRight') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Next'))?.parentElement;
            } else if (e.key.toLowerCase() === 'k') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Trajectory'))?.parentElement;
            } else if (e.key.toLowerCase() === 'e') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Exclude Selected'))?.parentElement;
            } else if (e.key.toLowerCase() === 'r') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Restore Selected Exclusions'))?.parentElement;
            } else if (e.key.toLowerCase() === 'd') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Mark Division'))?.parentElement;
            } else if (e.key.toLowerCase() === 'm') {
                btn = Array.from(doc.querySelectorAll('button p')).find(el => el.innerText.includes('Merge Cycles'))?.parentElement;
            }
            
            if (btn && !btn.disabled) { // Only click if the button is not disabled
                btn.click();
            }
        };
        host[handlerName] = keyHandler;
        doc.addEventListener('keydown', keyHandler);
        </script>
    """, height=0)

