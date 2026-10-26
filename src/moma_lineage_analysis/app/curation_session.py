"""Session coordinator for transactional trajectory autosave and recovery."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets

from moma_lineage_analysis.curation_store import create_dataset, fingerprint_sources, load_dataset, save_changes


def _signature(entry, selected):
    return (
        tuple(entry.get("div_indices", [])),
        frozenset(entry.get("excluded_indices", set())),
        tuple(entry.get("original_div_indices", [])),
        bool(selected),
        entry.get("_review_status", "unreviewed"),
        entry.get("_reviewed_at"),
    )


def _current_key(state):
    keys = list(state.get("trajectory_dict", {}))
    if not keys:
        return None
    return keys[int(state.get("current_idx", 0)) % len(keys)]


def attach_saved_dataset(state, payload, db_path):
    """Restore measurements and durable labels; UI/AI caches are rebuilt separately."""

    entries = payload["trajectory_dict"]
    selected = set(payload["selected_keys"])
    keys = list(entries)
    state["trajectory_dict"] = entries
    state["selected_keys"] = selected
    state["current_idx"] = keys.index(payload["current_key"]) if payload.get("current_key") in keys else 0
    state["curation_db_path"] = str(Path(db_path).resolve())
    state["curation_dataset_id"] = payload["dataset_id"]
    state["curation_revision"] = payload["revision"]
    state["curation_saved_signatures"] = {
        key: _signature(entry, key in selected) for key, entry in entries.items()
    }
    state["curation_saved_cursor"] = _current_key(state)
    state["curation_settings"] = payload["settings"]
    state["curation_sources"] = payload["sources"]
    state["curation_logic_version"] = payload["logic_version"]
    state["curation_error"] = None
    state["curation_last_saved"] = payload.get("updated_at", "")
    state.pop("curation_pending_initial", None)


def initialize_curation(state, db_path, *, sources, settings, logic_version, builder):
    """Create a complete version once, or recover its existing saved review."""

    state["curation_pending_initial"] = {
        "db_path": str(db_path), "sources": sources,
        "settings": settings, "logic_version": logic_version,
    }
    entries = state["trajectory_dict"]
    labeled = {key: builder(entry, key) for key, entry in entries.items()}
    info = create_dataset(
        db_path, trajectory_dict=entries, labeled_frames=labeled,
        selected_keys=set(state["selected_keys"]), sources=sources,
        settings=settings, logic_version=logic_version, current_key=_current_key(state),
    )
    if info.get("created"):
        # New data are already in memory; do not load a duplicate copy or risk
        # discarding a later edit after a transient post-commit read failure.
        payload = dict(info, trajectory_dict=entries, selected_keys=set(state["selected_keys"]))
    else:
        payload = load_dataset(db_path, info["dataset_id"], source_dir=Path(db_path).parent)
    attach_saved_dataset(state, payload, db_path)
    return payload



def save_current_session_to_curation(state, data_path, builder):
    """Persist a pre-autosave session as a new version, never as old review data."""

    directory = Path(data_path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Choose an existing input directory before saving this session.")
    entries = state.get("trajectory_dict", {})
    if not entries:
        raise ValueError("Load trajectories before saving this session.")
    source_paths = {str(entry["source_csv_path"]) for entry in entries.values()}
    if not all(Path(path).is_file() and Path(path).resolve().parent == directory for path in source_paths):
        raise ValueError("Every trajectory must have a source CSV in the selected input directory.")
    settings = dict(state.get("curation_current_parameters") or {})
    if not settings:
        raise ValueError("Current processing parameters are unavailable; use Process Data first.")
    settings["session_import_id"] = secrets.token_hex(4)
    return initialize_curation(
        state, directory / "moma_curation.sqlite3",
        sources=fingerprint_sources(source_paths), settings=settings,
        logic_version=state.get("curation_current_logic_version"),
        builder=builder,
    )

def save_recovery_copy(state, builder):
    """Commit all unsaved local edits into a separate DB before conflict reload."""

    original = Path(state["curation_db_path"])
    recovery_path = original.with_name(f"moma_curation_recovery_{secrets.token_hex(4)}.sqlite3")
    entries = state["trajectory_dict"]
    create_dataset(
        recovery_path, trajectory_dict=entries,
        labeled_frames={key: builder(entry, key) for key, entry in entries.items()},
        selected_keys=set(state["selected_keys"]), sources=state["curation_sources"],
        settings=state["curation_settings"], logic_version=state["curation_logic_version"],
        current_key=_current_key(state),
    )
    return recovery_path

def flush_curation(state, builder, *, current_key=None, mark_reviewed=None):
    """Save all dirty trajectories in one transaction, retaining failures in memory."""

    if state.get("curation_pending_initial"):
        context = state["curation_pending_initial"].copy()
        try:
            initialize_curation(state, builder=builder, **context)
        except Exception as exc:
            state["curation_error"] = str(exc)
            return False
    entries = state.get("trajectory_dict", {})
    selected = set(state.get("selected_keys", set()))
    baseline = state.get("curation_saved_signatures", {})
    changed = {}
    review_before = {}
    for key, entry in entries.items():
        previous = baseline.get(key)
        signature = _signature(entry, key in selected)
        # Editing labels or membership revokes the earlier review confirmation.
        if ((previous is None and state.get("curation_dataset_id"))
                or (previous is not None and signature[:4] != previous[:4])):
            entry["_review_status"] = "in_progress"
            entry["_reviewed_at"] = None
        if key == mark_reviewed and entry.get("_review_status") != "reviewed":
            review_before[key] = (entry.get("_review_status", "unreviewed"), entry.get("_reviewed_at"))
            entry["_review_status"] = "reviewed"
            entry["_reviewed_at"] = datetime.now(timezone.utc).isoformat()
        if _signature(entry, key in selected) != previous:
            changed[key] = entry
    cursor = current_key if current_key is not None else _current_key(state)
    if not state.get("curation_dataset_id"):
        # Preserve review protection in sessions opened before autosave.
        state["curation_saved_signatures"] = {
            key: _signature(entry, key in selected) for key, entry in entries.items()
        }
        state["curation_saved_cursor"] = cursor
        return True
    if not changed and cursor == state.get("curation_saved_cursor") and not state.get("curation_error"):
        return True
    try:
        labeled = {key: builder(entry, key) for key, entry in changed.items()}
        revision = save_changes(
            state["curation_db_path"], state["curation_dataset_id"],
            expected_revision=state["curation_revision"], trajectory_dict=changed,
            labeled_frames=labeled, selected_keys=selected, current_key=cursor,
        )
    except Exception as exc:
        for key, (status, reviewed_at) in review_before.items():
            entries[key]["_review_status"] = status
            entries[key]["_reviewed_at"] = reviewed_at
        state["curation_error"] = str(exc)
        return False
    state["curation_revision"] = revision
    state["curation_saved_signatures"] = {
        key: _signature(entry, key in selected) for key, entry in entries.items()
    }
    state["curation_saved_cursor"] = cursor
    state["curation_error"] = None
    state["curation_last_saved"] = datetime.now(timezone.utc).isoformat()
    return True
