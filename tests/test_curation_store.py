"""Contract tests for local curation sessions (not the frozen training store)."""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
import pytest

from moma_lineage_analysis.curation_store import (
    DB_FILENAME, CurationConflictError, CurationStoreError, create_dataset,
    find_dataset, fingerprint_sources, list_datasets, load_dataset, save_changes,
)

LABELS = ["division_point", "excluded_point", "effective_division_point"]


def session(source_dir, second=False):
    source_dir.mkdir(exist_ok=True)
    path = source_dir / ("two_cells.csv" if second else "one_cells.csv")
    path.write_bytes(b"raw csv")
    key = path.stem
    raw = pd.DataFrame({
        "fov_cell_index": pd.Series([10, 20, 30], dtype="int64"),
        "time_h": pd.Series([2.0, np.inf, np.nan], dtype="float64"),
        "cell_text": pd.Series(["third", "second", "first"], dtype="string"),
        "fluor": pd.Series([1.5, np.nan, -np.inf], dtype="float64"),
        "integer_marker": pd.Series([1, 2, 3], dtype="int64"),
    })
    raw.index = [9, 9, 9]
    if second:
        raw = raw.rename(columns={"fluor": "other_fluor"})
        raw["nullable_fluor"] = pd.Series([4, pd.NA, 6], dtype="Int64")
    labels = raw.iloc[[2, 1, 0]].reset_index(drop=True).copy()
    labels["trajectory_key"] = key
    labels["division_point"] = labels["fov_cell_index"].eq(30)
    labels["excluded_point"] = labels["fov_cell_index"].eq(20)
    labels["effective_division_point"] = labels["fov_cell_index"].eq(10)
    labels["cell_cycle_index"] = [0, 0, 1]
    entry = {"trajectory": raw, "div_indices": [30], "original_div_indices": [30],
             "excluded_indices": {20}, "source_csv_path": str(path)}
    return {key: entry}, {key: labels}, key


def make_store(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir(exist_ok=True)
    trajectories, labels, key = session(directory)
    sources = fingerprint_sources([entry["source_csv_path"] for entry in trajectories.values()])
    db = tmp_path / DB_FILENAME
    settings = {"division_mode": "effective", "nested": {"alpha": [1, 2]}}
    info = create_dataset(db, trajectory_dict=trajectories, labeled_frames=labels,
                          selected_keys={key}, sources=sources, settings=settings,
                          logic_version="2.3", current_key=key)
    return db, trajectories, labels, key, sources, settings, info


def read_view(db, view):
    with sqlite3.connect(db) as connection:
        columns = [r[1] for r in connection.execute(f'PRAGMA table_info("{view}")')]
        rows = connection.execute(f'SELECT * FROM "{view}"').fetchall()
    return pd.DataFrame(rows, columns=columns)


def test_roundtrip_recovery_dtypes_cursor_baseline_and_relocation(tmp_path):
    db, original, _, key, sources, settings, info = make_store(tmp_path)
    assert db.name.endswith(".sqlite3")
    recovered = load_dataset(db, info["dataset_id"], source_dir=tmp_path / "relocated")
    entry = recovered["trajectory_dict"][key]
    pd.testing.assert_frame_equal(entry["trajectory"], original[key]["trajectory"].reset_index(drop=True))
    assert entry["div_indices"] == [30]
    assert entry["original_div_indices"] == [30]
    assert entry["excluded_indices"] == {20}
    assert entry["_review_status"] == "unreviewed"
    assert entry["_reviewed_at"] is None
    assert entry["_saved_revision"] == recovered["revision"] == 0
    assert entry["source_csv_path"] == str(tmp_path / "relocated" / "one_cells.csv")
    assert recovered["current_key"] == key
    assert recovered["selected_keys"] == {key}
    assert recovered["settings"] == settings
    assert recovered["logic_version"] == "2.3"
    assert recovered["sources"] == sources


def test_views_match_supplied_labels_and_filter_reviewed_and_dropped(tmp_path):
    db, _, labels, key, _, _, info = make_store(tmp_path)
    view = read_view(db, "selected_labeled")
    expected = labels[key].copy()
    expected[LABELS] = expected[LABELS].astype("int64")
    pd.testing.assert_frame_equal(view, expected, check_dtype=False)
    assert list(view.columns) == list(expected.columns)
    assert read_view(db, "reviewed_selected_labeled").empty
    entry = load_dataset(db, info["dataset_id"])["trajectory_dict"][key]
    entry["_review_status"] = "reviewed"
    entry["_reviewed_at"] = "2026-10-27T00:00:00+00:00"
    assert save_changes(db, info["dataset_id"], expected_revision=0,
                        trajectory_dict={key: entry}, labeled_frames=labels,
                        selected_keys={key}, current_key=key) == 1
    assert len(read_view(db, "reviewed_selected_labeled")) == 3
    assert save_changes(db, info["dataset_id"], expected_revision=1,
                        trajectory_dict={}, labeled_frames={}, selected_keys=set(),
                        current_key=None) == 2
    assert read_view(db, "selected_labeled").empty
    assert read_view(db, "reviewed_selected_labeled").empty
    pd.testing.assert_frame_equal(read_view(db, "all_labeled"), expected, check_dtype=False)
    assert list_datasets(db)[0]["selected_count"] == 0
    assert list_datasets(db)[0]["reviewed_count"] == 1


def test_create_is_idempotent_and_does_not_reset_labels_or_cursor(tmp_path):
    db, original, original_labels, key, sources, settings, info = make_store(tmp_path)
    edited = original_labels[key].copy()
    edited.loc[edited["fov_cell_index"].eq(30), "division_point"] = False
    entry = load_dataset(db, info["dataset_id"])["trajectory_dict"][key]
    entry["div_indices"] = []
    entry["_review_status"] = "reviewed"
    save_changes(db, info["dataset_id"], expected_revision=0,
                 trajectory_dict={key: entry}, labeled_frames={key: edited},
                 selected_keys=set(), current_key=None)
    again = create_dataset(db, trajectory_dict=original, labeled_frames=original_labels,
                           selected_keys={key}, sources=sources, settings=settings,
                           logic_version="2.3", current_key=key)
    assert again["dataset_id"] == info["dataset_id"]
    assert again["revision"] == 1
    recovered = load_dataset(db, info["dataset_id"])
    assert recovered["current_key"] is None
    assert recovered["selected_keys"] == set()
    assert recovered["trajectory_dict"][key]["div_indices"] == []
    assert recovered["trajectory_dict"][key]["_review_status"] == "reviewed"


@pytest.mark.parametrize("change", ["settings", "logic_version", "sources"])
def test_new_identity_keeps_old_datasets_and_switches_views(tmp_path, change):
    db, trajectories, labels, key, sources, settings, old = make_store(tmp_path)
    logic_version = "2.3"
    if change == "settings":
        settings = {**settings, "new": True}
    elif change == "logic_version":
        logic_version = "2.4"
    else:
        sources = [dict(sources[0], sha256="0" * 64)]
    new = create_dataset(db, trajectory_dict=trajectories, labeled_frames=labels,
                         selected_keys={key}, sources=sources, settings=settings,
                         logic_version=logic_version)
    assert new["dataset_id"] != old["dataset_id"]
    assert len(list_datasets(db)) == 2
    assert list_datasets(db)[0]["dataset_id"] == new["dataset_id"]
    assert find_dataset(db, sources=sources, settings=settings,
                        logic_version=logic_version)["dataset_id"] == new["dataset_id"]
    load_dataset(db, old["dataset_id"])
    assert read_view(db, "selected_labeled")["cell_cycle_index"].tolist() == [0, 0, 1]


def test_find_and_list_do_not_create_missing_database(tmp_path):
    db = tmp_path / "not-created.sqlite3"
    assert list_datasets(db) == []
    assert find_dataset(db, sources=[], settings={}, logic_version="2.3") is None
    assert not db.exists()


def test_revision_cas_and_batch_rollback_leave_no_partial_edits(tmp_path):
    db, _, labels, key, _, _, info = make_store(tmp_path)
    entry = load_dataset(db, info["dataset_id"])["trajectory_dict"][key]
    entry["_review_status"] = "reviewed"
    edited = labels[key].copy()
    edited.loc[edited["fov_cell_index"].eq(30), "division_point"] = False
    entry["div_indices"] = []
    with pytest.raises(CurationConflictError):
        save_changes(db, info["dataset_id"], expected_revision=4,
                     trajectory_dict={key: entry}, labeled_frames={key: edited},
                     selected_keys=set(), current_key=key)
    unchanged = load_dataset(db, info["dataset_id"])
    assert unchanged["revision"] == 0
    assert unchanged["selected_keys"] == {key}
    assert unchanged["trajectory_dict"][key]["_review_status"] == "unreviewed"
    assert read_view(db, "selected_labeled")["division_point"].tolist() == [1, 0, 0]
    entry = unchanged["trajectory_dict"][key]
    entry["_review_status"] = "reviewed"
    entry["trajectory"].loc[0, "fluor"] = 100.0
    with pytest.raises(CurationStoreError):
        save_changes(db, info["dataset_id"], expected_revision=0,
                     trajectory_dict={key: entry}, labeled_frames=labels,
                     selected_keys=set(), current_key=key)
    assert load_dataset(db, info["dataset_id"])["revision"] == 0


def test_duplicate_ids_are_rejected_atomically(tmp_path):
    db, trajectories, labels, key, sources, settings, _ = make_store(tmp_path)
    db = tmp_path / "other.sqlite3"
    trajectories[key]["trajectory"].iloc[1, 0] = 10
    labels[key].iloc[1, 0] = 10
    with pytest.raises(CurationStoreError, match="Duplicate"):
        create_dataset(db, trajectory_dict=trajectories, labeled_frames=labels,
                       selected_keys={key}, sources=sources, settings=settings,
                       logic_version="2.3")
    assert not db.exists()


def test_two_trajectory_schema_union_and_mixed_nullable_types(tmp_path):
    first, first_labels, first_key = session(tmp_path / "input")
    second, second_labels, second_key = session(tmp_path / "input", second=True)
    trajectories = {**first, **second}
    labels = {**first_labels, **second_labels}
    sources = fingerprint_sources([entry["source_csv_path"] for entry in trajectories.values()])
    db = tmp_path / DB_FILENAME
    info = create_dataset(db, trajectory_dict=trajectories, labeled_frames=labels,
                          selected_keys={first_key, second_key}, sources=sources,
                          settings={}, logic_version="2.3")
    actual = read_view(db, "selected_labeled")
    expected = pd.concat([first_labels[first_key], second_labels[second_key]], ignore_index=True)
    expected[LABELS] = expected[LABELS].astype("int64")
    raw_union = list(dict.fromkeys(column for frame in (first[first_key]["trajectory"], second[second_key]["trajectory"]) for column in frame.columns))
    assert list(actual.columns) == raw_union + ["trajectory_key"] + LABELS + ["cell_cycle_index"]
    pd.testing.assert_frame_equal(actual, expected[actual.columns], check_dtype=False)
    loaded = load_dataset(db, info["dataset_id"])
    assert str(loaded["trajectory_dict"][second_key]["trajectory"]["nullable_fluor"].dtype) == "Int64"
    assert pd.isna(loaded["trajectory_dict"][second_key]["trajectory"].loc[1, "nullable_fluor"])


def test_fingerprints_are_stable_and_reject_duplicates_and_missing(tmp_path):
    (tmp_path / "b").write_bytes(b"B")
    (tmp_path / "a").write_bytes(b"A")
    other = tmp_path / "other"
    other.mkdir()
    (other / "a").write_bytes(b"different")
    fingerprints = fingerprint_sources([tmp_path / "b", tmp_path / "a"])
    assert [s["filename"] for s in fingerprints] == ["a", "b"]
    assert fingerprints[0]["sha256"] != fingerprint_sources([other / "a"])[0]["sha256"]
    with pytest.raises(CurationStoreError, match="Duplicate"):
        fingerprint_sources([tmp_path / "a", other / "a"])
    with pytest.raises(CurationStoreError):
        fingerprint_sources([tmp_path / "missing"])




def test_failed_initial_import_leaves_retryable_empty_schema(tmp_path, monkeypatch):
    import moma_lineage_analysis.curation_store as store

    trajectories, labels, key = session(tmp_path / "input")
    sources = fingerprint_sources([trajectories[key]["source_csv_path"]])
    db = tmp_path / DB_FILENAME
    arguments = dict(trajectory_dict=trajectories, labeled_frames=labels,
                     selected_keys={key}, sources=sources, settings={},
                     logic_version="2.3", current_key=key)
    def fail_insert(*args):
        raise sqlite3.IntegrityError("injected annotation failure")
    with monkeypatch.context() as patch:
        patch.setattr(store, "_insert_annotations", fail_insert)
        with pytest.raises(CurationStoreError):
            create_dataset(db, **arguments)
    assert list_datasets(db) == []
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
    created = create_dataset(db, **arguments)
    assert created["trajectory_count"] == 1
    assert load_dataset(db, created["dataset_id"])["current_key"] == key

def test_in_progress_review_and_none_settings_survive_reopen(tmp_path):
    db, _, labels, key, sources, _, info = make_store(tmp_path)
    entry = load_dataset(db, info["dataset_id"])["trajectory_dict"][key]
    entry["_review_status"] = "in_progress"
    save_changes(db, info["dataset_id"], expected_revision=0,
                 trajectory_dict={key: entry}, labeled_frames=labels,
                 selected_keys={key}, current_key=key)
    restored = load_dataset(db, info["dataset_id"])
    assert restored["trajectory_dict"][key]["_review_status"] == "in_progress"
    assert read_view(db, "reviewed_selected_labeled").empty
    other = tmp_path / "optional.sqlite3"
    trajectories, frames, _ = session(tmp_path / "input")
    created = create_dataset(other, trajectory_dict=trajectories, labeled_frames=frames,
                             selected_keys={key}, sources=sources, settings=None,
                             logic_version="2.3")
    assert created["settings"] == {}
    assert created["sources"] == sources
    assert find_dataset(other, sources=sources, settings=None,
                        logic_version="2.3")["dataset_id"] == created["dataset_id"]

def test_unknown_schema_is_not_replaced(tmp_path):
    db = tmp_path / DB_FILENAME
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE user_data (value TEXT)")
        connection.execute("INSERT INTO user_data VALUES ('keep')")
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(CurationStoreError):
        list_datasets(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT * FROM user_data").fetchone()[0] == "keep"
