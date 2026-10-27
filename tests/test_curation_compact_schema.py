"""Compact curation schema and reviewed-view/export contract."""

from contextlib import closing
import sqlite3

import pandas as pd

from moma_lineage_analysis.curation_store import create_dataset, fingerprint_sources, load_dataset, save_changes
from test_curation_store import LABELS, session


def _read_view(db, view):
    with closing(sqlite3.connect(db)) as connection:
        return pd.read_sql_query(f"SELECT * FROM {view}", connection)


def _expected_columns(trajectories):
    measurement = list(dict.fromkeys(
        column for entry in trajectories.values() for column in entry["trajectory"].columns
    ))
    # The five labeled-export fields are trajectory_key, three point labels and cycle.
    return measurement + ["trajectory_key"] + LABELS + ["cell_cycle_index"]


def test_compact_database_has_no_annotation_table_and_matching_views(tmp_path):
    trajectories, labels, key = session(tmp_path / "input")
    db = tmp_path / "compact.sqlite3"
    create_dataset(
        db, trajectory_dict=trajectories, labeled_frames=labels,
        selected_keys={key}, sources=fingerprint_sources([trajectories[key]["source_csv_path"]]),
        settings={}, logic_version="2.3", current_key=key,
    )
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        assert not connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='annotations'").fetchall()
    pd.testing.assert_frame_equal(
        _read_view(db, "selected_labeled"), labels[key], check_dtype=False,
    )


def test_reviewed_selected_view_matches_labeled_export_schema_and_rows(tmp_path):
    first, first_labels, first_key = session(tmp_path / "input")
    second, second_labels, second_key = session(tmp_path / "input", second=True)
    trajectories = {**first, **second}
    labeled = {**first_labels, **second_labels}
    sources = fingerprint_sources([entry["source_csv_path"] for entry in trajectories.values()])
    db = tmp_path / "compact.sqlite3"
    info = create_dataset(
        db, trajectory_dict=trajectories, labeled_frames=labeled,
        selected_keys={first_key, second_key}, sources=sources,
        settings={}, logic_version="2.3", current_key=first_key,
    )
    first_entry = first[first_key]
    first_entry["_review_status"] = "reviewed"
    save_changes(
        db, info["dataset_id"], expected_revision=0,
        trajectory_dict={first_key:first_entry}, labeled_frames={first_key:first_labels[first_key]},
        selected_keys={first_key, second_key}, current_key=first_key,
    )
    columns = _expected_columns(trajectories)
    expected_selected = pd.concat(
        [first_labels[first_key], second_labels[second_key]], ignore_index=True,
    ).reindex(columns=columns)
    pd.testing.assert_frame_equal(
        _read_view(db, "selected_labeled"), expected_selected, check_dtype=False,
    )
    expected_reviewed = first_labels[first_key].reindex(columns=columns)
    pd.testing.assert_frame_equal(
        _read_view(db, "reviewed_selected_labeled"), expected_reviewed, check_dtype=False,
    )


def test_internal_label_names_are_safely_aliased_for_raw_measurements(tmp_path):
    trajectories, labels, key = session(tmp_path / "input")
    renames = {"fluor":"_moma_division_state", "integer_marker":"_MOMA_LABEL_ORDER"}
    trajectories[key]["trajectory"] = trajectories[key]["trajectory"].rename(columns=renames)
    labels[key] = labels[key].rename(columns=renames)
    db = tmp_path / "compact.sqlite3"
    info = create_dataset(
        db, trajectory_dict=trajectories, labeled_frames=labels,
        selected_keys={key}, sources=fingerprint_sources([trajectories[key]["source_csv_path"]]),
        settings={}, logic_version="2.3", current_key=key,
    )
    recovered = load_dataset(db, info["dataset_id"])
    pd.testing.assert_frame_equal(
        recovered["trajectory_dict"][key]["trajectory"], trajectories[key]["trajectory"].reset_index(drop=True),
    )
    exported = _read_view(db, "selected_labeled")
    assert exported["_moma_division_state"].iloc[[0,2]].tolist() == [-float("inf"), 1.5]
    assert exported["_moma_division_state"].isna().sum() == 1
    assert exported["division_point"].tolist() == [1,0,0]


def test_trajectory_metadata_uses_shared_source_and_schema_references(tmp_path):
    trajectories, labels, key = session(tmp_path / "input")
    second_key = key + "_second_channel"
    trajectories[second_key] = dict(trajectories[key], trajectory=trajectories[key]["trajectory"].copy())
    labels[second_key] = labels[key].assign(trajectory_key=second_key)
    db = tmp_path / "shared.sqlite3"
    arguments = dict(trajectory_dict=trajectories, labeled_frames=labels,
                     selected_keys=set(trajectories), sources=fingerprint_sources([trajectories[key]["source_csv_path"]]),
                     settings={}, logic_version="2.3", current_key=key)
    create_dataset(db, **arguments)
    with closing(sqlite3.connect(db)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(trajectories)")}
        assert not columns & {"source_filename", "source_path", "original_div_indices_json", "dtype_json"}
        assert {"source_id", "schema_id"} <= columns
        assert connection.execute("SELECT count(*) FROM source_files").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM measurement_schemas").fetchone()[0] == 1
    create_dataset(db, **dict(arguments, settings={"new_version":True}))
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute("SELECT count(*) FROM measurement_schemas").fetchone()[0] == 1
    loaded = load_dataset(db, 1, source_dir=tmp_path / "moved")
    for entry in loaded["trajectory_dict"].values():
        assert entry["source_csv_path"] == str(tmp_path / "moved" / "one_cells.csv")
        pd.testing.assert_frame_equal(entry["trajectory"], trajectories[key]["trajectory"].reset_index(drop=True))


def test_reset_baseline_order_survives_current_label_edits_and_reopen(tmp_path):
    from moma_lineage_analysis.curation_store import CurationStoreError
    import pytest

    trajectories, labels, key = session(tmp_path / "input")
    trajectories[key]["original_div_indices"] = [30,10]
    db = tmp_path / "baseline.sqlite3"
    create_dataset(db, trajectory_dict=trajectories, labeled_frames=labels, selected_keys={key},
                   sources=fingerprint_sources([trajectories[key]["source_csv_path"]]), settings={}, logic_version="2.3")
    entry = load_dataset(db, 1)["trajectory_dict"][key]
    assert entry["original_div_indices"] == [30,10]
    entry["div_indices"] = []
    modified = labels[key].assign(division_point=False)
    save_changes(db, 1, expected_revision=0, trajectory_dict={key:entry}, labeled_frames={key:modified},
                 selected_keys={key}, current_key=key)
    entry = load_dataset(db, 1)["trajectory_dict"][key]
    assert entry["div_indices"] == []
    assert entry["original_div_indices"] == [30,10]
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute("SELECT _moma_division_state FROM observations ORDER BY _moma_row_order").fetchall() == [(4,),(0,),(2,)]
        assert connection.execute("SELECT division_point FROM selected_labeled").fetchall() == [(0,),(0,),(0,)]
    entry["original_div_indices"] = [10,30]
    with pytest.raises(CurationStoreError, match="immutable"):
        save_changes(db, 1, expected_revision=1, trajectory_dict={key:entry}, labeled_frames={key:modified},
                     selected_keys={key}, current_key=key)
    assert load_dataset(db, 1)["revision"] == 1


def test_integral_float_baseline_ids_can_save_against_integer_measurements(tmp_path):
    trajectories, labels, key = session(tmp_path / "input")
    trajectories[key]["original_div_indices"] = [30.0]
    db = tmp_path / "numeric_ids.sqlite3"
    create_dataset(db, trajectory_dict=trajectories, labeled_frames=labels, selected_keys={key},
                   sources=fingerprint_sources([trajectories[key]["source_csv_path"]]), settings={}, logic_version="2.3")
    save_changes(db, 1, expected_revision=0, trajectory_dict=trajectories, labeled_frames=labels,
                 selected_keys={key}, current_key=key)
    assert load_dataset(db, 1)["trajectory_dict"][key]["original_div_indices"] == [30]
