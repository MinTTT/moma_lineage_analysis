"""Scoped tests for the training-set review backend (store.py).

All tests build temporary SQLite stores from synthetic CSVs; the live database
is never touched.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from moma_lineage_analysis.trajectory_cleaning import dataset_store
from moma_lineage_analysis.trainset_review import store


def _row(
    key: str,
    fov: int,
    time_h: float,
    spine: float,
    division: bool = False,
    excluded: bool = False,
    source: str = "synthetic",
    source_file: str = "synthetic.csv",
) -> dict:
    return {
        "trajectory_key": key,
        "fov_cell_index": fov,
        "time_h": time_h,
        "spine_length": spine,
        "division_point": division,
        "excluded_point": excluded,
        "source_dataset": source,
        "source_file": source_file,
        "source_file_sha256": "0" * 64,
        "source_trajectory_key": key,
        "source_fov_cell_index": fov,
    }


def _write_csv(root: Path, name: str, rows: list[dict]) -> Path:
    path = root / "corpora" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _import(conn: sqlite3.Connection, root: Path, csv_path: Path, dataset_id: str) -> dict:
    return dataset_store.import_corpus(
        conn,
        repo_root=root,
        csv_path=csv_path,
        dataset_id=dataset_id,
        kind="batch",
        label=f"test corpus {dataset_id}",
        round_name="R900",
        verification_basis="synthetic_test",
        human_verified=True,
    )


def _build_store(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    rows_a = [
        _row("t1", 1, 0.0, 6.0),
        _row("t1", 2, 0.5, 5.8, division=True),
        _row("t1", 3, 1.0, 5.6),
        _row("t1", 4, 1.5, 5.3),
        _row("t2", 1, 0.0, 7.0),
        _row("t2", 2, 0.5, 6.8, excluded=True),
        _row("t2", 3, 1.0, 6.6),
        _row("t_long", 1, 0.0, 8.0, excluded=True),
        _row("t_long", 2, 0.5, 7.9, excluded=True),
        _row("t_long", 3, 1.0, 7.8, excluded=True),
        _row("t_long", 4, 1.5, 7.7, excluded=True),
        _row("t_long", 5, 2.0, 7.6, excluded=True),
        _row("t_long", 6, 2.5, 7.5, excluded=True),
    ]
    rows_b = [
        _row("t1", 1, 0.0, 6.0),
        _row("t1", 2, 0.5, 5.8, division=True),
        _row("t1", 3, 1.0, 5.6),
        _row("t1", 4, 1.5, 5.3),
        _row("t3", 1, 0.0, 4.0),
        _row("t3", 2, 0.5, 3.9),
    ]
    csv_a = _write_csv(root, "corpus_a.csv", rows_a)
    csv_b = _write_csv(root, "corpus_b.csv", rows_b)
    db_path = tmp_path / "store.sqlite3"
    conn = dataset_store.connect(db_path)
    try:
        dataset_store.initialize(conn)
        result_a = _import(conn, root, csv_a, "corpus_a")
        result_b = _import(conn, root, csv_b, "corpus_b")
    finally:
        conn.close()
    return {
        "db": db_path,
        "root": root,
        "lineage_a": result_a["lineage_id"],
        "lineage_b": result_b["lineage_id"],
        "version_a": result_a["version"],
        "version_b": result_b["version"],
    }


def _table_names(db: Path) -> list[str]:
    conn = dataset_store.connect_readonly(db)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]
    finally:
        conn.close()


def _schema_version(db: Path) -> int:
    conn = dataset_store.connect_readonly(db)
    try:
        return int(conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0])
    finally:
        conn.close()


def _labels_for(
    frame: pd.DataFrame,
    key: str,
    division: list[int] | None = None,
    excluded: list[int] | None = None,
) -> pd.DataFrame:
    subset = frame[frame["trajectory_key"] == key].copy()
    labels = subset[["fov_cell_index", "time_h", "spine_length"]].copy()
    labels["division_point"] = False
    labels["excluded_point"] = False
    if division:
        labels.loc[labels["fov_cell_index"].isin(division), "division_point"] = True
    if excluded:
        labels.loc[labels["fov_cell_index"].isin(excluded), "excluded_point"] = True
    return labels.reset_index(drop=True)


def _state_for_lineage(db: Path, lineage_id: str) -> dict:
    dataset_id, version, _ = dataset_store.parse_lineage_id(lineage_id)
    conn = dataset_store.connect_readonly(db)
    try:
        row = conn.execute(
            """SELECT s.* FROM review_version_state s
               JOIN dataset_version v ON v.version_id = s.version_id
               WHERE v.dataset_id = ? AND v.version = ?""",
            (dataset_id, version),
        ).fetchone()
        return dict(row) if row is not None else {}
    finally:
        conn.close()


def test_read_apis_do_not_mutate_store(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    assert "review_state" not in _table_names(db)
    assert store.revision(db) == 0
    versions = store.list_versions(db)
    assert {item["lineage_id"] for item in versions} == {
        fixture["lineage_a"],
        fixture["lineage_b"],
    }
    counts = {item["lineage_id"]: item["counts"] for item in versions}
    assert counts[fixture["lineage_a"]]["rows"] == 13
    assert counts[fixture["lineage_a"]]["trajectories"] == 3
    assert counts[fixture["lineage_b"]]["rows"] == 6
    assert counts[fixture["lineage_b"]]["trajectories"] == 2
    frame = store.load_frame(db, fixture["lineage_a"])
    assert list(frame.columns) == list(dataset_store.OUTPUT_COLUMNS)
    assert len(frame) == 13
    trajectory = store.load_trajectory(db, "t1")
    assert list(trajectory.columns) == list(dataset_store.OUTPUT_COLUMNS)
    assert len(trajectory) == 4
    trajectories = store.list_trajectories(db, fixture["lineage_a"])
    assert set(trajectories.columns) == {
        "trajectory_key",
        "source_dataset",
        "rows",
        "division_points",
        "excluded_points",
        "reviewed",
        "revision",
        "dirty",
    }
    assert set(trajectories["trajectory_key"]) == {"t1", "t2", "t_long"}
    status = store.review_status(db, "t1")
    assert status["exists"] is True
    assert status["reviewed"] is False
    assert status["revision"] == 0
    assert store.read_history(db) == []
    impact = store.impact(db, "t1")
    assert impact["counts"]["versions"] == 2
    assert impact["counts"]["trajectory_rows"] == 4
    snapshot = store.read_snapshot(db, fixture["lineage_a"], "t1")
    assert snapshot["revision"] == 0
    assert snapshot["trajectory_key"] == "t1"
    assert len(snapshot["frame"]) == 4
    assert snapshot["lineage"]["frozen"] is True
    assert "review_state" not in _table_names(db)
    assert _schema_version(db) == dataset_store.SCHEMA_VERSION
    assert store.revision(db) == 0


def test_save_labels_dirty_marks_all_versions_and_snapshots(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    original_versions = {item["lineage_id"]: item for item in store.list_versions(db)}
    original_a = original_versions[fixture["lineage_a"]]
    conn = dataset_store.connect_readonly(db)
    try:
        original_canonical = str(
            conn.execute(
                "SELECT canonical_hash FROM dataset_version"
                " WHERE dataset_id = 'corpus_a'"
            ).fetchone()[0]
        )
    finally:
        conn.close()
    labels = _labels_for(
        store.load_frame(db, fixture["lineage_a"]), "t1", division=[1], excluded=[4]
    )
    result = store.save_labels(db, "t1", labels, 0)
    assert result["revision"] == 1
    assert result["rows"] == 4
    assert {version["lineage_id"] for version in result["versions"]} == {
        fixture["lineage_a"],
        fixture["lineage_b"],
    }
    snapshot_path = Path(result["snapshot"]["path"])
    assert snapshot_path.is_file()
    assert (
        hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        == result["snapshot"]["sha256"]
    )
    versions = {item["lineage_id"]: item for item in store.list_versions(db)}
    for lineage_id in (fixture["lineage_a"], fixture["lineage_b"]):
        assert versions[lineage_id]["frozen"] is False
        assert versions[lineage_id]["dirty"] is True
        assert (
            versions[lineage_id]["counts"]["rows"]
            == original_versions[lineage_id]["counts"]["rows"]
        )
    assert versions[fixture["lineage_a"]]["counts"]["division_points"] == 1
    assert versions[fixture["lineage_a"]]["counts"]["excluded_points"] == 8
    state = _state_for_lineage(db, fixture["lineage_a"])
    assert state["original_frozen"] == 1
    assert state["original_canonical_hash"] == original_canonical
    assert state["original_rows"] == original_a["counts"]["rows"]
    assert state["original_legacy_csv_path"] is not None
    assert state["original_legacy_csv_sha256"] is not None
    assert state["working_hash"] != original_canonical
    conn = dataset_store.connect_readonly(db)
    try:
        with pytest.raises(ValueError):
            dataset_store.load_training_frame(
                conn, [fixture["lineage_a"]], require_human_verified=True
            )
    finally:
        conn.close()
    snap = sqlite3.connect(f"file:{snapshot_path.as_posix()}?mode=ro", uri=True)
    try:
        old_division = snap.execute(
            "SELECT COUNT(*) FROM observation WHERE division_point = 1"
            " AND trajectory_id = (SELECT trajectory_id FROM trajectory"
            " WHERE trajectory_key = 't1')"
        ).fetchone()[0]
        snap_integrity = snap.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        snap.close()
    assert snap_integrity == "ok"
    assert old_division == 1
    current = store.load_trajectory(db, "t1")
    assert int(current["division_point"].sum()) == 1
    assert int(current["excluded_point"].sum()) == 1


def test_save_labels_validation_and_cas_rollback(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    labels = _labels_for(store.load_frame(db, fixture["lineage_a"]), "t1", division=[1])
    store.save_labels(db, "t1", labels, 0)
    before = store.load_trajectory(db, "t1")
    revision_before = store.revision(db)
    with pytest.raises(store.RevisionConflict):
        store.save_labels(db, "t1", labels, 0)
    assert store.revision(db) == revision_before
    assert store.load_trajectory(db, "t1").equals(before)
    bad = labels.copy()
    bad["division_point"] = bad["division_point"].astype(object)
    bad.loc[0, "division_point"] = "maybe"
    with pytest.raises(store.InvalidLabels):
        store.save_labels(db, "t1", bad, revision_before)
    assert store.revision(db) == revision_before
    assert store.load_trajectory(db, "t1").equals(before)
    changed = labels.copy()
    changed.loc[0, "time_h"] = float(changed.loc[0, "time_h"]) + 1.0
    with pytest.raises(store.InvalidLabels):
        store.save_labels(db, "t1", changed, revision_before)
    assert store.revision(db) == revision_before
    partial = labels.iloc[:-1].copy()
    with pytest.raises(store.InvalidLabels):
        store.save_labels(db, "t1", partial, revision_before)
    assert store.revision(db) == revision_before


def test_undo_labels_restores_labels_only(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    original = store.load_trajectory(db, "t1")
    labels = _labels_for(
        store.load_frame(db, fixture["lineage_a"]), "t1",
        division=[1, 4], excluded=[2, 3],
    )
    store.save_labels(db, "t1", labels, 0)
    assert store.impact(db, "t1")["can_undo"] is True
    undo = store.undo_labels(db, "t1", 1)
    assert undo["revision"] == 2
    assert store.load_trajectory(db, "t1").equals(original)
    assert store.impact(db, "t1")["can_undo"] is False
    with pytest.raises(store.ReviewError):
        store.undo_labels(db, "t1", 2)


def test_delete_trajectory_across_versions(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    impact = store.impact(db, "t1")
    assert impact["counts"]["versions"] == 2
    assert impact["counts"]["trajectory_rows"] == 4
    before_revision = store.revision(db)
    with pytest.raises(store.ConfirmationError):
        store.delete_trajectory(db, "t1", before_revision, "wrong")
    assert store.revision(db) == before_revision
    assert len(store.load_trajectory(db, "t1")) == 4
    result = store.delete_trajectory(db, "t1", before_revision, "t1")
    assert result["revision"] == before_revision + 1
    assert result["deleted"]["observations"] == 4
    assert result["deleted"]["memberships"] == 2
    assert Path(result["snapshot"]["path"]).is_file()
    with pytest.raises(store.ReviewError):
        store.load_trajectory(db, "t1")
    assert store.review_status(db, "t1")["exists"] is False
    with pytest.raises(store.ReviewError):
        store.undo_labels(db, "t1", result["revision"])
    conn = dataset_store.connect_readonly(db)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM dataset_member m JOIN trajectory t"
            " ON t.trajectory_id = m.trajectory_id WHERE t.trajectory_key = 't1'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM review_payload WHERE trajectory_key = 't1'"
        ).fetchone()[0] == 0
        events = conn.execute(
            "SELECT COUNT(*) FROM review_event WHERE trajectory_key = 't1'"
        ).fetchone()[0]
        assert events >= 1
        # audit metadata is retained; only label/undo payload rows are purged
        assert conn.execute(
            "SELECT COUNT(*) FROM review_payload WHERE trajectory_key = 't1'"
        ).fetchone()[0] == 0
        for dataset_id in ("corpus_a", "corpus_b"):
            version_id = int(
                conn.execute(
                    "SELECT version_id FROM dataset_version WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchone()[0]
            )
            ordinals = [
                int(row[0])
                for row in conn.execute(
                    "SELECT ordinal FROM dataset_member WHERE version_id = ?"
                    " ORDER BY ordinal",
                    (version_id,),
                ).fetchall()
            ]
            assert ordinals == list(range(1, len(ordinals) + 1))
        remaining = conn.execute(
            "SELECT COUNT(*) FROM trajectory WHERE trajectory_key IN ('t2', 't_long', 't3')"
        ).fetchone()[0]
        assert remaining == 3
        with pytest.raises(ValueError):
            dataset_store.load_training_frame(
                conn, [fixture["lineage_a"]], require_human_verified=True
            )
    finally:
        conn.close()
    versions = {item["lineage_id"]: item for item in store.list_versions(db)}
    assert versions[fixture["lineage_a"]]["counts"]["rows"] == 9
    assert versions[fixture["lineage_a"]]["counts"]["trajectories"] == 2


def test_finalize_creates_review_native_version(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    labels = _labels_for(
        store.load_frame(db, fixture["lineage_a"]), "t1", division=[1], excluded=[4]
    )
    save = store.save_labels(db, "t1", labels, 0)
    result = store.finalize(db, fixture["lineage_a"], "reviewed_a", save["revision"])
    assert result["lineage_id"] == "reviewed_a@v1"
    assert result["dropped_keys"] == ["t_long"]
    assert result["frozen"] is True
    assert result["human_verified"] is True
    assert result["legacy_csv_path"] is None
    assert isinstance(result["legacy_csv_sha256"], str)
    assert result["trajectories"] == 2
    conn = dataset_store.connect_readonly(db)
    try:
        frame, block = dataset_store.load_training_frame(
            conn, [result["lineage_id"]], require_human_verified=True
        )
        assert len(frame) == len(store.load_frame(db, fixture["lineage_a"])) - 6
        assert "t_long" not in set(frame["trajectory_key"])
        member = block["lineages"][0]
        assert member["frozen"] is True
        assert member["human_verified"] is True
        with pytest.raises(ValueError):
            dataset_store.load_training_frame(
                conn, [fixture["lineage_a"]], require_human_verified=True
            )
    finally:
        conn.close()
    with tempfile.TemporaryDirectory() as temp_dir:
        export_conn = dataset_store.connect_readonly(db)
        try:
            export = dataset_store.export_version_csv(
                export_conn,
                result["version_id"],
                Path(temp_dir) / "export.csv",
            )
        finally:
            export_conn.close()
    assert export["sha256"] == result["legacy_csv_sha256"]
    listed = {item["lineage_id"]: item for item in store.list_versions(db)}
    assert listed[result["lineage_id"]]["frozen"] is True
    assert listed[result["lineage_id"]]["dirty"] is False
    with pytest.raises(store.InvalidDataset):
        store.finalize(db, fixture["lineage_a"], "reviewed_a", store.revision(db))
    with pytest.raises(store.InvalidDataset):
        store.finalize(db, fixture["lineage_a"], "bad id;drop", store.revision(db))
    all_result = store.finalize(
        db,
        fixture["lineage_a"],
        "reviewed_a_all",
        store.revision(db),
        apply_r004=False,
    )
    assert all_result["dropped_keys"] == []
    assert all_result["trajectories"] == 3


def test_finalize_rejects_empty_version(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    csv_path = _write_csv(
        root,
        "corpus_c.csv",
        [_row("solo", 1, 0.0, 5.0), _row("solo", 2, 0.5, 4.9)],
    )
    db = tmp_path / "store.sqlite3"
    conn = dataset_store.connect(db)
    try:
        dataset_store.initialize(conn)
        imported = _import(conn, root, csv_path, "corpus_c")
    finally:
        conn.close()
    revision_number = store.revision(db)
    store.delete_trajectory(db, "solo", revision_number, "solo")
    assert store.revision(db) == revision_number + 1
    empty_frame = store.load_frame(db, imported["lineage_id"])
    assert len(empty_frame) == 0
    assert list(empty_frame.columns) == list(dataset_store.OUTPUT_COLUMNS)
    with pytest.raises(store.ReviewError):
        store.finalize(db, imported["lineage_id"], "reviewed_c", store.revision(db))


def test_mark_reviewed_and_read_snapshot(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    snapshot = store.read_snapshot(db, fixture["lineage_a"], "t1")
    assert snapshot["revision"] == store.revision(db) == 0
    result = store.mark_reviewed(db, "t1", True, snapshot["revision"])
    assert result["reviewed"] is True
    assert result["revision"] == 1
    assert store.review_status(db, "t1")["reviewed"] is True
    assert store.review_status(db, "t1")["revision"] == 1
    events = store.read_history(db, "t1")
    assert len(events) == 1
    assert events[0]["event_type"] == "mark_reviewed"
    assert events[0]["snapshot_sha256"] is None
    assert store.list_versions(db)[0]["dirty"] is False
    marked = store.read_snapshot(db, fixture["lineage_a"], "t1")
    assert marked["revision"] == 1
    store.mark_reviewed(db, "t1", False, marked["revision"])
    assert store.review_status(db, "t1")["reviewed"] is False


def test_finalized_version_is_dirtied_by_later_edits(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    final = store.finalize(db, fixture["lineage_a"], "reviewed_lock", 0)
    assert final["revision"] == 1
    listed = {item["lineage_id"]: item for item in store.list_versions(db)}
    assert listed[final["lineage_id"]]["frozen"] is True
    labels = _labels_for(
        store.load_frame(db, fixture["lineage_a"]), "t1", division=[1, 4]
    )
    store.save_labels(db, "t1", labels, final["revision"])
    listed = {item["lineage_id"]: item for item in store.list_versions(db)}
    assert listed[final["lineage_id"]]["frozen"] is False
    assert listed[final["lineage_id"]]["dirty"] is True
    conn = dataset_store.connect_readonly(db)
    try:
        with pytest.raises(ValueError):
            dataset_store.load_training_frame(
                conn, [final["lineage_id"]], require_human_verified=True
            )
    finally:
        conn.close()


def test_delete_purges_payloads_but_keeps_audit_metadata(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    labels = _labels_for(store.load_frame(db, fixture["lineage_a"]), "t1", division=[1])
    store.save_labels(db, "t1", labels, 0)
    revision_number = store.revision(db)
    result = store.delete_trajectory(db, "t1", revision_number, "t1")
    history = store.read_history(db, "t1")
    assert len(history) >= 2
    raw = dataset_store.connect_readonly(db)
    try:
        assert raw.execute(
            "SELECT COUNT(*) FROM review_payload WHERE trajectory_key = 't1'"
        ).fetchone()[0] == 0
        assert raw.execute(
            "SELECT COUNT(*) FROM review_event WHERE trajectory_key = 't1'"
        ).fetchone()[0] >= 2
    finally:
        raw.close()
    assert result["deleted"]["undo_payloads"] >= 1
    assert store.revision(db) == revision_number + 1

def _build_big_fov_store(tmp_path: Path) -> dict:
    root = tmp_path / "repo_big"
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    big = 2 ** 53 + 1
    rows = [
        _row("big", big, 0.0, 5.0),
        _row("big", big + 1, 0.5, 4.9),
    ]
    csv_path = _write_csv(root, "corpus_big.csv", rows)
    db = tmp_path / "big_store.sqlite3"
    conn = dataset_store.connect(db)
    try:
        dataset_store.initialize(conn)
        imported = _import(conn, root, csv_path, "corpus_big")
    finally:
        conn.close()
    return {"db": db, "lineage": imported["lineage_id"], "big": big}


def test_big_int64_ids_are_preserved_and_coercions_rejected(tmp_path: Path) -> None:
    fixture = _build_big_fov_store(tmp_path)
    db = fixture["db"]
    big = fixture["big"]
    trajectory = store.load_trajectory(db, "big")
    assert trajectory["fov_cell_index"].dtype == "int64"
    assert [int(value) for value in trajectory["fov_cell_index"]] == [big, big + 1]
    labels = _labels_for(
        store.load_frame(db, fixture["lineage"]), "big", division=[big]
    )
    saved = store.save_labels(db, "big", labels, 0)
    assert saved["revision"] == 1
    # float ids (would lose precision beyond 2**53) are rejected
    as_float = labels.copy()
    as_float["fov_cell_index"] = as_float["fov_cell_index"].astype(float)
    with pytest.raises(store.InvalidLabels):
        store.save_labels(db, "big", as_float, 1)
    # booleans are not ids
    as_bool = labels.copy()
    as_bool["fov_cell_index"] = True
    with pytest.raises(store.InvalidLabels):
        store.save_labels(db, "big", as_bool, 1)
    # numeric 0/1 labels are not true booleans
    numeric = labels.copy()
    numeric["division_point"] = numeric["division_point"].astype(int)
    with pytest.raises(store.InvalidLabels):
        store.save_labels(db, "big", numeric, 1)
    # CAS revision must be a non-bool integer, never int-cast from a float
    with pytest.raises(store.ReviewError):
        store.save_labels(db, "big", labels, 1.0)
    with pytest.raises(store.ReviewError):
        store.save_labels(db, "big", labels, True)
    assert store.revision(db) == 1


def test_snapshot_metadata_is_recorded(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    labels = _labels_for(store.load_frame(db, fixture["lineage_a"]), "t1",
                        division=[1])
    result = store.save_labels(db, "t1", labels, 0)
    conn = dataset_store.connect_readonly(db)
    try:
        snapshot = conn.execute(
            "SELECT * FROM review_snapshot WHERE sha256 = ?",
            (result["snapshot"]["sha256"],),
        ).fetchone()
        event = conn.execute(
            "SELECT snapshot_path, snapshot_sha256 FROM review_event"
            " WHERE event_type = 'save_labels' ORDER BY revision DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert snapshot is not None
    assert snapshot["integrity"] == "ok"
    assert int(snapshot["bytes"]) == result["snapshot"]["bytes"]
    assert event["snapshot_path"] == result["snapshot"]["path"]
    assert event["snapshot_sha256"] == result["snapshot"]["sha256"]


def test_delete_purges_all_undo_payloads(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    labels = _labels_for(store.load_frame(db, fixture["lineage_a"]), "t1",
                        division=[1])
    store.save_labels(db, "t1", labels, 0)
    labels2 = labels.copy()
    labels2["division_point"] = ~labels2["division_point"].astype(bool)
    store.save_labels(db, "t1", labels2, 1)
    revision_number = store.revision(db)
    result = store.delete_trajectory(db, "t1", revision_number, "t1")
    assert result["deleted"]["undo_payloads"] == 2
    conn = dataset_store.connect_readonly(db)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM review_payload WHERE trajectory_key = 't1'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_finalize_rejects_tampered_hash(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    raw = sqlite3.connect(str(db))
    try:
        raw.execute(
            "UPDATE trajectory SET content_hash = 'deadbeef'"
            " WHERE trajectory_key = 't1'"
        )
        raw.commit()
    finally:
        raw.close()
    with pytest.raises(store.ReviewError):
        store.finalize(db, fixture["lineage_a"], "tampered", 0)
    assert store.revision(db) == 0


def test_finalize_rejects_zero_observation_member(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    raw = sqlite3.connect(str(db))
    try:
        raw.execute(
            "DELETE FROM observation WHERE trajectory_id ="
            " (SELECT trajectory_id FROM trajectory WHERE trajectory_key = 't2')"
        )
        raw.commit()
    finally:
        raw.close()
    with pytest.raises(store.ReviewError):
        store.finalize(db, fixture["lineage_a"], "zero_obs", 0)
    assert store.revision(db) == 0


def test_remove_dataset_versions_keeps_shared_observations(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    finalized = store.finalize(db, fixture["lineage_a"], "reviewed_a", 0)
    assert finalized["trajectories"] == 2  # t_long drops under the R004 rule
    revision = store.revision(db)
    result = store.remove_dataset_versions(
        db,
        [finalized["lineage_id"]],
        revision,
        reason="synthetic duplicate cleanup",
    )
    assert result["revision"] == revision + 1
    assert result["members_removed"] == 2
    assert result["removed_datasets"] == ["reviewed_a"]
    assert [item["lineage_id"] for item in result["removed"]] == [
        finalized["lineage_id"]
    ]
    assert result["store_totals"] == {
        "versions": 2,
        "trajectories": 4,
        "observations": 15,
    }
    lineages = {item["lineage_id"] for item in store.list_versions(db)}
    assert lineages == {fixture["lineage_a"], fixture["lineage_b"]}
    conn = dataset_store.connect_readonly(db)
    try:
        for table in ("dataset",):
            assert (
                conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE dataset_id = 'reviewed_a'"
                ).fetchone()[0]
                == 0
            )
        for table in ("dataset_member", "ingest_provenance", "review_version_state"):
            assert (
                conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE version_id = ?",
                    (finalized["version_id"],),
                ).fetchone()[0]
                == 0
            )
        assert conn.execute("SELECT COUNT(*) FROM trajectory").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM observation").fetchone()[0] == 15
        events = conn.execute(
            "SELECT payload_json FROM review_event WHERE event_type = ?",
            ("remove_dataset_versions",),
        ).fetchall()
        assert len(events) == 1
        payload = json.loads(events[0]["payload_json"])
        assert payload["members_removed"] == 2
        assert payload["reason"] == "synthetic duplicate cleanup"
    finally:
        conn.close()


def test_remove_dataset_versions_refuses_to_orphan_trajectories(
    tmp_path: Path,
) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    with pytest.raises(store.ReviewError, match="lose every dataset membership"):
        store.remove_dataset_versions(
            db,
            [fixture["lineage_a"]],
            0,
            reason="must not orphan t2/t_long",
        )
    assert store.revision(db) == 0
    assert fixture["lineage_a"] in {
        item["lineage_id"] for item in store.list_versions(db)
    }


def test_remove_dataset_versions_is_revision_guarded(tmp_path: Path) -> None:
    fixture = _build_store(tmp_path)
    db = fixture["db"]
    with pytest.raises(store.RevisionConflict):
        store.remove_dataset_versions(
            db, [fixture["lineage_b"]], 99, reason="stale revision"
        )
    with pytest.raises(store.ReviewError):
        store.remove_dataset_versions(
            db, ["missing@v1"], store.revision(db), reason="unknown lineage"
        )
    with pytest.raises(store.InvalidDataset):
        store.remove_dataset_versions(db, [], store.revision(db), reason="empty")

