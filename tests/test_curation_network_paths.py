"""Curation database path and journal compatibility on network storage."""

from contextlib import closing
from pathlib import PurePosixPath, PureWindowsPath
import sqlite3
from urllib.parse import urlsplit

import pytest

from moma_lineage_analysis import curation_store as store
from test_curation_store import make_store, session


@pytest.mark.parametrize("path, expected", [
    (PureWindowsPath(r"C:\data folder\moma #1?.sqlite3"),
     "file:///C:/data%20folder/moma%20%231%3F.sqlite3?mode=rw"),
    (PureWindowsPath(r"\\server.example\share name\moma #1.sqlite3"),
     "file:////server.example/share%20name/moma%20%231.sqlite3?mode=rw"),
    (PureWindowsPath(r"\\?\UNC\server.example\share\moma.sqlite3"),
     "file:////server.example/share/moma.sqlite3?mode=rw"),
    (PureWindowsPath(r"\\?\C:\data\moma.sqlite3"),
     "file:///C:/data/moma.sqlite3?mode=rw"),
    (PurePosixPath("/tmp/data é.sqlite3"), "file:///tmp/data%20%C3%A9.sqlite3?mode=rw"),
])
def test_sqlite_uri_encodes_path_without_nonlocal_authority(path, expected):
    uri = store._sqlite_uri(path, "rw")
    assert uri == expected
    assert urlsplit(uri).netloc == ""


def test_detects_unc_extended_unc_and_mapped_drive(monkeypatch):
    assert store._is_network_path(PureWindowsPath(r"\\server\share\db.sqlite3"))
    assert store._is_network_path(PureWindowsPath(r"\\?\UNC\server\share\db.sqlite3"))
    monkeypatch.setattr(store._windows_drive_type.__globals__["os"], "name", "nt")
    monkeypatch.setattr(store, "_windows_drive_type", lambda root: 4 if root == "Z:\\" else 3)
    assert store._is_network_path(PureWindowsPath(r"Z:\experiment\db.sqlite3"))
    assert not store._is_network_path(PureWindowsPath(r"G:\experiment\db.sqlite3"))
    assert not store._is_network_path(PureWindowsPath(r"\\?\G:\experiment\db.sqlite3"))


def test_local_database_keeps_wal_and_full_synchronization(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_is_network_path", lambda path: False)
    db, *_ = make_store(tmp_path)
    with store._connect(db, write=True) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_network_policy_uses_delete_and_preserves_autosave_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_is_network_path", lambda path: True)
    db, _, labels, key, _, _, info = make_store(tmp_path)
    entry = store.load_dataset(db, info["dataset_id"])["trajectory_dict"][key]
    entry["_review_status"] = "reviewed"
    revision = store.save_changes(
        db, info["dataset_id"], expected_revision=0, trajectory_dict={key:entry},
        labeled_frames=labels, selected_keys={key}, current_key=key,
    )
    assert revision == 1
    recovered = store.load_dataset(db, info["dataset_id"])
    assert recovered["trajectory_dict"][key]["_review_status"] == "reviewed"
    with store._connect(db, write=True) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert not db.with_name(db.name + "-wal").exists()
    assert not db.with_name(db.name + "-shm").exists()


def test_existing_wal_database_can_move_to_network_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_is_network_path", lambda path: False)
    db, *_ = make_store(tmp_path)
    monkeypatch.setattr(store, "_is_network_path", lambda path: True)
    assert store.load_dataset(db, 1)["trajectory_count"] == 1
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_database_path_special_characters_round_trip(tmp_path):
    folder = tmp_path / "share space # percent% 中文"
    folder.mkdir()
    trajectories, labels, key = session(folder)
    db = folder / "curation #20% 中文.sqlite3"
    info = store.create_dataset(
        db, trajectory_dict=trajectories, labeled_frames=labels,
        selected_keys={key}, sources=store.fingerprint_sources([trajectories[key]["source_csv_path"]]),
        settings={}, logic_version="2.3", current_key=key,
    )
    assert store.load_dataset(db, info["dataset_id"])["current_key"] == key


def test_journal_mode_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_is_network_path", lambda path: True)
    with closing(sqlite3.connect(":memory:")) as connection:
        with pytest.raises(store.CurationStoreError, match="Could not enable DELETE"):
            store._configure_journal(connection, tmp_path / "db.sqlite3")

def test_journal_setup_failure_leaves_retryable_schema(tmp_path, monkeypatch):
    trajectories, labels, key = session(tmp_path / "input")
    db = tmp_path / store.DB_FILENAME
    arguments = dict(
        trajectory_dict=trajectories, labeled_frames=labels,
        selected_keys={key}, sources=store.fingerprint_sources([trajectories[key]["source_csv_path"]]),
        settings={}, logic_version="2.3", current_key=key,
    )
    with monkeypatch.context() as temporary:
        def fail_journal(*args):
            raise sqlite3.OperationalError("injected journal setup failure")
        temporary.setattr(store, "_configure_journal", fail_journal)
        with pytest.raises(store.CurationStoreError, match="journal setup failure"):
            store.create_dataset(db, **arguments)
    assert store.list_datasets(db) == []
    created = store.create_dataset(db, **arguments)
    assert created["trajectory_count"] == 1
    assert store.load_dataset(db, created["dataset_id"])["current_key"] == key
