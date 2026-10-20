"""Tests for the local SQLite dataset store (phase 1).

These tests use synthetic corpora only; they do not touch datasets/store/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from moma_lineage_analysis.trajectory_cleaning import dataset_store


def _frame(keys=("traj_B", "traj_A"), modified=False):
    rows = []
    for key_index, key in enumerate(keys):
        division_rows = {2}
        if modified and key == "traj_B":
            division_rows = {1}
        for index in range(4):
            rows.append(
                {
                    "trajectory_key": key,
                    "fov_cell_index": 100 * (key_index + 1) + index,
                    "time_h": 0.5 * index,
                    "spine_length": 5.0 - 0.5 * index + 0.1 * key_index,
                    "division_point": index in division_rows,
                    "excluded_point": index == 3 and key == "traj_A",
                    "source_dataset": "synthetic",
                    "source_file": "synthetic_source.csv",
                    "source_file_sha256": "0" * 64,
                    "source_trajectory_key": key,
                    "source_fov_cell_index": 100 * (key_index + 1) + index,
                }
            )
    return pd.DataFrame(rows, columns=dataset_store.OUTPUT_COLUMNS)


def _write_corpus(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _import(connection, root: Path, csv_path: Path, dataset_id: str):
    return dataset_store.import_corpus(
        connection,
        repo_root=root,
        csv_path=csv_path,
        dataset_id=dataset_id,
        kind="batch",
        label="synthetic test corpus",
        round_name="R999",
        verification_basis="synthetic",
        human_verified=True,
    )


def test_import_export_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    csv_path = _write_corpus(tmp_path / "corpus.csv", _frame())
    legacy_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    connection = dataset_store.connect(tmp_path / "store.sqlite3")
    try:
        dataset_store.initialize(connection)
        result = _import(connection, tmp_path, csv_path, "synthetic_a")
        assert result["status"] == "imported"
        assert result["rows"] == 8
        assert result["trajectories"] == 2
        assert result["legacy_csv_sha256"] == legacy_sha

        export_dir = tmp_path / "exports"
        verification = dataset_store.verify_store(
            connection,
            repo_root=tmp_path,
            export_dir=export_dir,
            expected_versions=["synthetic_a@v1"],
        )
        assert verification["status"] == "PASS"
        export_sha = hashlib.sha256(
            (export_dir / "synthetic_a_v1.csv").read_bytes()
        ).hexdigest()
        assert export_sha == legacy_sha
    finally:
        connection.close()


def test_verify_accepts_retired_legacy_csv(tmp_path: Path) -> None:
    csv_path = _write_corpus(tmp_path / "corpus.csv", _frame())
    connection = dataset_store.connect(tmp_path / "store.sqlite3")
    try:
        dataset_store.initialize(connection)
        _import(connection, tmp_path, csv_path, "synthetic_a")
        csv_path.unlink()
        export_dir = tmp_path / "exports"
        retirement_manifest = tmp_path / "retired_csv_manifest.json"
        retirement_manifest.write_text(
            json.dumps({"retired_files": [{"lineage_id": "synthetic_a@v1"}]}),
            encoding="utf-8",
        )
        retired = dataset_store.verify_store(
            connection,
            repo_root=tmp_path,
            export_dir=export_dir,
            expected_versions=["synthetic_a@v1"],
            retirement_manifest=retirement_manifest,
        )
        assert retired["status"] == "PASS"
        version_check = next(
            check for check in retired["checks"] if check["check"] == "dataset_version"
        )
        assert version_check["legacy_csv_present"] is False
        assert version_check["legacy_csv_state"] == "retired"
        unrecorded = dataset_store.verify_store(
            connection,
            repo_root=tmp_path,
            export_dir=export_dir,
            expected_versions=["synthetic_a@v1"],
        )
        assert unrecorded["status"] == "FAIL"
        unrecorded_check = next(
            check for check in unrecorded["checks"] if check["check"] == "dataset_version"
        )
        assert unrecorded_check["legacy_csv_state"] == "missing_unrecorded"
        strict = dataset_store.verify_store(
            connection,
            repo_root=tmp_path,
            export_dir=export_dir,
            expected_versions=["synthetic_a@v1"],
            require_legacy_csv=True,
        )
        assert strict["status"] == "FAIL"
        assert any("legacy CSV missing" in failure for failure in strict["failures"])
    finally:
        connection.close()

def test_reimport_same_file_is_idempotent(tmp_path: Path) -> None:
    csv_path = _write_corpus(tmp_path / "corpus.csv", _frame())
    connection = dataset_store.connect(tmp_path / "store.sqlite3")
    try:
        dataset_store.initialize(connection)
        _import(connection, tmp_path, csv_path, "synthetic_a")
        again = _import(connection, tmp_path, csv_path, "synthetic_a")
        assert again["status"] == "already_imported"
        assert again["version"] == 1
        counts = connection.execute("SELECT COUNT(*) FROM observation").fetchone()[0]
        assert counts == 8
    finally:
        connection.close()


def test_second_version_reuses_trajectory_rows(tmp_path: Path) -> None:
    first = _write_corpus(tmp_path / "first.csv", _frame())
    extended = pd.concat(
        [
            _frame(),
            _frame(keys=("traj_C",)).assign(
                fov_cell_index=lambda frame: frame["fov_cell_index"] + 500,
                source_fov_cell_index=lambda frame: frame["fov_cell_index"] + 500,
            ),
        ],
        ignore_index=True,
    )
    second = _write_corpus(tmp_path / "second.csv", extended)
    connection = dataset_store.connect(tmp_path / "store.sqlite3")
    try:
        dataset_store.initialize(connection)
        _import(connection, tmp_path, first, "synthetic_a")
        result = _import(connection, tmp_path, second, "synthetic_b")
        assert result["inserted_trajectories"] == 1
        assert result["inserted_observations"] == 4
        rows = connection.execute("SELECT COUNT(*) FROM observation").fetchone()[0]
        trajectories = connection.execute(
            "SELECT COUNT(*) FROM trajectory"
        ).fetchone()[0]
        assert rows == 12
        assert trajectories == 3
        verification = dataset_store.verify_store(
            connection,
            repo_root=tmp_path,
            expected_versions=["synthetic_a@v1", "synthetic_b@v1"],
        )
        assert verification["status"] == "PASS"
    finally:
        connection.close()


def test_in_place_label_change_is_rejected(tmp_path: Path) -> None:
    original = _write_corpus(tmp_path / "original.csv", _frame())
    changed = _write_corpus(tmp_path / "changed.csv", _frame(modified=True))
    connection = dataset_store.connect(tmp_path / "store.sqlite3")
    try:
        dataset_store.initialize(connection)
        _import(connection, tmp_path, original, "synthetic_a")
        with pytest.raises(ValueError, match="different content"):
            _import(connection, tmp_path, changed, "synthetic_b")
        assert connection.execute(
            "SELECT COUNT(*) FROM dataset_version"
        ).fetchone()[0] == 1
    finally:
        connection.close()


def test_lineage_ids(tmp_path: Path) -> None:
    csv_path = _write_corpus(tmp_path / "corpus.csv", _frame())
    connection = dataset_store.connect(tmp_path / "store.sqlite3")
    try:
        dataset_store.initialize(connection)
        result = _import(connection, tmp_path, csv_path, "synthetic_a")
        assert result["lineage_id"] == "synthetic_a@v1"
        assert result["content_ref"].startswith("synthetic_a@v1#")
        assert len(result["content_ref"].split("#")[1]) == 12
    finally:
        connection.close()


def test_backup_creates_compressed_snapshot(tmp_path: Path) -> None:
    csv_path = _write_corpus(tmp_path / "corpus.csv", _frame())
    db_path = tmp_path / "store.sqlite3"
    connection = dataset_store.connect(db_path)
    try:
        dataset_store.initialize(connection)
        _import(connection, tmp_path, csv_path, "synthetic_a")
    finally:
        connection.close()
    backup = dataset_store.backup_database(db_path, tmp_path / "backups")
    assert Path(backup["path"]).is_file()
    assert Path(backup["path"]).suffix == ".gz"
    assert len(backup["sha256"]) == 64
