"""Review/edit backend for the local training trajectory database.

Additive review extension on top of the frozen SQLite dataset store
(moma_lineage_analysis.trajectory_cleaning.dataset_store).  The core schema and
its schema_version are NOT changed: the review_* extension tables are created on
the first mutating call only, so read-only catalogue/load calls never write.

User-approved semantics (task-23):

- Every mutation is CAS-guarded by a single global revision counter
  (revision(db)); a stale expected_revision raises RevisionConflict and changes
  nothing.
- Editing/deleting a trajectory is allowed even for frozen versions.  Every
  version that references the trajectory is invalidated: its original pins,
  counts, frozen flag and legacy CSV hashes are copied into
  review_version_state, it is marked frozen = 0, its legacy export pin is
  cleared and its working canonical hash/counts/ordinals are recomputed in
  place.  Core training paths (dataset_store.resolve_lineage /
  load_training_frame) therefore refuse invalidated versions until a new
  version is finalized from them.
- Deletion is a hard, permanent delete from the database of the selected
  trajectory (observations + memberships across every version + the trajectory
  row).  There is no restore UI/path; only label-ish undo payloads are purged
  while non-payload audit metadata is kept.
- save_labels / undo_labels / delete_trajectory take an online SQLite backup
  before mutating.  The writer holds BEGIN IMMEDIATE (RESERVED lock) and the
  snapshot is taken from a separate read-only connection, then verified with
  PRAGMA integrity_check; path/sha256/bytes are recorded in review_snapshot and
  in the audit event.
- Whole duplicate dataset versions can be removed (dataset-level counterpart
  of a trajectory delete) only when every trajectory stays referenced by
  another version; the removal is revision-guarded, backed up and audited.
- Only labels may change; time_h / spine_length (measurements) are immutable.
  New frozen versions created by finalize have no legacy CSV path but keep a
  review-native legacy_csv_sha256 equal to the byte-exact database export, and
  are attested human_verified = 1.  By default the R004 long exclusion-run
  filter is applied to the NEW membership only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from moma_lineage_analysis.trajectory_cleaning import dataset_store

__all__ = [
    "REVIEW_SCHEMA_VERSION",
    "ReviewError",
    "RevisionConflict",
    "ConfirmationError",
    "InvalidLabels",
    "InvalidDataset",
    "list_versions",
    "load_frame",
    "load_trajectory",
    "read_snapshot",
    "revision",
    "list_trajectories",
    "review_status",
    "read_history",
    "impact",
    "save_labels",
    "mark_reviewed",
    "undo_labels",
    "delete_trajectory",
    "finalize",
    "remove_dataset_versions",
]

REVIEW_SCHEMA_VERSION = 1
DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
CANONICAL_COLUMNS = list(dataset_store.OUTPUT_COLUMNS)
R004_MAX_EXCLUSION_RUN = 5


class ReviewError(ValueError):
    """Base error for the review backend."""


class RevisionConflict(ReviewError):
    """Raised when the supplied expected_revision is stale."""


class ConfirmationError(ReviewError):
    """Raised when a destructive confirmation token does not match."""


class InvalidLabels(ReviewError):
    """Raised when a label frame is malformed or changes immutable data."""


class InvalidDataset(ReviewError):
    """Raised when a requested dataset id/version is invalid."""


# ---------------------------------------------------------------------------
# Paths / connections
# ---------------------------------------------------------------------------


def _db_file(db: str | Path) -> Path:
    path = Path(db)
    if not path.is_file():
        raise FileNotFoundError(f"dataset store not found: {path}")
    return path


def _connect_write(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    # WAL + NORMAL avoids two full fsyncs per label click. Committed data still
    # survives a process crash; only OS/power loss can drop the last transaction,
    # and every mutation has its own pre-change snapshot/undo payload.
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _connect_read(db_path: Path) -> sqlite3.Connection:
    return dataset_store.connect_readonly(db_path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Extension schema (additive tables only; core schema_version untouched)
# ---------------------------------------------------------------------------

_REVIEW_SQL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS review_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        schema_version INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_trajectory (
        trajectory_key TEXT PRIMARY KEY,
        reviewed INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0, 1)),
        revision INTEGER NOT NULL DEFAULT 0,
        last_actor TEXT,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_event (
        event_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        trajectory_key TEXT,
        dataset_id TEXT,
        lineage_id TEXT,
        actor TEXT,
        payload_json TEXT,
        snapshot_path TEXT,
        snapshot_sha256 TEXT,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_payload (
        payload_id TEXT PRIMARY KEY,
        trajectory_key TEXT NOT NULL,
        revision INTEGER NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_snapshot (
        snapshot_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        integrity TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_version_state (
        version_id INTEGER PRIMARY KEY,
        original_frozen INTEGER NOT NULL,
        original_canonical_hash TEXT NOT NULL,
        original_rows INTEGER NOT NULL,
        original_trajectories INTEGER NOT NULL,
        original_division_points INTEGER NOT NULL,
        original_excluded_points INTEGER NOT NULL,
        original_overlap_rows INTEGER NOT NULL,
        original_legacy_csv_path TEXT,
        original_legacy_csv_sha256 TEXT,
        working_hash TEXT NOT NULL,
        dirty INTEGER NOT NULL DEFAULT 1,
        invalidated_at TEXT NOT NULL,
        revision INTEGER NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_review_event_key ON review_event(trajectory_key)",
    "CREATE INDEX IF NOT EXISTS idx_review_payload_key ON review_payload(trajectory_key)",
)


def _has_extension(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'review_state'"
    ).fetchone()
    return row is not None


def _ensure_extension(connection: sqlite3.Connection) -> None:
    if _has_extension(connection):
        return
    for statement in _REVIEW_SQL:
        connection.execute(statement)
    connection.execute(
        """INSERT OR IGNORE INTO review_state
           (singleton, schema_version, revision, updated_at)
           VALUES (1, ?, 0, ?)""",
        (REVIEW_SCHEMA_VERSION, _now()),
    )


def _read_revision(connection: sqlite3.Connection) -> int:
    try:
        row = connection.execute(
            "SELECT revision FROM review_state WHERE singleton = 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["revision"]) if row is not None else 0


# ---------------------------------------------------------------------------
# Generic read helpers
# ---------------------------------------------------------------------------


def _version_row(connection: sqlite3.Connection, lineage_id: str) -> sqlite3.Row:
    dataset_id, version, hash_prefix = dataset_store.parse_lineage_id(lineage_id)
    row = connection.execute(
        """SELECT v.version_id, v.dataset_id, v.version, v.canonical_hash,
                  v.rows, v.trajectories, v.division_points, v.excluded_points,
                  v.overlap_rows, v.legacy_csv_path, v.legacy_csv_sha256,
                  v.frozen, v.created_at, d.kind, d.label, d.round_name,
                  d.verification_basis, d.human_verified
           FROM dataset_version v
           JOIN dataset d ON d.dataset_id = v.dataset_id
           WHERE v.dataset_id = ? AND v.version = ?""",
        (dataset_id, version),
    ).fetchone()
    if row is None:
        raise ReviewError(f"unknown dataset lineage: {lineage_id!r}")
    if hash_prefix is not None and not str(row["canonical_hash"]).startswith(
        hash_prefix
    ):
        raise ReviewError(
            f"dataset lineage {lineage_id!r} content hash does not match the store"
        )
    return row


def _trajectory_row(connection: sqlite3.Connection, trajectory_key: str) -> sqlite3.Row:
    if not isinstance(trajectory_key, str) or not trajectory_key:
        raise ReviewError("trajectory_key must be a non-empty string")
    row = connection.execute(
        "SELECT * FROM trajectory WHERE trajectory_key = ?", (trajectory_key,)
    ).fetchone()
    if row is None:
        raise ReviewError(f"unknown trajectory: {trajectory_key!r}")
    return row


def _version_counts(connection: sqlite3.Connection, version_id: int) -> dict[str, int]:
    row = connection.execute(
        """SELECT COUNT(o.trajectory_id) AS rows,
                  COUNT(DISTINCT m.trajectory_id) AS trajectories,
                  COALESCE(SUM(o.division_point), 0) AS division_points,
                  COALESCE(SUM(o.excluded_point), 0) AS excluded_points,
                  COALESCE(SUM(
                      CASE WHEN o.division_point = 1 AND o.excluded_point = 1
                           THEN 1 ELSE 0 END), 0) AS overlap_rows
           FROM dataset_member m
           LEFT JOIN observation o ON o.trajectory_id = m.trajectory_id
           WHERE m.version_id = ?""",
        (version_id,),
    ).fetchone()
    return {
        key: int(row[key] or 0)
        for key in (
            "rows",
            "trajectories",
            "division_points",
            "excluded_points",
            "overlap_rows",
        )
    }


def _version_canonical_hash(connection: sqlite3.Connection, version_id: int) -> str:
    members = connection.execute(
        """SELECT t.trajectory_key, t.content_hash
           FROM dataset_member m
           JOIN trajectory t ON t.trajectory_id = m.trajectory_id
           WHERE m.version_id = ? ORDER BY t.trajectory_key""",
        (version_id,),
    ).fetchall()
    digest = hashlib.sha256()
    for member in members:
        digest.update(
            f"{member['trajectory_key']}\t{member['content_hash']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _member_count(connection: sqlite3.Connection, version_id: int) -> int:
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM dataset_member WHERE version_id = ?",
            (version_id,),
        ).fetchone()[0]
    )


def _membership_version_ids(
    connection: sqlite3.Connection, trajectory_id: int
) -> list[int]:
    rows = connection.execute(
        "SELECT DISTINCT version_id FROM dataset_member WHERE trajectory_id = ?"
        " ORDER BY version_id",
        (trajectory_id,),
    ).fetchall()
    return [int(row[0]) for row in rows]


def _dirty_version_ids(connection: sqlite3.Connection) -> dict[int, bool]:
    if not _has_extension(connection):
        return {}
    rows = connection.execute(
        "SELECT version_id, dirty FROM review_version_state WHERE dirty = 1"
    ).fetchall()
    return {int(row["version_id"]): bool(row["dirty"]) for row in rows}


def _longest_exclusion_run(connection: sqlite3.Connection, trajectory_id: int) -> int:
    values = [
        int(row[0])
        for row in connection.execute(
            "SELECT excluded_point FROM observation WHERE trajectory_id = ?"
            " ORDER BY fov_cell_index",
            (trajectory_id,),
        ).fetchall()
    ]
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trajectory_key": pd.Series(dtype="object"),
            "fov_cell_index": pd.Series(dtype="int64"),
            "time_h": pd.Series(dtype="float64"),
            "spine_length": pd.Series(dtype="float64"),
            "division_point": pd.Series(dtype="bool"),
            "excluded_point": pd.Series(dtype="bool"),
            "source_dataset": pd.Series(dtype="object"),
            "source_file": pd.Series(dtype="object"),
            "source_file_sha256": pd.Series(dtype="object"),
            "source_trajectory_key": pd.Series(dtype="object"),
            "source_fov_cell_index": pd.Series(dtype="int64"),
        }
    )
    return frame.loc[:, CANONICAL_COLUMNS]


def _frame_from_records(records: Sequence[Sequence[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(records), columns=CANONICAL_COLUMNS)
    for column in ("fov_cell_index", "source_fov_cell_index"):
        frame[column] = frame[column].astype("int64")
    for column in ("time_h", "spine_length"):
        frame[column] = frame[column].astype("float64")
    for column in ("division_point", "excluded_point"):
        frame[column] = frame[column].astype("bool")
    return frame.loc[:, CANONICAL_COLUMNS]


def _trajectory_frame(
    connection: sqlite3.Connection, trajectory_key: str
) -> pd.DataFrame:
    records = connection.execute(
        """SELECT t.trajectory_key, o.fov_cell_index, o.time_h,
                  o.spine_length, o.division_point, o.excluded_point,
                  t.source_dataset, t.source_file, t.source_file_sha256,
                  t.source_trajectory_key,
                  o.fov_cell_index AS source_fov_cell_index
           FROM trajectory t
           JOIN observation o ON o.trajectory_id = t.trajectory_id
           WHERE t.trajectory_key = ?
           ORDER BY o.fov_cell_index""",
        (trajectory_key,),
    ).fetchall()
    if not records:
        return _empty_frame()
    return _frame_from_records([tuple(record) for record in records])


# ---------------------------------------------------------------------------
# Public read-only API
# ---------------------------------------------------------------------------


def revision(db: str | Path) -> int:
    """Return the global CAS revision (0 when the review extension is absent)."""

    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        return _read_revision(connection)
    finally:
        connection.close()


def list_versions(db: str | Path) -> list[dict[str, Any]]:
    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        dirty = _dirty_version_ids(connection)
        rows = connection.execute(
            """SELECT v.version_id, v.dataset_id, v.version, v.rows,
                      v.trajectories, v.division_points, v.excluded_points,
                      v.overlap_rows, v.frozen, d.label, d.kind, d.round_name,
                      d.human_verified
               FROM dataset_version v
               JOIN dataset d ON d.dataset_id = v.dataset_id
               ORDER BY v.dataset_id, v.version"""
        ).fetchall()
        return [
            {
                "lineage_id": dataset_store.version_lineage_id(
                    str(row["dataset_id"]), int(row["version"])
                ),
                "label": str(row["label"]),
                "frozen": bool(int(row["frozen"])),
                "dirty": bool(dirty.get(int(row["version_id"]), False)),
                "counts": {
                    "rows": int(row["rows"]),
                    "trajectories": int(row["trajectories"]),
                    "division_points": int(row["division_points"]),
                    "excluded_points": int(row["excluded_points"]),
                    "overlap_rows": int(row["overlap_rows"]),
                },
                "dataset_id": str(row["dataset_id"]),
                "version": int(row["version"]),
                "kind": str(row["kind"]),
                "round_name": row["round_name"],
                "human_verified": bool(int(row["human_verified"])),
            }
            for row in rows
        ]
    finally:
        connection.close()


def load_frame(db: str | Path, lineage_id: str) -> pd.DataFrame:
    """Load a version's current working frame (dirty/unfrozen allowed)."""

    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        row = _version_row(connection, lineage_id)
        version_id = int(row["version_id"])
        if _member_count(connection, version_id) == 0:
            return _empty_frame()
        return dataset_store.export_version(connection, version_id)
    finally:
        connection.close()


def load_trajectory(db: str | Path, trajectory_key: str) -> pd.DataFrame:
    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        _trajectory_row(connection, trajectory_key)
        return _trajectory_frame(connection, trajectory_key)
    finally:
        connection.close()


def read_snapshot(
    db: str | Path,
    lineage_id: str,
    trajectory_key: str | None = None,
) -> dict[str, Any]:
    """Read a frame and the CAS revision inside one consistent read snapshot.

    Returns {lineage_id, trajectory_key, revision, frame, version_id, lineage}.
    The UI must pass the returned revision to save_labels so that labels and CAS
    refer to the same database state (a later write only conflicts if another
    mutation landed in between).
    """

    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        connection.execute("BEGIN")
        try:
            revision_number = _read_revision(connection)
            row = _version_row(connection, lineage_id)
            version_id = int(row["version_id"])
            if trajectory_key is None:
                if _member_count(connection, version_id) == 0:
                    frame = _empty_frame()
                else:
                    frame = dataset_store.export_version(connection, version_id)
            else:
                _trajectory_row(connection, trajectory_key)
                membership = connection.execute(
                    """SELECT 1 FROM dataset_member m
                       JOIN trajectory t ON t.trajectory_id = m.trajectory_id
                       WHERE m.version_id = ? AND t.trajectory_key = ?""",
                    (version_id, trajectory_key),
                ).fetchone()
                if membership is None:
                    raise ReviewError(
                        f"trajectory {trajectory_key!r} is not a member of "
                        f"version {lineage_id!r}"
                    )
                frame = _trajectory_frame(connection, trajectory_key)
            lineage = {
                "lineage_id": dataset_store.version_lineage_id(
                    str(row["dataset_id"]), int(row["version"])
                ),
                "dataset_id": str(row["dataset_id"]),
                "version": int(row["version"]),
                "canonical_hash": str(row["canonical_hash"]),
                "frozen": bool(int(row["frozen"])),
                "dirty": bool(_dirty_version_ids(connection).get(version_id, False)),
                "rows": int(row["rows"]),
                "trajectories": int(row["trajectories"]),
            }
            return {
                "lineage_id": dataset_store.version_lineage_id(
                    str(row["dataset_id"]), int(row["version"])
                ),
                "trajectory_key": trajectory_key,
                "revision": revision_number,
                "frame": frame,
                "version_id": version_id,
                "lineage": lineage,
            }
        finally:
            connection.rollback()
    finally:
        connection.close()


def list_trajectories(db: str | Path, lineage_id: str) -> pd.DataFrame:
    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        row = _version_row(connection, lineage_id)
        version_id = int(row["version_id"])
        base = connection.execute(
            """SELECT m.trajectory_id, m.ordinal, t.trajectory_key,
                      t.source_dataset,
                      COUNT(o.trajectory_id) AS rows,
                      COALESCE(SUM(o.division_point), 0) AS division_points,
                      COALESCE(SUM(o.excluded_point), 0) AS excluded_points
               FROM dataset_member m
               JOIN trajectory t ON t.trajectory_id = m.trajectory_id
               LEFT JOIN observation o ON o.trajectory_id = m.trajectory_id
               WHERE m.version_id = ?
               GROUP BY m.trajectory_id, m.ordinal, t.trajectory_key,
                        t.source_dataset
               ORDER BY m.ordinal""",
            (version_id,),
        ).fetchall()
        extension = _has_extension(connection)
        review_map: dict[str, sqlite3.Row] = {}
        dirty_trajectory_ids: set[int] = set()
        if extension:
            review_map = {
                str(item["trajectory_key"]): item
                for item in connection.execute(
                    "SELECT trajectory_key, reviewed, revision FROM review_trajectory"
                ).fetchall()
            }
            dirty_trajectory_ids = {
                int(item[0])
                for item in connection.execute(
                    """SELECT DISTINCT m.trajectory_id
                       FROM dataset_member m
                       JOIN review_version_state s ON s.version_id = m.version_id
                       WHERE s.dirty = 1"""
                ).fetchall()
            }
        records = []
        for item in base:
            key = str(item["trajectory_key"])
            review = review_map.get(key)
            records.append(
                (
                    key,
                    str(item["source_dataset"]),
                    int(item["rows"]),
                    int(item["division_points"]),
                    int(item["excluded_points"]),
                    bool(int(review["reviewed"])) if review is not None else False,
                    int(review["revision"]) if review is not None else 0,
                    int(item["trajectory_id"]) in dirty_trajectory_ids,
                )
            )
        frame = pd.DataFrame.from_records(
            records,
            columns=[
                "trajectory_key",
                "source_dataset",
                "rows",
                "division_points",
                "excluded_points",
                "reviewed",
                "revision",
                "dirty",
            ],
        )
        if not frame.empty:
            frame["reviewed"] = frame["reviewed"].astype(bool)
            frame["dirty"] = frame["dirty"].astype(bool)
        return frame
    finally:
        connection.close()


def review_status(db: str | Path, trajectory_key: str) -> dict[str, Any]:
    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        exists = (
            connection.execute(
                "SELECT 1 FROM trajectory WHERE trajectory_key = ?",
                (trajectory_key,),
            ).fetchone()
            is not None
        )
        status = {
            "trajectory_key": trajectory_key,
            "exists": bool(exists),
            "reviewed": False,
            "revision": 0,
            "last_actor": None,
            "last_event": None,
            "updated_at": None,
        }
        if not _has_extension(connection):
            return status
        row = connection.execute(
            "SELECT * FROM review_trajectory WHERE trajectory_key = ?",
            (trajectory_key,),
        ).fetchone()
        if row is not None:
            status.update(
                {
                    "reviewed": bool(int(row["reviewed"])),
                    "revision": int(row["revision"]),
                    "last_actor": row["last_actor"],
                    "updated_at": row["updated_at"],
                }
            )
        event = connection.execute(
            """SELECT event_type, actor FROM review_event
               WHERE trajectory_key = ?
               ORDER BY revision DESC, rowid DESC LIMIT 1""",
            (trajectory_key,),
        ).fetchone()
        if event is not None:
            status["last_event"] = str(event["event_type"])
            if status["last_actor"] is None:
                status["last_actor"] = event["actor"]
        return status
    finally:
        connection.close()


def read_history(
    db: str | Path, trajectory_key: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    limit = int(limit)
    if limit < 1 or limit > 1000:
        raise ReviewError("limit must be between 1 and 1000")
    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        if not _has_extension(connection):
            return []
        columns = """
            event_id, revision, event_type, trajectory_key, dataset_id,
            lineage_id, actor, snapshot_path, snapshot_sha256, created_at
        """
        if trajectory_key is None:
            rows = connection.execute(
                f"SELECT {columns} FROM review_event"
                " ORDER BY revision DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT {columns} FROM review_event WHERE trajectory_key = ?"
                " ORDER BY revision DESC, rowid DESC LIMIT ?",
                (str(trajectory_key), limit),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def impact(db: str | Path, trajectory_key: str) -> dict[str, Any]:
    db_path = _db_file(db)
    connection = _connect_read(db_path)
    try:
        trajectory = _trajectory_row(connection, trajectory_key)
        trajectory_id = int(trajectory["trajectory_id"])
        version_ids = _membership_version_ids(connection, trajectory_id)
        dirty = _dirty_version_ids(connection)
        versions: list[dict[str, Any]] = []
        total_version_rows = 0
        observation_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM observation WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()[0]
        )
        for version_id in version_ids:
            row = connection.execute(
                """SELECT v.dataset_id, v.version, v.frozen, v.rows, v.trajectories
                   FROM dataset_version v WHERE v.version_id = ?""",
                (version_id,),
            ).fetchone()
            total_version_rows += int(row["rows"])
            versions.append(
                {
                    "lineage_id": dataset_store.version_lineage_id(
                        str(row["dataset_id"]), int(row["version"])
                    ),
                    "frozen": bool(int(row["frozen"])),
                    "dirty": bool(dirty.get(version_id, False)),
                    "rows": int(row["rows"]),
                    "trajectories": int(row["trajectories"]),
                    "affected_rows": observation_rows,
                }
            )
        can_undo = False
        reviewed = False
        if _has_extension(connection):
            can_undo = (
                connection.execute(
                    "SELECT 1 FROM review_payload"
                    " WHERE trajectory_key = ? AND kind = 'label_undo' LIMIT 1",
                    (trajectory_key,),
                ).fetchone()
                is not None
            )
            row = connection.execute(
                "SELECT reviewed FROM review_trajectory WHERE trajectory_key = ?",
                (trajectory_key,),
            ).fetchone()
            reviewed = bool(int(row["reviewed"])) if row is not None else False
        return {
            "trajectory_key": trajectory_key,
            "content_hash": str(trajectory["content_hash"]),
            "reviewed": reviewed,
            "can_undo": can_undo,
            "versions": versions,
            "counts": {
                "versions": len(versions),
                "memberships": len(version_ids),
                "trajectory_rows": observation_rows,
                "total_version_rows": total_version_rows,
            },
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Mutation infrastructure
# ---------------------------------------------------------------------------


def _take_snapshot(
    db_path: Path,
    writer: sqlite3.Connection,
    event_type: str,
    revision_number: int,
) -> dict[str, Any]:
    backup_dir = db_path.parent / "review_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = backup_dir / (
        f"{db_path.stem}_review_{stamp}_r{revision_number}_{event_type}.sqlite3"
    )
    source = sqlite3.connect(
        f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True
    )
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    check = sqlite3.connect(f"file:{destination.as_posix()}?mode=ro", uri=True)
    try:
        integrity = str(check.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        check.close()
    if integrity != "ok":
        raise ReviewError(f"snapshot integrity check failed: {integrity}")
    sha256 = dataset_store.sha256_file(destination)
    size = destination.stat().st_size
    snapshot_id = uuid.uuid4().hex
    writer.execute(
        """INSERT INTO review_snapshot
           (snapshot_id, revision, event_type, path, sha256, bytes,
            integrity, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot_id,
            revision_number,
            event_type,
            str(destination),
            sha256,
            size,
            integrity,
            _now(),
        ),
    )
    return {
        "snapshot_id": snapshot_id,
        "path": str(destination),
        "sha256": sha256,
        "bytes": size,
        "integrity": integrity,
    }


def _insert_event(
    connection: sqlite3.Connection,
    *,
    revision_number: int,
    event_type: str,
    actor: str | None,
    event_info: dict[str, Any],
    snapshot: dict[str, Any] | None,
) -> str:
    event_id = uuid.uuid4().hex
    snapshot = snapshot or {}
    connection.execute(
        """INSERT INTO review_event
           (event_id, revision, event_type, trajectory_key, dataset_id,
            lineage_id, actor, payload_json, snapshot_path, snapshot_sha256,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id,
            revision_number,
            event_type,
            event_info.get("trajectory_key"),
            event_info.get("dataset_id"),
            event_info.get("lineage_id"),
            actor,
            json.dumps(event_info.get("event_payload"), sort_keys=True),
            snapshot.get("path"),
            snapshot.get("sha256"),
            _now(),
        ),
    )
    return event_id


def _run_mutation(
    db: str | Path,
    expected_revision: int,
    event_type: str,
    actor: str | None,
    mutate: Callable[[sqlite3.Connection, int], tuple[dict[str, Any], dict[str, Any]]],
    *,
    backup: bool,
) -> dict[str, Any]:
    db_path = _db_file(db)
    expected = _coerce_revision(expected_revision)
    connection = _connect_write(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_extension(connection)
        current = _read_revision(connection)
        if expected != current:
            raise RevisionConflict(
                f"stale revision: expected {expected_revision!r}, "
                f"current is {current}"
            )
        snapshot = (
            _take_snapshot(db_path, connection, event_type, current) if backup else None
        )
        new_revision = current + 1
        event_info, result = mutate(connection, new_revision)
        _insert_event(
            connection,
            revision_number=new_revision,
            event_type=event_type,
            actor=actor,
            event_info=event_info,
            snapshot=snapshot,
        )
        connection.execute(
            "UPDATE review_state SET revision = ?, updated_at = ? WHERE singleton = 1",
            (new_revision, _now()),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    output = dict(result)
    output["revision"] = new_revision
    if snapshot is not None:
        output["snapshot"] = {
            "path": snapshot["path"],
            "sha256": snapshot["sha256"],
            "bytes": snapshot["bytes"],
        }
    return output


def _renumber_ordinals(connection: sqlite3.Connection, version_id: int) -> None:
    members = [
        int(row[0])
        for row in connection.execute(
            "SELECT trajectory_id FROM dataset_member WHERE version_id = ?"
            " ORDER BY ordinal",
            (version_id,),
        ).fetchall()
    ]
    connection.execute(
        "UPDATE dataset_member SET ordinal = -ordinal - 1 WHERE version_id = ?",
        (version_id,),
    )
    for ordinal, trajectory_id in enumerate(members, start=1):
        connection.execute(
            "UPDATE dataset_member SET ordinal = ?"
            " WHERE version_id = ? AND trajectory_id = ?",
            (ordinal, version_id, trajectory_id),
        )


def _mark_version_dirty(
    connection: sqlite3.Connection, version_id: int, revision_number: int,
    counts_delta: tuple[int, int, int, int, int] | None = None,
) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM dataset_version WHERE version_id = ?", (version_id,)
    ).fetchone()
    if row is None:
        raise ReviewError(f"unknown version_id {version_id}")
    if counts_delta is None:
        counts = _version_counts(connection, version_id)
    else:
        # Label edits and deletions change counts by a known delta. Avoid four
        # full 400k-observation aggregate scans per click.
        row_delta, trajectory_delta, division_delta, exclusion_delta, overlap_delta = counts_delta
        counts = {
            'rows': int(row['rows']) + row_delta,
            'trajectories': int(row['trajectories']) + trajectory_delta,
            'division_points': int(row['division_points']) + division_delta,
            'excluded_points': int(row['excluded_points']) + exclusion_delta,
            'overlap_rows': int(row['overlap_rows']) + overlap_delta,
        }
    working_hash = _version_canonical_hash(connection, version_id)
    now = _now()
    existing = connection.execute(
        "SELECT version_id FROM review_version_state WHERE version_id = ?",
        (version_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            """INSERT INTO review_version_state
               (version_id, original_frozen, original_canonical_hash,
                original_rows, original_trajectories, original_division_points,
                original_excluded_points, original_overlap_rows,
                original_legacy_csv_path, original_legacy_csv_sha256,
                working_hash, dirty, invalidated_at, revision)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                version_id,
                int(row["frozen"]),
                str(row["canonical_hash"]),
                int(row["rows"]),
                int(row["trajectories"]),
                int(row["division_points"]),
                int(row["excluded_points"]),
                int(row["overlap_rows"]),
                row["legacy_csv_path"],
                row["legacy_csv_sha256"],
                working_hash,
                now,
                revision_number,
            ),
        )
    else:
        connection.execute(
            """UPDATE review_version_state
               SET working_hash = ?, dirty = 1, invalidated_at = ?, revision = ?
               WHERE version_id = ?""",
            (working_hash, now, revision_number, version_id),
        )
    connection.execute(
        """UPDATE dataset_version
           SET canonical_hash = ?, rows = ?, trajectories = ?,
               division_points = ?, excluded_points = ?, overlap_rows = ?,
               frozen = 0, legacy_csv_path = NULL, legacy_csv_sha256 = NULL
           WHERE version_id = ?""",
        (
            working_hash,
            counts["rows"],
            counts["trajectories"],
            counts["division_points"],
            counts["excluded_points"],
            counts["overlap_rows"],
            version_id,
        ),
    )
    _renumber_ordinals(connection, version_id)
    return {
        "lineage_id": dataset_store.version_lineage_id(
            str(row["dataset_id"]), int(row["version"])
        ),
        "dataset_id": str(row["dataset_id"]),
        "version_id": version_id,
        "dirty": True,
        "frozen": False,
        "rows": counts["rows"],
        "trajectories": counts["trajectories"],
        "division_points": counts["division_points"],
        "excluded_points": counts["excluded_points"],
        "overlap_rows": counts["overlap_rows"],
        "canonical_hash": working_hash,
    }


def _dirty_versions_for_trajectory(
    connection: sqlite3.Connection, trajectory_id: int, revision_number: int,
    counts_delta: tuple[int, int, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    return [
        _mark_version_dirty(connection, version_id, revision_number, counts_delta)
        for version_id in _membership_version_ids(connection, trajectory_id)
    ]


def _recompute_trajectory_hash(
    connection: sqlite3.Connection, trajectory_id: int
) -> str:
    rows = connection.execute(
        """SELECT fov_cell_index, time_h, spine_length, division_point,
                  excluded_point
           FROM observation WHERE trajectory_id = ? ORDER BY fov_cell_index""",
        (trajectory_id,),
    ).fetchall()
    observations = [
        (int(row[0]), float(row[1]), float(row[2]), bool(row[3]), bool(row[4]))
        for row in rows
    ]
    content_hash = dataset_store.trajectory_content_hash(observations)
    connection.execute(
        "UPDATE trajectory SET content_hash = ? WHERE trajectory_id = ?",
        (content_hash, trajectory_id),
    )
    return content_hash


def _current_labels(
    connection: sqlite3.Connection, trajectory_id: int
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT fov_cell_index, division_point, excluded_point
           FROM observation WHERE trajectory_id = ? ORDER BY fov_cell_index""",
        (trajectory_id,),
    ).fetchall()
    return [
        {
            "fov_cell_index": int(row["fov_cell_index"]),
            "division_point": bool(row["division_point"]),
            "excluded_point": bool(row["excluded_point"]),
        }
        for row in rows
    ]


_INT64_MAX = 2 ** 63 - 1


def _coerce_revision(value: Any, name: str = "expected_revision") -> int:
    """Strict global revision argument: a non-bool integer (no floats/strings)."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ReviewError(f"{name} must be an integer (not {type(value).__name__})")
    number = int(value)
    if number < 0:
        raise ReviewError(f"{name} must be non-negative")
    return number


def _coerce_fov_values(series: pd.Series) -> list[int]:
    """Strict fov_cell_index parsing.

    Accepts only non-boolean integers or canonical decimal strings and rejects
    floats, NaN/NA, booleans and out-of-int64-range values so a UI cannot round
    or coerce an identifier (e.g. a float beyond 2**53) into another row's id.
    """

    values: list[int] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise InvalidLabels("fov_cell_index must be a non-boolean integer")
        if isinstance(value, (int, np.integer)):
            number = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None:
            number = int(value)
        else:
            raise InvalidLabels(
                "fov_cell_index must be a non-boolean integer or decimal string; "
                f"invalid value: {value!r}"
            )
        if number < 0 or number > _INT64_MAX:
            raise InvalidLabels(
                f"fov_cell_index out of range: {number!r}"
            )
        values.append(number)
    return values


def _coerce_bool_series(series: pd.Series, name: str) -> pd.Series:
    """Strict boolean validation: only bool / numpy.bool_ values are accepted."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values: list[bool] = []
    bad: list[Any] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        else:
            bad.append(value)
    if bad:
        raise InvalidLabels(
            f"{name} must be strict boolean (True/False); invalid values: {bad[:5]!r}"
        )
    return pd.Series(values, index=series.index, dtype=bool)


def _coerce_bool_scalar(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ReviewError(f"{name} must be strict boolean (True/False)")


def _validate_labels(
    connection: sqlite3.Connection,
    trajectory_key: str,
    trajectory_id: int,
    labels: pd.DataFrame,
) -> dict[int, tuple[bool, bool]]:
    if not isinstance(labels, pd.DataFrame):
        raise InvalidLabels("labels must be a pandas DataFrame")
    required = {"fov_cell_index", "division_point", "excluded_point"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise InvalidLabels(f"labels is missing required columns: {missing}")
    fovs = _coerce_fov_values(labels["fov_cell_index"])
    if len(set(fovs)) != len(fovs):
        raise InvalidLabels("labels contains duplicate fov_cell_index values")
    current = {
        int(row["fov_cell_index"]): row
        for row in connection.execute(
            """SELECT fov_cell_index, time_h, spine_length
               FROM observation WHERE trajectory_id = ?""",
            (trajectory_id,),
        ).fetchall()
    }
    if set(fovs) != set(current):
        extra = sorted(set(fovs) - set(current))
        missing_rows = sorted(set(current) - set(fovs))
        raise InvalidLabels(
            "labels must cover exactly the trajectory's existing observations; "
            f"extra={extra[:5]!r}, missing={missing_rows[:5]!r}"
        )
    if "trajectory_key" in labels.columns:
        keys = {str(value) for value in labels["trajectory_key"].tolist()}
        if keys != {trajectory_key}:
            raise InvalidLabels(
                "labels.trajectory_key does not match the requested trajectory"
            )
    divisions = _coerce_bool_series(labels["division_point"], "division_point")
    excluded = _coerce_bool_series(labels["excluded_point"], "excluded_point")
    if "time_h" in labels.columns:
        times = pd.to_numeric(labels["time_h"], errors="coerce")
        if times.isna().any():
            raise InvalidLabels("time_h contains missing/non-numeric values")
        for fov, value in zip(fovs, times.tolist()):
            if not math.isfinite(float(value)) or float(value) != float(
                current[fov]["time_h"]
            ):
                raise InvalidLabels(
                    "measurements are immutable: time_h cannot be changed"
                )
    if "spine_length" in labels.columns:
        lengths = pd.to_numeric(labels["spine_length"], errors="coerce")
        if lengths.isna().any():
            raise InvalidLabels("spine_length contains missing/non-numeric values")
        for fov, value in zip(fovs, lengths.tolist()):
            if not math.isfinite(float(value)) or float(value) != float(
                current[fov]["spine_length"]
            ):
                raise InvalidLabels(
                    "measurements are immutable: spine_length cannot be changed"
                )
    result: dict[int, tuple[bool, bool]] = {}
    for position, fov in enumerate(fovs):
        result[fov] = (bool(divisions.iloc[position]), bool(excluded.iloc[position]))
    return result


def _upsert_review_state(
    connection: sqlite3.Connection,
    trajectory_key: str,
    revision_number: int,
    actor: str | None,
    reviewed: bool | None = None,
) -> None:
    now = _now()
    existing = connection.execute(
        "SELECT trajectory_key FROM review_trajectory WHERE trajectory_key = ?",
        (trajectory_key,),
    ).fetchone()
    if existing is None:
        connection.execute(
            """INSERT INTO review_trajectory
               (trajectory_key, reviewed, revision, last_actor, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (trajectory_key, int(bool(reviewed)), revision_number, actor, now),
        )
    elif reviewed is None:
        connection.execute(
            """UPDATE review_trajectory
               SET revision = ?, last_actor = ?, updated_at = ?
               WHERE trajectory_key = ?""",
            (revision_number, actor, now, trajectory_key),
        )
    else:
        connection.execute(
            """UPDATE review_trajectory
               SET reviewed = ?, revision = ?, last_actor = ?, updated_at = ?
               WHERE trajectory_key = ?""",
            (int(bool(reviewed)), revision_number, actor, now, trajectory_key),
        )


def _insert_payload(
    connection: sqlite3.Connection,
    trajectory_key: str,
    revision_number: int,
    kind: str,
    payload: Any,
) -> str:
    payload_id = uuid.uuid4().hex
    connection.execute(
        """INSERT INTO review_payload
           (payload_id, trajectory_key, revision, kind, payload_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            payload_id,
            trajectory_key,
            revision_number,
            kind,
            json.dumps(payload, sort_keys=True),
            _now(),
        ),
    )
    return payload_id


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def save_labels(
    db: str | Path,
    trajectory_key: str,
    labels: pd.DataFrame,
    expected_revision: int,
    actor: str = "user",
) -> dict[str, Any]:
    """Replace the division/exclusion labels of one trajectory (global edit)."""

    def mutate(connection: sqlite3.Connection, new_revision: int):
        trajectory = _trajectory_row(connection, trajectory_key)
        trajectory_id = int(trajectory["trajectory_id"])
        previous = _current_labels(connection, trajectory_id)
        new_labels = _validate_labels(
            connection, trajectory_key, trajectory_id, labels
        )
        connection.executemany(
            """UPDATE observation
               SET division_point = ?, excluded_point = ?
               WHERE trajectory_id = ? AND fov_cell_index = ?""",
            [
                (int(division), int(excluded), trajectory_id, fov)
                for fov, (division, excluded) in new_labels.items()
            ],
        )
        content_hash = _recompute_trajectory_hash(connection, trajectory_id)
        _upsert_review_state(
            connection, trajectory_key, new_revision, actor, reviewed=None
        )
        _insert_payload(
            connection, trajectory_key, new_revision, "label_undo", previous
        )
        division_points = sum(1 for value in new_labels.values() if value[0])
        excluded_points = sum(1 for value in new_labels.values() if value[1])
        overlap_rows = sum(
            1 for value in new_labels.values() if value[0] and value[1]
        )
        old_division = sum(1 for row in previous if row["division_point"])
        old_excluded = sum(1 for row in previous if row["excluded_point"])
        old_overlap = sum(
            1 for row in previous if row["division_point"] and row["excluded_point"]
        )
        versions = _dirty_versions_for_trajectory(
            connection, trajectory_id, new_revision,
            (0, 0, division_points - old_division,
             excluded_points - old_excluded, overlap_rows - old_overlap),
        )
        result = {
            "trajectory_key": trajectory_key,
            "content_hash": content_hash,
            "rows": len(new_labels),
            "division_points": division_points,
            "excluded_points": excluded_points,
            "overlap_rows": overlap_rows,
            "versions": versions,
        }
        event = {
            "trajectory_key": trajectory_key,
            "event_payload": {
                "rows": len(new_labels),
                "content_hash": content_hash,
                "division_points": division_points,
                "excluded_points": excluded_points,
                "versions": [version["lineage_id"] for version in versions],
            },
        }
        return event, result

    return _run_mutation(
        db, expected_revision, "save_labels", actor, mutate, backup=True
    )


def mark_reviewed(
    db: str | Path,
    trajectory_key: str,
    reviewed: bool,
    expected_revision: int,
    actor: str = "user",
) -> dict[str, Any]:
    """Set the reviewed flag of one trajectory (no dirtying)."""

    reviewed_flag = _coerce_bool_scalar(reviewed, "reviewed")

    def mutate(connection: sqlite3.Connection, new_revision: int):
        _trajectory_row(connection, trajectory_key)
        _upsert_review_state(
            connection, trajectory_key, new_revision, actor, reviewed=reviewed_flag
        )
        result = {"trajectory_key": trajectory_key, "reviewed": reviewed_flag}
        event = {
            "trajectory_key": trajectory_key,
            "event_payload": {"reviewed": reviewed_flag},
        }
        return event, result

    return _run_mutation(
        db, expected_revision, "mark_reviewed", actor, mutate, backup=False
    )


def undo_labels(
    db: str | Path,
    trajectory_key: str,
    expected_revision: int,
    actor: str = "user",
) -> dict[str, Any]:
    """Restore the latest saved label set for a trajectory (labels only)."""

    def mutate(connection: sqlite3.Connection, new_revision: int):
        trajectory = _trajectory_row(connection, trajectory_key)
        trajectory_id = int(trajectory["trajectory_id"])
        payload = connection.execute(
            """SELECT payload_id, payload_json FROM review_payload
               WHERE trajectory_key = ? AND kind = 'label_undo'
               ORDER BY rowid DESC LIMIT 1""",
            (trajectory_key,),
        ).fetchone()
        if payload is None:
            raise ReviewError(f"nothing to undo for trajectory {trajectory_key!r}")
        rows = json.loads(str(payload["payload_json"]))
        existing = {
            int(row[0])
            for row in connection.execute(
                "SELECT fov_cell_index FROM observation WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchall()
        }
        payload_fovs = {int(row["fov_cell_index"]) for row in rows}
        if payload_fovs != existing:
            raise ReviewError(
                "undo payload no longer matches the stored observations; "
                "refusing to restore partial labels"
            )
        before = _current_labels(connection, trajectory_id)
        old_division = sum(1 for row in before if row["division_point"])
        old_excluded = sum(1 for row in before if row["excluded_point"])
        old_overlap = sum(
            1 for row in before if row["division_point"] and row["excluded_point"]
        )
        connection.executemany(
            """UPDATE observation
               SET division_point = ?, excluded_point = ?
               WHERE trajectory_id = ? AND fov_cell_index = ?""",
            [
                (
                    int(bool(row["division_point"])),
                    int(bool(row["excluded_point"])),
                    trajectory_id,
                    int(row["fov_cell_index"]),
                )
                for row in rows
            ],
        )
        connection.execute(
            "DELETE FROM review_payload WHERE payload_id = ?",
            (str(payload["payload_id"]),),
        )
        content_hash = _recompute_trajectory_hash(connection, trajectory_id)
        result_division = sum(1 for row in rows if row["division_point"])
        result_excluded = sum(1 for row in rows if row["excluded_point"])
        result_overlap = sum(
            1 for row in rows if row["division_point"] and row["excluded_point"]
        )
        versions = _dirty_versions_for_trajectory(
            connection, trajectory_id, new_revision,
            (0, 0, result_division - old_division,
             result_excluded - old_excluded, result_overlap - old_overlap),
        )
        _upsert_review_state(
            connection, trajectory_key, new_revision, actor, reviewed=None
        )
        result = {
            "trajectory_key": trajectory_key,
            "content_hash": content_hash,
            "rows": len(rows),
            "division_points": sum(
                1 for row in rows if bool(row["division_point"])
            ),
            "excluded_points": sum(
                1 for row in rows if bool(row["excluded_point"])
            ),
            "versions": versions,
        }
        event = {
            "trajectory_key": trajectory_key,
            "event_payload": {
                "rows": len(rows),
                "content_hash": content_hash,
                "versions": [version["lineage_id"] for version in versions],
            },
        }
        return event, result

    return _run_mutation(
        db, expected_revision, "undo_labels", actor, mutate, backup=True
    )


def delete_trajectory(
    db: str | Path,
    trajectory_key: str,
    expected_revision: int,
    confirmation_key: str,
    actor: str = "user",
) -> dict[str, Any]:
    """Permanently delete one trajectory from every dataset version."""

    if not isinstance(confirmation_key, str) or confirmation_key != trajectory_key:
        raise ConfirmationError(
            "confirmation_key must exactly equal the trajectory_key "
            "(permanent delete has no restore path)"
        )

    def mutate(connection: sqlite3.Connection, new_revision: int):
        trajectory = _trajectory_row(connection, trajectory_key)
        trajectory_id = int(trajectory["trajectory_id"])
        version_ids = _membership_version_ids(connection, trajectory_id)
        observation_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM observation WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()[0]
        )
        undo_payloads = int(
            connection.execute(
                "SELECT COUNT(*) FROM review_payload WHERE trajectory_key = ?",
                (trajectory_key,),
            ).fetchone()[0]
        )
        connection.execute(
            "DELETE FROM dataset_member WHERE trajectory_id = ?", (trajectory_id,)
        )
        connection.execute(
            "DELETE FROM observation WHERE trajectory_id = ?", (trajectory_id,)
        )
        connection.execute(
            "DELETE FROM trajectory WHERE trajectory_id = ?", (trajectory_id,)
        )
        connection.execute(
            "DELETE FROM review_payload WHERE trajectory_key = ?", (trajectory_key,)
        )
        connection.execute(
            "DELETE FROM review_trajectory WHERE trajectory_key = ?",
            (trajectory_key,),
        )
        versions = [
            _mark_version_dirty(connection, version_id, new_revision)
            for version_id in version_ids
        ]
        result = {
            "trajectory_key": trajectory_key,
            "deleted": {
                "observations": observation_rows,
                "memberships": len(version_ids),
                "trajectories": 1,
                "undo_payloads": undo_payloads,
            },
            "versions": versions,
        }
        event = {
            "trajectory_key": trajectory_key,
            "event_payload": {
                "observations": observation_rows,
                "memberships": len(version_ids),
                "versions": [version["lineage_id"] for version in versions],
                "undo_payloads_purged": undo_payloads,
            },
        }
        return event, result

    return _run_mutation(
        db, expected_revision, "delete_trajectory", actor, mutate, backup=True
    )


def finalize(
    db: str | Path,
    lineage_id: str,
    dataset_id: str,
    expected_revision: int,
    verification_basis: str = "user_reviewed",
    apply_r004: bool = True,
    actor: str = "user",
) -> dict[str, Any]:
    """Create a new frozen dataset version from a working (possibly dirty) one."""

    if not isinstance(dataset_id, str) or not DATASET_ID_PATTERN.match(dataset_id):
        raise InvalidDataset(
            "dataset_id must match ^[A-Za-z0-9_.-]+$ (letters, digits, _ . -)"
        )
    if not isinstance(verification_basis, str) or not verification_basis.strip():
        raise InvalidDataset("verification_basis must be a non-empty string")
    if not isinstance(apply_r004, (bool, np.bool_)):
        raise InvalidDataset("apply_r004 must be a boolean")
    apply_r004 = bool(apply_r004)

    def mutate(connection: sqlite3.Connection, new_revision: int):
        source = _version_row(connection, lineage_id)
        source_version_id = int(source["version_id"])
        members = connection.execute(
            """SELECT m.trajectory_id, t.trajectory_key, t.content_hash
               FROM dataset_member m
               JOIN trajectory t ON t.trajectory_id = m.trajectory_id
               WHERE m.version_id = ? ORDER BY m.ordinal""",
            (source_version_id,),
        ).fetchall()
        if not members:
            raise ReviewError(f"cannot finalize {lineage_id!r}: the version is empty")
        keep: list[sqlite3.Row] = []
        dropped: list[str] = []
        for member in members:
            if (
                apply_r004
                and _longest_exclusion_run(connection, int(member["trajectory_id"]))
                >= R004_MAX_EXCLUSION_RUN
            ):
                dropped.append(str(member["trajectory_key"]))
            else:
                keep.append(member)
        if not keep:
            raise ReviewError(
                "cannot finalize: the R004 filter would drop every trajectory "
                "in this version (pass apply_r004=False to keep them explicitly)"
            )
        if (
            connection.execute(
                "SELECT 1 FROM dataset WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            is not None
        ):
            raise InvalidDataset(f"dataset_id {dataset_id!r} already exists")
        version_number = 1
        canonical = hashlib.sha256()
        counts = {"rows": 0, "division_points": 0, "excluded_points": 0,
                  "overlap_rows": 0}
        recomputed_hashes: dict[int, str] = {}
        for member in keep:
            trajectory_id = int(member["trajectory_id"])
            observations = connection.execute(
                """SELECT fov_cell_index, time_h, spine_length, division_point,
                          excluded_point
                   FROM observation WHERE trajectory_id = ?
                   ORDER BY fov_cell_index""",
                (trajectory_id,),
            ).fetchall()
            if not observations:
                raise ReviewError(
                    "cannot finalize: trajectory "
                    f"{member['trajectory_key']!r} has no observations"
                )
            validated = []
            for observation in observations:
                time_h = float(observation["time_h"])
                spine = float(observation["spine_length"])
                division = int(observation["division_point"])
                excluded = int(observation["excluded_point"])
                if not math.isfinite(time_h):
                    raise ReviewError("cannot finalize: non-finite time_h")
                if not math.isfinite(spine) or spine <= 0:
                    raise ReviewError(
                        "cannot finalize: non-finite or non-positive spine_length"
                    )
                if division not in (0, 1) or excluded not in (0, 1):
                    raise ReviewError(
                        "cannot finalize: non-boolean label in the database"
                    )
                validated.append(
                    (
                        int(observation["fov_cell_index"]),
                        time_h,
                        spine,
                        bool(division),
                        bool(excluded),
                    )
                )
                counts["division_points"] += division
                counts["excluded_points"] += excluded
                counts["overlap_rows"] += 1 if division and excluded else 0
            recomputed = dataset_store.trajectory_content_hash(validated)
            if recomputed != str(member["content_hash"]):
                raise ReviewError(
                    "cannot finalize: stored content_hash does not match the "
                    f"observations for trajectory {member['trajectory_key']!r}"
                )
            recomputed_hashes[trajectory_id] = recomputed
            counts["rows"] += len(validated)
        for member in sorted(keep, key=lambda item: str(item["trajectory_key"])):
            trajectory_id = int(member["trajectory_id"])
            canonical.update(
                f"{member['trajectory_key']}\t{recomputed_hashes[trajectory_id]}\n".encode(
                    "utf-8"
                )
            )
        canonical_hash = canonical.hexdigest()
        new_lineage = dataset_store.version_lineage_id(dataset_id, version_number)
        connection.execute(
            """INSERT INTO dataset
               (dataset_id, kind, label, round_name, verification_basis,
                human_verified, notes)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (
                dataset_id,
                "review_finalized",
                f"review-finalized from {lineage_id}",
                None,
                verification_basis,
                (
                    f"created by review backend from {lineage_id} at global "
                    f"revision {new_revision}; apply_r004={apply_r004}; "
                    f"dropped_keys={dropped}"
                ),
            ),
        )
        cursor = connection.execute(
            """INSERT INTO dataset_version
               (dataset_id, version, canonical_hash, rows, trajectories,
                division_points, excluded_points, overlap_rows,
                legacy_csv_path, legacy_csv_sha256, frozen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 1, ?)""",
            (
                dataset_id,
                version_number,
                canonical_hash,
                counts["rows"],
                len(keep),
                counts["division_points"],
                counts["excluded_points"],
                counts["overlap_rows"],
                _now(),
            ),
        )
        new_version_id = int(cursor.lastrowid)
        connection.executemany(
            """INSERT INTO dataset_member (version_id, trajectory_id, ordinal)
               VALUES (?, ?, ?)""",
            [
                (new_version_id, int(member["trajectory_id"]), ordinal)
                for ordinal, member in enumerate(keep, start=1)
            ],
        )
        with tempfile.TemporaryDirectory(prefix="review_finalize_export_") as temp_dir:
            export_result = dataset_store.export_version_csv(
                connection,
                new_version_id,
                Path(temp_dir) / f"{dataset_id}_v{version_number}.csv",
            )
        legacy_csv_sha256 = str(export_result["sha256"])
        connection.execute(
            "UPDATE dataset_version SET legacy_csv_sha256 = ? WHERE version_id = ?",
            (legacy_csv_sha256, new_version_id),
        )
        connection.execute(
            """INSERT INTO ingest_provenance (version_id, role, path, sha256)
               VALUES (?, ?, ?, ?)""",
            (
                new_version_id,
                "source_lineage",
                lineage_id,
                str(source["canonical_hash"]),
            ),
        )
        result = {
            "lineage_id": new_lineage,
            "content_ref": dataset_store.version_content_ref(
                dataset_id, version_number, canonical_hash
            ),
            "dataset_id": dataset_id,
            "version": version_number,
            "version_id": new_version_id,
            "source_lineage_id": lineage_id,
            "rows": counts["rows"],
            "trajectories": len(keep),
            "division_points": counts["division_points"],
            "excluded_points": counts["excluded_points"],
            "overlap_rows": counts["overlap_rows"],
            "dropped_keys": dropped,
            "canonical_hash": canonical_hash,
            "legacy_csv_path": None,
            "legacy_csv_sha256": legacy_csv_sha256,
            "frozen": True,
            "human_verified": True,
        }
        event = {
            "dataset_id": dataset_id,
            "lineage_id": new_lineage,
            "event_payload": {
                "version_id": new_version_id,
                "source_lineage": lineage_id,
                "rows": counts["rows"],
                "trajectories": len(keep),
                "dropped_keys": dropped,
                "apply_r004": apply_r004,
                "canonical_hash": canonical_hash,
                "legacy_csv_path": None,
                "legacy_csv_sha256": legacy_csv_sha256,
            },
        }
        return event, result

    return _run_mutation(
        db, expected_revision, "finalize", actor, mutate, backup=False
    )


def remove_dataset_versions(
    db: str | Path,
    lineage_ids: Sequence[str],
    expected_revision: int,
    *,
    reason: str,
    actor: str = "user",
) -> dict[str, Any]:
    """Permanently remove whole duplicate dataset versions from the store.

    This is the dataset-level counterpart of a per-trajectory delete: a version
    is removable only when every one of its trajectories is still referenced by
    another version, so no observation loses all dataset memberships.  The
    removal takes an online backup, bumps the global review revision and records
    a review event.  The core store schema is unchanged.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise InvalidDataset("reason must be a non-empty string")
    if isinstance(lineage_ids, str) or not isinstance(lineage_ids, Sequence):
        raise InvalidDataset("lineage_ids must be a sequence of lineage ids")
    requested = [str(item).strip() for item in lineage_ids]
    if not requested:
        raise InvalidDataset("at least one lineage id is required")
    if any(not item for item in requested):
        raise InvalidDataset("lineage ids must be non-empty")
    if len(set(requested)) != len(requested):
        raise InvalidDataset("lineage ids must be unique")

    def mutate(connection: sqlite3.Connection, new_revision: int):
        rows = [(lineage_id, _version_row(connection, lineage_id)) for lineage_id in requested]
        version_ids = [int(row["version_id"]) for _, row in rows]
        if len(set(version_ids)) != len(version_ids):
            raise InvalidDataset("lineage ids resolve to duplicate dataset versions")
        placeholders = ",".join("?" for _ in version_ids)
        parameters = tuple(version_ids)
        orphans = connection.execute(
            f"""SELECT DISTINCT t.trajectory_key
                FROM dataset_member m
                JOIN trajectory t ON t.trajectory_id = m.trajectory_id
                WHERE m.version_id IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM dataset_member other
                      WHERE other.trajectory_id = m.trajectory_id
                        AND other.version_id NOT IN ({placeholders})
                  )
                ORDER BY t.trajectory_key""",
            parameters + parameters,
        ).fetchall()
        if orphans:
            keys = [str(row["trajectory_key"]) for row in orphans]
            preview = ", ".join(keys[:5])
            raise ReviewError(
                f"cannot remove {len(version_ids)} dataset version(s): "
                f"{len(keys)} trajectory/trajectories would lose every dataset "
                f"membership ({preview}); remove only duplicate versions"
            )
        members_removed = int(
            connection.execute(
                "SELECT COUNT(*) FROM dataset_member"
                f" WHERE version_id IN ({placeholders})",
                parameters,
            ).fetchone()[0]
        )
        for statement in (
            "DELETE FROM review_version_state"
            f" WHERE version_id IN ({placeholders})",
            "DELETE FROM ingest_provenance"
            f" WHERE version_id IN ({placeholders})",
            "DELETE FROM dataset_member"
            f" WHERE version_id IN ({placeholders})",
            "DELETE FROM dataset_version"
            f" WHERE version_id IN ({placeholders})",
        ):
            connection.execute(statement, parameters)
        removed_datasets: list[str] = []
        for dataset_id in sorted({str(row["dataset_id"]) for _, row in rows}):
            remaining = int(
                connection.execute(
                    "SELECT COUNT(*) FROM dataset_version WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchone()[0]
            )
            if remaining == 0:
                connection.execute(
                    "DELETE FROM dataset WHERE dataset_id = ?", (dataset_id,)
                )
                removed_datasets.append(dataset_id)
        store_totals = {
            "versions": int(
                connection.execute("SELECT COUNT(*) FROM dataset_version").fetchone()[0]
            ),
            "trajectories": int(
                connection.execute("SELECT COUNT(*) FROM trajectory").fetchone()[0]
            ),
            "observations": int(
                connection.execute("SELECT COUNT(*) FROM observation").fetchone()[0]
            ),
        }
        removed = [
            {
                "lineage_id": lineage_id,
                "version_id": int(row["version_id"]),
                "dataset_id": str(row["dataset_id"]),
                "version": int(row["version"]),
                "rows": int(row["rows"]),
                "trajectories": int(row["trajectories"]),
            }
            for lineage_id, row in rows
        ]
        payload = {
            "removed": removed,
            "removed_datasets": removed_datasets,
            "members_removed": members_removed,
            "reason": reason,
            "store_totals": store_totals,
        }
        return {"event_payload": payload}, dict(payload)

    return _run_mutation(
        db,
        expected_revision,
        "remove_dataset_versions",
        actor,
        mutate,
        backup=True,
    )
