"""Local SQLite store for the frozen verified trajectory corpora (phase 1).

Design (see docs/PLAN_DATASET_STORE.md):

- observations are stored once per (trajectory_key, fov_cell_index);
- a corpus is a frozen dataset_version whose membership is a set of
  trajectories (dataset_member keeps an explicit ordinal so exports can
  reproduce the historical CSV row order byte-for-byte);
- each version keeps both a canonical content hash (computed from the stored
  rows) and the legacy CSV sha256 (the existing integrity pins), so the
  transition does not invalidate datasets/manifest.json;
- only dataset data lives here: training runs keep their own small metadata
  and reference the dataset lineage id (dataset_id@version + content hash).

Phase 1 scope: build + verify the store. The training path is untouched and
the verified_* CSV files are not deleted yet.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

SCHEMA_VERSION = 2

OUTPUT_COLUMNS = [
    "trajectory_key",
    "fov_cell_index",
    "time_h",
    "spine_length",
    "division_point",
    "excluded_point",
    "source_dataset",
    "source_file",
    "source_file_sha256",
    "source_trajectory_key",
    "source_fov_cell_index",
]

PROVENANCE_ROLES = (
    "dataset_manifest",
    "merge_manifest",
    "staging_report",
    "rename_map",
    "qc_report",
    "longrun_excluded_trajectories",
    "splits_no_longruns",
)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_file (
    source_file_id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    corpus TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    rows INTEGER NOT NULL,
    imported_at TEXT NOT NULL,
    UNIQUE (filename, corpus)
);

CREATE TABLE IF NOT EXISTS trajectory (
    trajectory_id INTEGER PRIMARY KEY,
    trajectory_key TEXT NOT NULL UNIQUE,
    source_dataset TEXT NOT NULL,
    source_trajectory_key TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_file_sha256 TEXT NOT NULL,
    rename_of TEXT,
    label_policy TEXT NOT NULL DEFAULT 'effective',
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observation (
    trajectory_id INTEGER NOT NULL REFERENCES trajectory(trajectory_id),
    fov_cell_index INTEGER NOT NULL,
    time_h REAL NOT NULL,
    spine_length REAL NOT NULL,
    division_point INTEGER NOT NULL CHECK (division_point IN (0, 1)),
    excluded_point INTEGER NOT NULL CHECK (excluded_point IN (0, 1)),
    PRIMARY KEY (trajectory_id, fov_cell_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS dataset (
    dataset_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    round_name TEXT,
    verification_basis TEXT,
    human_verified INTEGER NOT NULL CHECK (human_verified IN (0, 1)),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS dataset_version (
    version_id INTEGER PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES dataset(dataset_id),
    version INTEGER NOT NULL,
    canonical_hash TEXT NOT NULL,
    rows INTEGER NOT NULL,
    trajectories INTEGER NOT NULL,
    division_points INTEGER NOT NULL,
    excluded_points INTEGER NOT NULL,
    overlap_rows INTEGER NOT NULL,
    legacy_csv_path TEXT,
    legacy_csv_sha256 TEXT,
    frozen INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_id, version)
);

CREATE TABLE IF NOT EXISTS dataset_member (
    version_id INTEGER NOT NULL REFERENCES dataset_version(version_id),
    trajectory_id INTEGER NOT NULL REFERENCES trajectory(trajectory_id),
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (version_id, trajectory_id),
    UNIQUE (version_id, ordinal)
);

CREATE TABLE IF NOT EXISTS ingest_provenance (
    version_id INTEGER NOT NULL REFERENCES dataset_version(version_id),
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    PRIMARY KEY (version_id, path)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_float(value: float) -> str:
    return repr(float(value))


def trajectory_content_hash(
    observations: Iterable[tuple[int, float, float, bool, bool]]
) -> str:
    digest = hashlib.sha256()
    for fov, time_h, spine, division, excluded in sorted(observations):
        line = "\t".join(
            (
                str(int(fov)),
                _fmt_float(time_h),
                _fmt_float(spine),
                "1" if division else "0",
                "1" if excluded else "0",
            )
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def version_lineage_id(dataset_id: str, version: int) -> str:
    return f"{dataset_id}@v{version}"


def version_content_ref(dataset_id: str, version: int, canonical_hash: str) -> str:
    return f"{version_lineage_id(dataset_id, version)}#{canonical_hash[:12]}"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    applied = connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    if not applied:
        connection.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utc_now()),
        )
    current = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    if current != SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {current} != code version {SCHEMA_VERSION}"
        )
    connection.commit()


def _validate_corpus_frame(frame: pd.DataFrame, csv_path: Path) -> None:
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    if frame.empty:
        raise ValueError(f"{csv_path}: empty corpus")
    for column in ("trajectory_key", "fov_cell_index"):
        if frame[column].isna().any():
            raise ValueError(f"{csv_path}: {column} contains missing values")
    duplicated = int(
        frame.duplicated(subset=["trajectory_key", "fov_cell_index"]).sum()
    )
    if duplicated:
        raise ValueError(f"{csv_path}: {duplicated} duplicated observation rows")
    for column in ("time_h", "spine_length"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{csv_path}: {column} contains missing values")
    if not (pd.to_numeric(frame["spine_length"], errors="coerce") > 0).all():
        raise ValueError(f"{csv_path}: non-positive spine_length")


def _rename_lookup(repo_root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    datasets_dir = repo_root / "datasets"
    for path in sorted(datasets_dir.glob("verified_*/rename_map.json")):
        mapping = json.loads(path.read_text(encoding="utf-8"))
        for old_key, new_key in mapping.items():
            lookup[str(new_key)] = str(old_key)
    return lookup


def _provenance_files(corpus_dir: Path, repo_root: Path) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for role in PROVENANCE_ROLES:
        path = corpus_dir / f"{role}.json"
        if path.is_file():
            records.append((role, str(path.relative_to(repo_root)), sha256_file(path)))
    return records


def _trajectory_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trajectory_key",
        "source_dataset",
        "source_trajectory_key",
        "source_file",
        "source_file_sha256",
    ]
    summary = frame.groupby("trajectory_key", sort=False)[columns[1:]].nunique()
    bad = summary[(summary > 1).any(axis=1)]
    if not bad.empty:
        raise ValueError(
            f"inconsistent per-trajectory provenance: {bad.index.tolist()[:3]}"
        )
    first = (
        frame.groupby("trajectory_key", sort=False)[columns[1:]]
        .first()
        .reset_index()
    )
    return first[columns]


def _next_version(connection: sqlite3.Connection, dataset_id: str) -> int:
    row = connection.execute(
        "SELECT MAX(version) FROM dataset_version WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()[0]
    return int(row or 0) + 1


def assert_import_allowed(connection: sqlite3.Connection) -> None:
    """Fence historical CSV ingestion after explicit training-store curation."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_version_state'"
    ).fetchone()
    if exists and connection.execute("SELECT 1 FROM review_version_state LIMIT 1").fetchone():
        raise ValueError(
            "This store has reviewed working data. Historical CSV build/import is disabled "
            "to prevent restoring permanently deleted trajectories; use the training-review "
            "app to finalize current data, or build a separate historical store."
        )


def import_corpus(
    connection: sqlite3.Connection,
    *,
    repo_root: Path,
    csv_path: Path,
    dataset_id: str,
    kind: str,
    label: str,
    round_name: str | None,
    verification_basis: str | None,
    human_verified: bool,
    notes: str | None = None,
) -> dict[str, Any]:
    """Import one frozen corpus CSV as a new (or existing) dataset version."""

    # An explicitly curated working store must not silently resurrect deleted
    # trajectories from historical corpus CSVs during a maintenance rebuild.
    assert_import_allowed(connection)
    csv_path = csv_path.resolve()
    legacy_sha = sha256_file(csv_path)
    # Keep provenance/identifier columns as text: hashes and keys that look
    # numeric (e.g. an all-zero sha placeholder) must not be coerced to ints.
    frame = pd.read_csv(
        csv_path,
        dtype={
            "trajectory_key": str,
            "source_dataset": str,
            "source_file": str,
            "source_file_sha256": str,
            "source_trajectory_key": str,
        },
    )
    _validate_corpus_frame(frame, csv_path)

    existing = connection.execute(
        "SELECT version_id, version FROM dataset_version WHERE legacy_csv_sha256 = ?",
        (legacy_sha,),
    ).fetchone()
    if existing is not None:
        return {
            "dataset_id": dataset_id,
            "version": int(existing["version"]),
            "version_id": int(existing["version_id"]),
            "legacy_csv_sha256": legacy_sha,
            "status": "already_imported",
        }

    provenance = _trajectory_provenance(frame)
    rename_lookup = _rename_lookup(repo_root)

    ordinal_by_key = {
        str(key): ordinal
        for ordinal, key in enumerate(frame["trajectory_key"].drop_duplicates().tolist())
    }

    # Exports rely on one contiguous block per trajectory in the legacy order.
    for key, indices in frame.groupby("trajectory_key", sort=False).indices.items():
        positions = list(indices)
        if positions and (positions[-1] - positions[0] + 1) != len(positions):
            raise ValueError(f"trajectory {key!r} rows are not contiguous")

    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """INSERT INTO dataset
               (dataset_id, kind, label, round_name, verification_basis,
                human_verified, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(dataset_id) DO UPDATE SET
                 kind = excluded.kind,
                 label = excluded.label,
                 round_name = excluded.round_name,
                 verification_basis = excluded.verification_basis,
                 human_verified = excluded.human_verified,
                 notes = COALESCE(excluded.notes, dataset.notes)""",
            (
                dataset_id,
                kind,
                label,
                round_name,
                verification_basis,
                int(human_verified),
                notes,
            ),
        )

        connection.execute(
            """INSERT OR IGNORE INTO source_file
               (filename, sha256, rows, corpus, imported_at)
               VALUES (?, ?, ?, ?, ?)""",
            (csv_path.name, legacy_sha, int(len(frame)), dataset_id, utc_now()),
        )

        provenance_lookup = provenance.set_index("trajectory_key")
        key_to_id: dict[str, int] = {}
        inserted_trajectories = 0
        inserted_observations = 0

        for key, group in frame.groupby("trajectory_key", sort=False):
            key = str(key)
            observations = [
                (
                    int(row.fov_cell_index),
                    float(row.time_h),
                    float(row.spine_length),
                    bool(row.division_point),
                    bool(row.excluded_point),
                )
                for row in group.itertuples(index=False)
            ]
            content_hash = trajectory_content_hash(observations)
            found = connection.execute(
                "SELECT trajectory_id, content_hash FROM trajectory"
                " WHERE trajectory_key = ?",
                (key,),
            ).fetchone()
            if found is not None:
                if found["content_hash"] != content_hash:
                    raise ValueError(
                        f"trajectory {key!r} already stored with different content;"
                        " labels/features must not change in place"
                    )
                trajectory_id = int(found["trajectory_id"])
            else:
                info = provenance_lookup.loc[key]
                suffix = key.split("::")[-1]
                connection.execute(
                    """INSERT INTO trajectory
                       (trajectory_key, source_dataset, source_trajectory_key,
                        source_file, source_file_sha256, rename_of, label_policy,
                        content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key,
                        str(info["source_dataset"]),
                        str(info["source_trajectory_key"]),
                        str(info["source_file"]),
                        str(info["source_file_sha256"]),
                        rename_lookup.get(suffix),
                        "effective",
                        content_hash,
                    ),
                )
                trajectory_id = int(
                    connection.execute("SELECT last_insert_rowid()").fetchone()[0]
                )
                connection.executemany(
                    """INSERT INTO observation
                       (trajectory_id, fov_cell_index, time_h, spine_length,
                        division_point, excluded_point)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (trajectory_id, fov, time_h, spine, int(division), int(excluded))
                        for fov, time_h, spine, division, excluded in observations
                    ],
                )
                inserted_trajectories += 1
                inserted_observations += len(observations)
            key_to_id[key] = trajectory_id

        version = _next_version(connection, dataset_id)
        canonical = hashlib.sha256()
        for key in sorted(key_to_id):
            row = connection.execute(
                "SELECT content_hash FROM trajectory WHERE trajectory_key = ?",
                (key,),
            ).fetchone()
            canonical.update(f"{key}\t{row[0]}\n".encode("utf-8"))
        canonical_hash = canonical.hexdigest()

        division_points = int(frame["division_point"].astype(bool).sum())
        excluded_points = int(frame["excluded_point"].astype(bool).sum())
        overlap_rows = int(
            (
                frame["division_point"].astype(bool)
                & frame["excluded_point"].astype(bool)
            ).sum()
        )

        cursor = connection.execute(
            """INSERT INTO dataset_version
               (dataset_id, version, canonical_hash, rows, trajectories,
                division_points, excluded_points, overlap_rows,
                legacy_csv_path, legacy_csv_sha256, frozen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                dataset_id,
                version,
                canonical_hash,
                int(len(frame)),
                int(len(key_to_id)),
                division_points,
                excluded_points,
                overlap_rows,
                str(csv_path.relative_to(repo_root)),
                legacy_sha,
                utc_now(),
            ),
        )
        version_id = int(cursor.lastrowid)

        connection.executemany(
            "INSERT INTO dataset_member (version_id, trajectory_id, ordinal)"
            " VALUES (?, ?, ?)",
            [
                (version_id, key_to_id[key], ordinal)
                for key, ordinal in ordinal_by_key.items()
            ],
        )

        for role, path_text, digest in _provenance_files(csv_path.parent, repo_root):
            connection.execute(
                "INSERT INTO ingest_provenance (version_id, role, path, sha256)"
                " VALUES (?, ?, ?, ?)",
                (version_id, role, path_text, digest),
            )

        connection.commit()
    except Exception:
        connection.rollback()
        raise

    return {
        "dataset_id": dataset_id,
        "version": version,
        "version_id": version_id,
        "lineage_id": version_lineage_id(dataset_id, version),
        "content_ref": version_content_ref(dataset_id, version, canonical_hash),
        "rows": int(len(frame)),
        "trajectories": int(len(key_to_id)),
        "division_points": division_points,
        "excluded_points": excluded_points,
        "overlap_rows": overlap_rows,
        "legacy_csv_sha256": legacy_sha,
        "canonical_hash": canonical_hash,
        "inserted_trajectories": inserted_trajectories,
        "inserted_observations": inserted_observations,
        "status": "imported",
    }


def export_version(connection: sqlite3.Connection, version_id: int) -> pd.DataFrame:
    """Materialise a version in its historical CSV row order."""

    rows = connection.execute(
        """SELECT t.trajectory_key, o.fov_cell_index, o.time_h, o.spine_length,
                  o.division_point, o.excluded_point, t.source_dataset,
                  t.source_file, t.source_file_sha256, t.source_trajectory_key,
                  o.fov_cell_index AS source_fov_cell_index
           FROM dataset_member m
           JOIN trajectory t ON t.trajectory_id = m.trajectory_id
           JOIN observation o ON o.trajectory_id = m.trajectory_id
           WHERE m.version_id = ?
           ORDER BY m.ordinal, o.fov_cell_index""",
        (version_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"version {version_id} has no rows")
    frame = pd.DataFrame.from_records(
        [tuple(row) for row in rows], columns=OUTPUT_COLUMNS
    )
    for column in ("fov_cell_index", "source_fov_cell_index"):
        frame[column] = frame[column].astype("int64")
    for column in ("time_h", "spine_length"):
        frame[column] = frame[column].astype("float64")
    for column in ("division_point", "excluded_point"):
        frame[column] = frame[column].astype("bool")
    return frame.loc[:, OUTPUT_COLUMNS]


def export_version_csv(
    connection: sqlite3.Connection, version_id: int, output_path: Path
) -> dict[str, Any]:
    frame = export_version(connection, version_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return {
        "rows": int(len(frame)),
        "sha256": sha256_file(output_path),
        "path": str(output_path),
    }


def version_summary(connection: sqlite3.Connection, version_id: int) -> dict[str, Any]:
    row = connection.execute(
        """SELECT v.*, d.kind, d.label, d.round_name, d.verification_basis,
                  d.human_verified
           FROM dataset_version v JOIN dataset d ON d.dataset_id = v.dataset_id
           WHERE v.version_id = ?""",
        (version_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown version_id {version_id}")
    return dict(row)


def _count_observations(connection: sqlite3.Connection) -> dict[str, int]:
    row = connection.execute(
        """SELECT COUNT(*) AS rows, COUNT(DISTINCT trajectory_id) AS trajectories
           FROM observation"""
    ).fetchone()
    duplicate_blocks = connection.execute(
        """SELECT COUNT(*) FROM (
             SELECT content_hash FROM trajectory
             GROUP BY content_hash HAVING COUNT(*) > 1)"""
    ).fetchone()[0]
    return {
        "observation_rows": int(row["rows"]),
        "trajectories": int(row["trajectories"]),
        "duplicate_content_hashes": int(duplicate_blocks),
    }


def verify_store(
    connection: sqlite3.Connection,
    *,
    repo_root: Path,
    export_dir: Path | None = None,
    expected_versions: Sequence[str] = (),
    require_legacy_csv: bool = False,
    retirement_manifest: Path | None = None,
) -> dict[str, Any]:
    """Recompute hashes/counts and compare exports with the legacy CSVs.

    After phase 3 the legacy corpus CSVs are retired (deleted); the byte-exact
    export to the stored legacy CSV sha256 remains the integrity proof. Set
    ``require_legacy_csv=True`` to fail if a legacy CSV file is missing. When a
    ``retirement_manifest`` is supplied, a missing CSV is only accepted if that
    lineage id is recorded as retired there - otherwise it is a failure (guards
    against accidental deletion).
    """

    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    retired_lineage_ids: set[str] | None = None
    if retirement_manifest is not None:
        retirement_path = Path(retirement_manifest)
        if not retirement_path.is_file():
            raise FileNotFoundError(f"retirement manifest not found: {retirement_path}")
        manifest_json = json.loads(retirement_path.read_text(encoding="utf-8"))
        retired_lineage_ids = {
            str(record["lineage_id"]) for record in manifest_json["retired_files"]
        }

    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    checks.append(
        {
            "check": "sqlite_integrity_check",
            "status": "PASS" if integrity == "ok" else "FAIL",
            "detail": integrity,
        }
    )
    if integrity != "ok":
        failures.append(f"sqlite integrity_check: {integrity}")

    checks.append(
        {"check": "store_counts", "status": "PASS", "detail": _count_observations(connection)}
    )

    versions = connection.execute(
        """SELECT version_id, dataset_id, version, rows, trajectories,
                  division_points, excluded_points, overlap_rows, canonical_hash,
                  legacy_csv_path, legacy_csv_sha256
           FROM dataset_version ORDER BY version_id"""
    ).fetchall()

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    dirty_version_ids: set[int] = set()
    review_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_version_state'"
    ).fetchone()
    if review_table:
        dirty_version_ids = {int(r[0]) for r in connection.execute("SELECT version_id FROM review_version_state")}
    for row in versions:
        version_id = int(row["version_id"])
        lineage_id = version_lineage_id(row["dataset_id"], row["version"])
        if version_id in dirty_version_ids:
            state = connection.execute("SELECT frozen FROM dataset_version WHERE version_id=?", (version_id,)).fetchone()
            if state and not int(state["frozen"]):
                frame = export_version(connection, version_id) if int(row["rows"]) else pd.DataFrame(columns=OUTPUT_COLUMNS)
                digest = hashlib.sha256()
                members = connection.execute(
                    "SELECT t.trajectory_key,t.content_hash,m.ordinal FROM dataset_member m JOIN trajectory t ON t.trajectory_id=m.trajectory_id WHERE m.version_id=? ORDER BY t.trajectory_key", (version_id,)).fetchall()
                for member in members:
                    observed = frame.loc[frame["trajectory_key"].eq(member["trajectory_key"])]
                    actual_trajectory_hash = trajectory_content_hash(
                        (int(r.fov_cell_index), float(r.time_h), float(r.spine_length), bool(r.division_point), bool(r.excluded_point))
                        for r in observed.itertuples(index=False))
                    if actual_trajectory_hash != member["content_hash"]:
                        failures.append(f"working trajectory hash mismatch: {member['trajectory_key']}")
                    digest.update(f"{member['trajectory_key']}\t{actual_trajectory_hash}\n".encode())
                actual = (len(frame), frame["trajectory_key"].nunique(), int(frame["division_point"].sum()), int(frame["excluded_point"].sum()), int((frame["division_point"] & frame["excluded_point"]).sum()) if len(frame) else 0)
                expected = tuple(int(row[k]) for k in ("rows","trajectories","division_points","excluded_points","overlap_rows"))
                ordinals = sorted(int(m["ordinal"]) for m in members)
                ordered = not ordinals or ordinals == list(range(ordinals[0], ordinals[0]+len(ordinals)))
                ok = actual == expected and digest.hexdigest() == row["canonical_hash"] and ordered
                if not ok:
                    failures.append(f"working version hash/count/order mismatch: {lineage_id}")
                checks.append({"check": "dataset_version", "version": lineage_id,
                    "status": "WORKING" if ok else "FAIL", "legacy_csv_state": "invalidated_by_review",
                    "canonical_hash_matches_stored": digest.hexdigest() == row["canonical_hash"],
                    "counts_match": actual == expected, "detail": "Working data verified; finalize a new lineage before training."})
                if require_legacy_csv:
                    failures.append(f"working reviewed lineage is not a historical CSV: {lineage_id}")
                continue
        frame = export_version(connection, version_id)

        recalculated = hashlib.sha256()
        members = connection.execute(
            """SELECT t.trajectory_key, t.content_hash FROM dataset_member m
               JOIN trajectory t ON t.trajectory_id = m.trajectory_id
               WHERE m.version_id = ? ORDER BY t.trajectory_key""",
            (version_id,),
        ).fetchall()
        for member in members:
            recalculated.update(
                f"{member['trajectory_key']}\t{member['content_hash']}\n".encode("utf-8")
            )
        recalculated_hex = recalculated.hexdigest()

        actual_rows = int(len(frame))
        actual_trajectories = int(frame["trajectory_key"].nunique())
        actual_division = int(frame["division_point"].sum())
        actual_excluded = int(frame["excluded_point"].sum())
        actual_overlap = int((frame["division_point"] & frame["excluded_point"]).sum())

        legacy_sha = row["legacy_csv_sha256"]
        legacy_path = row["legacy_csv_path"]
        legacy_exists = bool(legacy_path) and (repo_root / legacy_path).is_file()
        legacy_file_sha = (
            sha256_file(repo_root / legacy_path) if legacy_exists else None
        )
        native_review = False
        if not legacy_path and legacy_sha:
            provenance = connection.execute(
                "SELECT 1 FROM ingest_provenance WHERE version_id=? AND role='source_lineage'",
                (version_id,),
            ).fetchone()
            kind = connection.execute("SELECT kind FROM dataset WHERE dataset_id=?", (row["dataset_id"],)).fetchone()
            events_exist = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_event'").fetchone()
            event_ok = False
            if events_exist:
                events = connection.execute(
                    "SELECT payload_json FROM review_event WHERE event_type='finalize' AND lineage_id=?", (lineage_id,)).fetchall()
                for event in events:
                    try:
                        metadata = json.loads(event["payload_json"])
                        event_ok = (metadata.get("version_id") == version_id
                            and metadata.get("canonical_hash") == row["canonical_hash"]
                            and metadata.get("legacy_csv_sha256") == legacy_sha)
                    except (ValueError, TypeError, AttributeError):
                        event_ok = False
                    if event_ok:
                        break
            native_review = bool(provenance and kind and kind["kind"] == "review_finalized" and event_ok)
        if native_review:
            # No legacy CSV ever existed for app-finalized data. Its byte-exact
            # export hash is pinned in the database and independently recomputed.
            actual_export_sha = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
            legacy_csv_ok = actual_export_sha == legacy_sha
            legacy_csv_state = "database_native"
        elif legacy_exists:
            legacy_csv_ok = legacy_file_sha == legacy_sha
            legacy_csv_state = "present"
        elif require_legacy_csv:
            legacy_csv_ok = False
            legacy_csv_state = "missing"
        elif retired_lineage_ids is None or lineage_id not in retired_lineage_ids:
            legacy_csv_ok = False
            legacy_csv_state = "missing_unrecorded"
        else:
            # Retired in phase 3: the export equality check below is the proof.
            legacy_csv_ok = True
            legacy_csv_state = "retired"
        if not legacy_csv_ok:
            failures.append(f"legacy CSV missing or mismatched for {lineage_id}")

        export_check: dict[str, Any] = {
            "check": "export_byte_identical",
            "version": lineage_id,
            "status": "SKIP",
        }
        if export_dir is not None:
            target = export_dir / f"{row['dataset_id']}_v{row['version']}.csv"
            result = export_version_csv(connection, version_id, target)
            identical = result["sha256"] == legacy_sha
            export_check.update(
                {
                    "status": "PASS" if identical else "FAIL",
                    "export_sha256": result["sha256"],
                    "legacy_sha256": legacy_sha,
                }
            )
            if not identical:
                failures.append(f"export not byte-identical for {lineage_id}")

        version_ok = (
            actual_rows == int(row["rows"])
            and actual_trajectories == int(row["trajectories"])
            and actual_division == int(row["division_points"])
            and actual_excluded == int(row["excluded_points"])
            and actual_overlap == int(row["overlap_rows"])
            and recalculated_hex == row["canonical_hash"]
            and legacy_csv_ok
        )
        if not version_ok:
            failures.append(f"version check failed for {lineage_id}")

        checks.append(
            {
                "check": "dataset_version",
                "version": lineage_id,
                "status": "PASS" if version_ok else "FAIL",
                "rows": actual_rows,
                "trajectories": actual_trajectories,
                "division_points": actual_division,
                "excluded_points": actual_excluded,
                "canonical_hash": recalculated_hex,
                "canonical_hash_matches_stored": recalculated_hex
                == row["canonical_hash"],
                "legacy_csv_sha256": legacy_sha,
                "legacy_csv_present": legacy_exists,
                "legacy_csv_state": legacy_csv_state,
                "legacy_csv_matches_stored": legacy_file_sha == legacy_sha,
            }
        )
        checks.append(export_check)

    present = {
        version_lineage_id(row["dataset_id"], row["version"]) for row in versions
    }
    for expected in expected_versions:
        if expected not in present:
            failures.append(f"expected version missing: {expected}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }


def backup_database(db_path: Path, backup_dir: Path) -> dict[str, Any]:
    """Snapshot the database with VACUUM INTO and gzip it."""

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    plain = backup_dir / f"lineage_{stamp}.sqlite3"
    connection = connect(db_path)
    try:
        connection.execute("VACUUM INTO ?", (str(plain),))
    finally:
        connection.close()
    compressed = plain.with_suffix(".sqlite3.gz")
    with plain.open("rb") as source, gzip.open(
        compressed, "wb", compresslevel=6
    ) as target:
        shutil.copyfileobj(source, target)
    plain.unlink()
    return {
        "path": str(compressed),
        "sha256": sha256_file(compressed),
        "bytes": compressed.stat().st_size,
    }


def load_manifest_entries(repo_root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(
        (repo_root / "datasets" / "manifest.json").read_text(encoding="utf-8")
    )
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["datasets"]:
        first_segment = str(entry["path"]).split("/")[0]
        entries[first_segment] = entry
    return entries

# ---------------------------------------------------------------------------
# Phase 2: training reads the store directly and records lineage ids.
# ---------------------------------------------------------------------------

LINEAGE_ID_PATTERN = re.compile(
    r"^(?P<dataset_id>[A-Za-z0-9_.-]+)@v(?P<version>[0-9]+)"
    r"(?:#(?P<hash_prefix>[0-9a-fA-F]{6,64}))?$"
)

TRAINING_COLUMNS = tuple(OUTPUT_COLUMNS)


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the dataset store read-only (training never mutates the store)."""

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"dataset store not found: {db_path}")
    connection = sqlite3.connect(
        f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def parse_lineage_id(lineage_id: str) -> tuple[str, int, str | None]:
    """Parse a lineage id: dataset_id@vN, optionally followed by #hash-prefix."""

    match = LINEAGE_ID_PATTERN.match(str(lineage_id).strip())
    if match is None:
        raise ValueError(
            f"invalid dataset lineage id {lineage_id!r}; expected "
            "<dataset_id>@v<version>[#<hash-prefix>]"
        )
    hash_prefix = match.group("hash_prefix")
    return (
        match.group("dataset_id"),
        int(match.group("version")),
        hash_prefix.lower() if hash_prefix else None,
    )


def resolve_lineage(
    connection: sqlite3.Connection,
    lineage_id: str,
    *,
    require_human_verified: bool = False,
) -> sqlite3.Row:
    """Resolve a lineage id to its immutable dataset version row."""

    dataset_id, version, hash_prefix = parse_lineage_id(lineage_id)
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
        raise ValueError(f"unknown dataset lineage {lineage_id!r}")
    if int(row["frozen"]) != 1:
        raise ValueError(f"dataset lineage {lineage_id!r} is not frozen")
    if hash_prefix is not None and not str(row["canonical_hash"]).startswith(
        hash_prefix
    ):
        raise ValueError(
            f"dataset lineage {lineage_id!r} content hash does not match the store"
        )
    if require_human_verified and int(row["human_verified"]) != 1:
        raise ValueError(
            f"dataset lineage {lineage_id!r} is not attested as human verified"
        )
    return row


def lineage_record(row: sqlite3.Row) -> dict[str, Any]:
    canonical_hash = str(row["canonical_hash"])
    return {
        "lineage_id": version_lineage_id(str(row["dataset_id"]), int(row["version"])),
        "content_ref": version_content_ref(
            str(row["dataset_id"]), int(row["version"]), canonical_hash
        ),
        "dataset_id": str(row["dataset_id"]),
        "version": int(row["version"]),
        "kind": str(row["kind"]),
        "label": str(row["label"]),
        "round_name": row["round_name"],
        "canonical_hash": canonical_hash,
        "legacy_csv_path": row["legacy_csv_path"],
        "legacy_csv_sha256": row["legacy_csv_sha256"],
        "rows": int(row["rows"]),
        "trajectories": int(row["trajectories"]),
        "frozen": bool(int(row["frozen"])),
        "human_verified": bool(int(row["human_verified"])),
    }


def _trajectory_keys_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for key in sorted(frame["trajectory_key"].drop_duplicates().tolist()):
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_training_frame(
    connection: sqlite3.Connection,
    lineage_ids: Sequence[str],
    *,
    require_human_verified: bool = True,
    store_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and compose a training frame from one or more frozen versions.

    Rows are ordered exactly like the version export (historical CSV order), so
    a single-lineage load reproduces the historical CSV and therefore the same
    deterministic split. Duplicate observations shared by composed versions are
    collapsed after verifying that their stored content is identical.
    """

    lineages = [str(lineage_id).strip() for lineage_id in lineage_ids]
    if not lineages:
        raise ValueError("at least one dataset lineage id is required")
    if len(set(lineages)) != len(lineages):
        raise ValueError("dataset lineage ids must be unique")

    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for lineage_id in lineages:
        row = resolve_lineage(
            connection,
            lineage_id,
            require_human_verified=require_human_verified,
        )
        records.append(lineage_record(row))
        frames.append(export_version(connection, int(row["version_id"])))

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    duplicate_rows = int(
        combined.duplicated(subset=["trajectory_key", "fov_cell_index"]).sum()
    )
    collapsed = 0
    if duplicate_rows:
        groups = combined.loc[
            combined.duplicated(
                subset=["trajectory_key", "fov_cell_index"], keep=False
            )
        ].groupby(["trajectory_key", "fov_cell_index"], sort=False)
        for key, group in groups:
            if len(group.drop_duplicates()) > 1:
                raise ValueError(
                    "composed dataset versions disagree on observation "
                    f"{key!r}; refusal to guess"
                )
        collapsed = duplicate_rows
        combined = combined.drop_duplicates(
            subset=["trajectory_key", "fov_cell_index"], keep="first"
        )
    combined = combined.reset_index(drop=True).loc[:, OUTPUT_COLUMNS]

    schema_version = int(
        connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    )
    block: dict[str, Any] = {
        "source": "sqlite_dataset_store",
        "store_path": str(store_path.resolve()) if store_path else None,
        "schema_version": schema_version,
        "lineages": records,
        "composition": {
            "lineage_ids": [record["lineage_id"] for record in records],
            "rows": int(len(combined)),
            "trajectories": int(combined["trajectory_key"].nunique()),
            "duplicate_observation_rows_collapsed": collapsed,
            "trajectory_keys_sha256": _trajectory_keys_sha256(combined),
        },
    }
    return combined, block
