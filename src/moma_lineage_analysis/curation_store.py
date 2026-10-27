"""Durable, versioned local curation sessions, separate from the training store.

Measurements occupy ordinary SQLite columns; annotations are supplied by the
app's CSV builder, never derived here. Each operation owns a short connection.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sqlite3
from typing import Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd

DB_FILENAME = "moma_curation_v3.sqlite3"
NETWORK_PATHS_SUPPORTED = True
_SCHEMA_VERSION = 3
_APPLICATION_ID = 0x4D4F4D43
_LABELS = ["trajectory_key", "division_point", "excluded_point",
           "effective_division_point", "cell_cycle_index"]
_BOOLEANS = _LABELS[1:4]
_INTERNAL = {
    "_moma_dataset_id", "_moma_trajectory_key", "_moma_cell_id", "_moma_row_order",
    "_moma_label_order", "_moma_division_state", "_moma_excluded_point",
    "_moma_effective_division_point", "_moma_cell_cycle_index",
}


class CurationStoreError(RuntimeError):
    """A session could not be validated, read, or durably committed."""


class CurationConflictError(CurationStoreError):
    """Another writer has advanced this dataset's revision."""


def _guard(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except CurationStoreError:
            raise
        except Exception as exc:
            raise CurationStoreError(f"{function.__name__}: {exc}") from exc
    return guarded


def _json(value):
    def scalar(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=scalar)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sources(sources):
    result, names = [], set()
    for source in sources:
        name = source["filename"]
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise CurationStoreError("Source filename must be a basename")
        if name.casefold() in names:
            raise CurationStoreError(f"Duplicate source filename: {name}")
        names.add(name.casefold())
        digest = source["sha256"]
        size = source["size"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in digest)):
            raise CurationStoreError(f"Invalid source SHA256: {name}")
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 0:
            raise CurationStoreError(f"Invalid source size: {name}")
        item = {"filename": name, "sha256": digest.lower(), "size": int(size)}
        if source.get("path") is not None:
            item["path"] = str(source["path"])
        result.append(item)
    return sorted(result, key=lambda item: item["filename"])


def _signature(sources, settings, logic_version):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict) or not isinstance(logic_version, str):
        raise CurationStoreError("settings must be a dict and logic_version a string")
    normalized = [{k: s[k] for k in ("filename", "sha256", "size")} for s in sources]
    return hashlib.sha256(_json({"sources": normalized, "settings": settings,
                                "logic_version": logic_version}).encode("utf-8")).hexdigest()


@_guard
def fingerprint_sources(paths: Iterable[str | Path]) -> list[dict]:
    """Hash only explicitly supplied source files, checking for concurrent changes."""
    result, names = [], set()
    for item in paths:
        path = Path(item)
        if path.name.casefold() in names:
            raise CurationStoreError(f"Duplicate source filename: {path.name}")
        names.add(path.name.casefold())
        before = path.stat()
        if not path.is_file():
            raise CurationStoreError(f"Source is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            opened = __import__("os").fstat(stream.fileno())
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            finished = __import__("os").fstat(stream.fileno())
        after = path.stat()
        def identity(stat):
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                    stat.st_ctime_ns)
        if not (identity(before) == identity(opened) == identity(finished) == identity(after)) or size != before.st_size:
            raise CurationStoreError(f"Source changed while hashing: {path}")
        result.append({"filename": path.name, "sha256": digest.hexdigest(),
                       "size": size, "path": str(path.resolve())})
    return sorted(result, key=lambda source: source["filename"])


_SCHEMA = [
    "CREATE TABLE store_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE datasets (
        dataset_id INTEGER PRIMARY KEY, signature TEXT NOT NULL UNIQUE,
        revision INTEGER NOT NULL CHECK(revision >= 0), settings_json TEXT NOT NULL,
        logic_version TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        current_key TEXT, measurement_schema_json TEXT NOT NULL)""",
    """CREATE TABLE source_files (
        dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id), source_id INTEGER NOT NULL,
        filename TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL, path TEXT,
        PRIMARY KEY(dataset_id, source_id), UNIQUE(dataset_id, filename))""",
    """CREATE TABLE measurement_schemas (
        schema_id INTEGER PRIMARY KEY, dtype_json TEXT NOT NULL UNIQUE)""",
    """CREATE TABLE trajectories (
        dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id), trajectory_key TEXT NOT NULL,
        trajectory_order INTEGER NOT NULL, source_id INTEGER NOT NULL, schema_id INTEGER NOT NULL
            REFERENCES measurement_schemas(schema_id),
        selected INTEGER NOT NULL CHECK(selected IN (0,1)),
        review_status TEXT NOT NULL CHECK(review_status IN ('reviewed','unreviewed','in_progress')),
        reviewed_at TEXT, PRIMARY KEY(dataset_id, trajectory_key),
        FOREIGN KEY(dataset_id, source_id) REFERENCES source_files(dataset_id, source_id))""",
    """CREATE TABLE observations (
        _moma_dataset_id INTEGER NOT NULL, _moma_trajectory_key TEXT NOT NULL,
        _moma_cell_id TEXT NOT NULL, _moma_row_order INTEGER NOT NULL,
        _moma_label_order INTEGER NOT NULL DEFAULT 0,
        _moma_division_state INTEGER NOT NULL DEFAULT 0 CHECK(_moma_division_state >= 0),
        _moma_excluded_point INTEGER NOT NULL DEFAULT 0 CHECK(_moma_excluded_point IN (0,1)),
        _moma_effective_division_point INTEGER NOT NULL DEFAULT 0 CHECK(_moma_effective_division_point IN (0,1)),
        _moma_cell_cycle_index INTEGER NOT NULL DEFAULT 0 CHECK(_moma_cell_cycle_index >= 0),
        PRIMARY KEY(_moma_dataset_id, _moma_trajectory_key, _moma_cell_id),
        UNIQUE(_moma_dataset_id, _moma_trajectory_key, _moma_row_order),
        FOREIGN KEY(_moma_dataset_id, _moma_trajectory_key)
            REFERENCES trajectories(dataset_id, trajectory_key))""",
    """CREATE TABLE app_state (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        active_dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id))""",
]


def _initialize(connection):
    for statement in _SCHEMA:
        connection.execute(statement)
    connection.executemany("INSERT INTO store_metadata VALUES (?,?)",
                           [("format", "moma_curation"), ("measurement_columns", "{}")])
    connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")


def _schema_version(connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _check_schema(connection):
    version = _schema_version(connection)
    app_id = connection.execute("PRAGMA application_id").fetchone()[0]
    if version != _SCHEMA_VERSION or app_id != _APPLICATION_ID:
        raise CurationStoreError(f"Unrecognized or unsupported curation schema (version {version}); database left intact")
    expected = {"store_metadata", "datasets", "source_files", "measurement_schemas",
                "trajectories", "observations", "app_state"}
    present = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
    expected_labels = {"_moma_label_order", "_moma_division_state", "_moma_excluded_point",
                       "_moma_effective_division_point", "_moma_cell_cycle_index"}
    if not expected_labels <= columns:
        raise CurationStoreError("Incomplete compact curation schema")
    if not expected <= present or connection.execute(
            "SELECT value FROM store_metadata WHERE key='format'").fetchone()[0] != "moma_curation":
        raise CurationStoreError("Invalid curation database schema")



def _sqlite_path_text(path) -> str:
    """Normalize Windows extended path prefixes without making a URI host."""

    text = path.as_posix()
    if text.lower().startswith("//?/unc/"):
        return "//" + text[8:]
    if text.startswith("//?/"):
        return text[4:]
    return text


def _sqlite_uri(path, mode: str) -> str:
    """Keep UNC hosts in the path: SQLite rejects file://server authorities."""

    if mode not in {"ro", "rw", "rwc"}:
        raise ValueError(f"Unsupported SQLite open mode: {mode}")
    text = _sqlite_path_text(path)
    if not text.startswith("/"):
        text = "/" + text
    return "file://" + quote(text, safe="/:") + f"?mode={mode}"


def _windows_drive_type(root: str) -> int:
    import ctypes
    from ctypes import wintypes

    get_drive_type = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
    get_drive_type.argtypes = [wintypes.LPCWSTR]
    get_drive_type.restype = wintypes.UINT
    return int(get_drive_type(root))


def _is_network_path(path) -> bool:
    """Recognize UNC paths and mapped remote drives without contacting the share."""

    text = _sqlite_path_text(path)
    if text.startswith("//"):
        return True
    if os.name != "nt":
        return False
    root = PureWindowsPath(text).anchor
    try:
        # DRIVE_REMOTE=4. An unidentifiable filesystem also uses conservative
        # rollback journaling, since WAL requires same-host shared memory.
        return _windows_drive_type(root) in (0, 1, 4)
    except (OSError, AttributeError):
        return True


def _configure_journal(connection, path) -> None:
    desired = "delete" if _is_network_path(path) else "wal"
    row = connection.execute(f"PRAGMA journal_mode={desired}").fetchone()
    actual = str(row[0]).lower() if row else ""
    if actual != desired:
        raise CurationStoreError(
            f"Could not enable {desired.upper()} journaling for {path} "
            f"(SQLite returned {actual!r}). Close other connections and retry autosave."
        )
    connection.execute("PRAGMA synchronous=FULL")


@contextmanager
def _connect(db_path, *, write=False, create=False):
    path = Path(db_path).resolve()
    existed = path.exists()
    if not existed and not create:
        raise CurationStoreError(f"Curation database does not exist: {path}")
    mode = "rwc" if create and not existed else "rw" if write else "ro"
    connection = sqlite3.connect(_sqlite_uri(path, mode), uri=True,
                                 timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        # Validate an existing file before changing its journal or any schema.
        if existed:
            _check_schema(connection)
        if write:
            if not existed:
                # A fresh SQLite file defaults to rollback journaling. Commit a
                # recognizable empty schema before switching journal modes so a
                # network/journal setup failure remains safely retryable.
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                _initialize(connection)
                connection.commit()
            _configure_journal(connection, path)
            connection.execute("BEGIN IMMEDIATE")
        else:
            connection.execute("BEGIN")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _summary(connection, dataset_id):
    row = connection.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
    if row is None:
        raise CurationStoreError(f"Unknown dataset_id: {dataset_id}")
    counts = connection.execute("""SELECT COUNT(*), COALESCE(SUM(review_status='reviewed'),0),
        COALESCE(SUM(selected),0) FROM trajectories WHERE dataset_id=?""", (dataset_id,)).fetchone()
    return {"dataset_id": row["dataset_id"], "revision": row["revision"],
            "signature": row["signature"], "settings": json.loads(row["settings_json"]),
            "logic_version": row["logic_version"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], "current_key": row["current_key"],
            "trajectory_count": counts[0], "reviewed_count": counts[1], "selected_count": counts[2],
            "sources": [{k: v for k, v in dict(source).items() if k not in {"dataset_id", "source_id"} and v is not None}
                        for source in connection.execute("SELECT * FROM source_files WHERE dataset_id=? ORDER BY filename", (dataset_id,))]}


@_guard
def find_dataset(db_path, *, sources: list[dict], settings: dict, logic_version: str) -> dict | None:
    signature = _signature(_sources(sources), settings, logic_version)
    if not Path(db_path).exists():
        return None
    with _connect(db_path) as connection:
        row = connection.execute("SELECT dataset_id FROM datasets WHERE signature=?", (signature,)).fetchone()
        return _summary(connection, row[0]) if row else None


@_guard
def list_datasets(db_path) -> list[dict]:
    if not Path(db_path).exists():
        return []
    with _connect(db_path) as connection:
        ids = connection.execute("SELECT dataset_id FROM datasets ORDER BY updated_at DESC, dataset_id DESC").fetchall()
        return [_summary(connection, row[0]) for row in ids]


def _cell_id(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or bool(pd.isna(value)):
        raise CurationStoreError("fov_cell_index cannot be missing")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise CurationStoreError("fov_cell_index must contain scalar string or numeric IDs")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CurationStoreError("fov_cell_index must be finite")
        if value.is_integer():
            value = int(value)
    return _json(value)


def _dtype_metadata(frame):
    metadata = []
    for name in frame.columns:
        dtype = frame[name].dtype
        info = {"name": name, "dtype": str(dtype)}
        if isinstance(dtype, pd.CategoricalDtype):
            info.update(categories=dtype.categories.tolist(), ordered=dtype.ordered)
        metadata.append(info)
    return metadata


def _sql_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not -(2**63) <= value < 2**63:
        return str(value)
    if isinstance(value, (str, int, float, bytes)):
        return value
    raise CurationStoreError(f"Unsupported measurement value: {type(value).__name__}")


def _restore_frame(rows, metadata):
    data = {}
    for index, info in enumerate(metadata):
        values = [row[index] for row in rows]
        if info["dtype"] == "category":
            series = pd.Series(pd.Categorical(values, categories=info["categories"], ordered=info["ordered"]))
        else:
            # Construct each Series directly with its dtype, avoiding pandas'
            # whole-frame float inference (which can round large integer IDs).
            series = pd.Series(values, dtype=info["dtype"])
        data[info["name"]] = series
    return pd.DataFrame(data)


def _equal_measurements(left, right):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                      check_exact=True, check_names=False)
    except (AssertionError, TypeError, ValueError) as exc:
        raise CurationStoreError("Labeled rows or updated immutable measurements differ from the raw trajectory") from exc


def _validate_public_columns(columns):
    folded = [name.casefold() for name in columns]
    if len(set(folded)) != len(folded) or set(folded) & {name.casefold() for name in _LABELS}:
        raise CurationStoreError("Measurement names must be unique ignoring case and cannot overlap export labels")


def _prepare(key, entry, labeled):
    if not isinstance(key, str) or not key:
        raise CurationStoreError("trajectory_key must be a nonempty string")
    raw = entry["trajectory"]
    if not isinstance(raw, pd.DataFrame) or not isinstance(labeled, pd.DataFrame):
        raise CurationStoreError("trajectory and labeled frame must be pandas DataFrames")
    columns = list(raw.columns)
    if (not raw.columns.is_unique or any(not isinstance(c, str) or not c or "\x00" in c for c in columns)
            or "fov_cell_index" not in columns or set(columns) & set(_LABELS)):
        raise CurationStoreError("Raw measurements need unique string columns and fov_cell_index; label columns are reserved")
    _validate_public_columns(columns)
    if not labeled.columns.is_unique or set(labeled.columns) != set(columns) | set(_LABELS):
        raise CurationStoreError("Labeled frame must contain exactly raw columns and the five export labels")
    raw_ids = [_cell_id(v) for v in raw["fov_cell_index"]]
    label_ids = [_cell_id(v) for v in labeled["fov_cell_index"]]
    if len(set(raw_ids)) != len(raw_ids) or len(set(label_ids)) != len(label_ids):
        raise CurationStoreError(f"Duplicate fov_cell_index within trajectory {key}")
    if len(raw_ids) != len(label_ids) or set(raw_ids) != set(label_ids):
        raise CurationStoreError("Labeled frame must contain every raw observation exactly once")
    if not labeled["trajectory_key"].eq(key).all():
        raise CurationStoreError("Labeled trajectory_key does not match its entry")
    positions = {cell: pos for pos, cell in enumerate(label_ids)}
    aligned = labeled.iloc[[positions[cell] for cell in raw_ids]][columns]
    _equal_measurements(raw, aligned)
    for column in _BOOLEANS:
        if labeled[column].isna().any() or not labeled[column].isin([False, True, 0, 1]).all():
            raise CurationStoreError(f"Invalid boolean label {column}")
    cycles = labeled["cell_cycle_index"]
    if cycles.isna().any() or any(isinstance(v, (str, bool)) or not math.isfinite(float(v)) or float(v) < 0 or not float(v).is_integer() for v in cycles):
        raise CurationStoreError("cell_cycle_index must be a nonnegative integer")
    division_ids = {_cell_id(v) for v in entry.get("div_indices", [])}
    excluded_ids = {_cell_id(v) for v in entry.get("excluded_indices", set())}
    original = list(entry.get("original_div_indices", entry.get("div_indices", [])))
    original_order = {_cell_id(v): order + 1 for order, v in enumerate(original)}
    original_ids = set(original_order)
    if len(original_order) != len(original):
        raise CurationStoreError("Original division baseline contains duplicate cell IDs")
    if not (division_ids | excluded_ids | original_ids) <= set(raw_ids):
        raise CurationStoreError("Annotations reference an unknown fov_cell_index")
    if division_ids != {cell for cell, flag in zip(label_ids, labeled["division_point"]) if flag} or excluded_ids != {cell for cell, flag in zip(label_ids, labeled["excluded_point"]) if flag}:
        raise CurationStoreError("Raw annotation sets disagree with supplied CSV labels")
    status = entry.get("_review_status", "unreviewed")
    if status not in ("unreviewed", "in_progress", "reviewed"):
        raise CurationStoreError("Unknown review status")
    reviewed_at = entry.get("_reviewed_at")
    if reviewed_at is not None and not isinstance(reviewed_at, str):
        raise CurationStoreError("_reviewed_at must be an ISO timestamp string or None")
    source_path = str(entry["source_csv_path"])
    annotations = [(cell, order, int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                   for order, (cell, row) in enumerate(zip(label_ids, labeled[_LABELS[1:]].itertuples(index=False, name=None)))]
    return {"raw": raw, "ids": raw_ids, "metadata": _dtype_metadata(raw),
            "values": [tuple(_sql_value(v) for v in row) for row in raw.itertuples(index=False, name=None)],
            "annotations": annotations, "original": _json(original),
            "original_order": original_order, "status": status,
            "reviewed_at": reviewed_at, "source_path": source_path,
            "source_filename": Path(source_path).name}


def _validate_membership(keys, selected_keys, current_key):
    if not set(selected_keys) <= set(keys):
        raise CurationStoreError("Selection contains an unknown trajectory")
    if current_key is not None and current_key not in keys:
        raise CurationStoreError("Current trajectory is unknown")


def _column_map(connection, names=()):
    mapping = json.loads(connection.execute("SELECT value FROM store_metadata WHERE key='measurement_columns'").fetchone()[0])
    used = {name.casefold() for name in _INTERNAL} | {name.casefold() for name in mapping.values()}
    for name in names:
        if name in mapping:
            continue
        physical = name
        if physical.casefold() in used:
            physical = f"_moma_measurement_{len(mapping)}"
            while physical.casefold() in used:
                physical += "_"
        used.add(physical.casefold())
        # No affinity lets SQLite preserve each scalar's actual numeric/text type.
        connection.execute(f"ALTER TABLE observations ADD COLUMN {_quote(physical)}")
        mapping[name] = physical
    if names:
        connection.execute("UPDATE store_metadata SET value=? WHERE key='measurement_columns'", (_json(mapping),))
    return mapping


def _activate(connection, dataset_id):
    schema = json.loads(connection.execute("SELECT measurement_schema_json FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()[0])
    mapping = _column_map(connection)
    projection = [f"o.{_quote(mapping[name])} AS {_quote(name)}" for name in schema]
    projection += ['t.trajectory_key AS "trajectory_key"']
    projection += ['(o._moma_division_state & 1) AS "division_point"']
    projection += [f"o._moma_{name} AS {_quote(name)}" for name in _LABELS[2:]]
    base = """ FROM observations o
        JOIN trajectories t ON t.dataset_id=o._moma_dataset_id AND t.trajectory_key=o._moma_trajectory_key
        JOIN app_state s ON s.singleton=1 AND s.active_dataset_id=t.dataset_id"""
    for view, condition in (("all_labeled", ""), ("selected_labeled", " WHERE t.selected=1"),
                            ("reviewed_selected_labeled", " WHERE t.selected=1 AND t.review_status='reviewed'")):
        connection.execute(f"DROP VIEW IF EXISTS {view}")
        label_order = "o._moma_label_order"
        connection.execute(f"CREATE VIEW {view} AS SELECT " + ", ".join(projection) + base + condition + f" ORDER BY t.trajectory_order, {label_order}")
    connection.execute("INSERT INTO app_state VALUES (1,?) ON CONFLICT(singleton) DO UPDATE SET active_dataset_id=excluded.active_dataset_id", (dataset_id,))


def _insert_annotations(connection, dataset_id, key, prepared):
    # Bit 0 is the current label; higher bits preserve the original division
    # ordinal (zero means absent). This replaces the per-trajectory JSON list
    # without adding a column or losing its leaf-to-root reset order.
    connection.executemany("""UPDATE observations SET _moma_label_order=?, _moma_division_state=?,
        _moma_excluded_point=?, _moma_effective_division_point=?, _moma_cell_cycle_index=?
        WHERE _moma_dataset_id=? AND _moma_trajectory_key=? AND _moma_cell_id=?""",
        [(order, (prepared["original_order"].get(cell, 0) << 1) | division,
          exclusion, effective, cycle, dataset_id, key, cell)
         for cell, order, division, exclusion, effective, cycle in prepared["annotations"]])


@_guard
def create_dataset(db_path, *, trajectory_dict: dict[str, dict],
                   labeled_frames: dict[str, pd.DataFrame], selected_keys: set[str],
                   sources: list[dict], settings: dict, logic_version: str,
                   current_key: str | None = None) -> dict:
    """Initialize atomically, or reuse a matching version without resetting it."""
    sources = _sources(sources)
    settings = {} if settings is None else settings
    signature = _signature(sources, settings, logic_version)
    # Idempotent callers need not regenerate old labels or reset their cursor.
    if Path(db_path).exists():
        with _connect(db_path, write=True) as connection:
            row = connection.execute("SELECT dataset_id FROM datasets WHERE signature=?", (signature,)).fetchone()
            if row:
                _activate(connection, row[0])
                return dict(_summary(connection, row[0]), created=False)
    if set(trajectory_dict) != set(labeled_frames):
        raise CurationStoreError("Trajectory and labeled frame keys must match")
    _validate_membership(trajectory_dict, selected_keys, current_key)
    prepared = {key: _prepare(key, entry, labeled_frames[key]) for key, entry in trajectory_dict.items()}
    filenames = {source["filename"] for source in sources}
    if any(p["source_filename"] not in filenames for p in prepared.values()):
        raise CurationStoreError("Trajectory source filename is absent from source fingerprints")
    schema = list(dict.fromkeys(column for p in prepared.values() for column in p["raw"].columns))
    _validate_public_columns(schema)
    with _connect(db_path, write=True, create=True) as connection:
        # Recheck under the writer lock: another process may have initialized it.
        row = connection.execute("SELECT dataset_id FROM datasets WHERE signature=?", (signature,)).fetchone()
        if row:
            _activate(connection, row[0])
            return dict(_summary(connection, row[0]), created=False)
        stamp = _now()
        dataset_id = connection.execute("""INSERT INTO datasets(signature,revision,settings_json,
            logic_version,created_at,updated_at,current_key,measurement_schema_json)
            VALUES (?,0,?,?,?,?,?,?)""", (signature, _json(settings), logic_version, stamp, stamp, current_key, _json(schema))).lastrowid
        source_ids = {}
        for source_id, source in enumerate(sources):
            source_ids[source["filename"]] = source_id
            path = source.get("path")
            if path is None:
                path = next((p["source_path"] for p in prepared.values()
                             if p["source_filename"] == source["filename"]), None)
            connection.execute("INSERT INTO source_files VALUES (?,?,?,?,?,?)",
                               (dataset_id, source_id, source["filename"], source["sha256"], source["size"], path))
        mapping = _column_map(connection, schema)
        schema_ids = {}
        for order, (key, p) in enumerate(prepared.items()):
            dtype_json = _json(p["metadata"])
            if dtype_json not in schema_ids:
                connection.execute("INSERT INTO measurement_schemas(dtype_json) VALUES (?) ON CONFLICT DO NOTHING", (dtype_json,))
                schema_ids[dtype_json] = connection.execute(
                    "SELECT schema_id FROM measurement_schemas WHERE dtype_json=?", (dtype_json,)).fetchone()[0]
            connection.execute("INSERT INTO trajectories VALUES (?,?,?,?,?,?,?,?)",
                               (dataset_id, key, order, source_ids[p["source_filename"]],
                                schema_ids[dtype_json], int(key in selected_keys), p["status"], p["reviewed_at"]))
            columns = [mapping[name] for name in p["raw"].columns]
            statement = "INSERT INTO observations (" + ",".join(_quote(name) for name in ["_moma_dataset_id", "_moma_trajectory_key", "_moma_cell_id", "_moma_row_order", *columns]) + ") VALUES (" + ",".join("?" for _ in range(4 + len(columns))) + ")"
            connection.executemany(statement, [(dataset_id, key, cell, row_order, *values) for row_order, (cell, values) in enumerate(zip(p["ids"], p["values"]))])
            _insert_annotations(connection, dataset_id, key, p)
        _activate(connection, dataset_id)
        return dict(_summary(connection, dataset_id), created=True)


def _read_raw(connection, dataset_id, key, metadata, mapping):
    columns = ",".join(_quote(mapping[info["name"]]) for info in metadata)
    rows = connection.execute(f"SELECT {columns} FROM observations WHERE _moma_dataset_id=? AND _moma_trajectory_key=? ORDER BY _moma_row_order", (dataset_id, key)).fetchall()
    return _restore_frame(rows, metadata)


def _annotation_rows(connection, dataset_id, key):
    """Read labels in raw row order from compact observation storage."""

    return connection.execute("""SELECT _moma_cell_id AS fov_cell_index,
        _moma_label_order AS label_order, (_moma_division_state & 1) AS division_point,
        (_moma_division_state >> 1) AS original_division_order,
        _moma_excluded_point AS excluded_point,
        _moma_effective_division_point AS effective_division_point,
        _moma_cell_cycle_index AS cell_cycle_index
        FROM observations WHERE _moma_dataset_id=? AND _moma_trajectory_key=?
        ORDER BY _moma_row_order""", (dataset_id, key)).fetchall()


def _original_divisions(annotations, values):
    return [values[row["fov_cell_index"]] for row in sorted(
        (row for row in annotations if row["original_division_order"]),
        key=lambda row: row["original_division_order"],
    )]


@_guard
def load_dataset(db_path, dataset_id: int, *, source_dir: str | Path | None = None) -> dict:
    """Recover raw measurements and annotations; activate the requested SQL views."""
    with _connect(db_path, write=True) as connection:
        result = _summary(connection, dataset_id)
        mapping = _column_map(connection)
        entries, selected = {}, set()
        for t in connection.execute("""SELECT t.*, s.filename, s.path, m.dtype_json FROM trajectories t
                JOIN source_files s ON s.dataset_id=t.dataset_id AND s.source_id=t.source_id
                JOIN measurement_schemas m ON m.schema_id=t.schema_id
                WHERE t.dataset_id=? ORDER BY t.trajectory_order""", (dataset_id,)).fetchall():
            key = t["trajectory_key"]
            frame = _read_raw(connection, dataset_id, key, json.loads(t["dtype_json"]), mapping)
            annotations = _annotation_rows(connection, dataset_id, key)
            values = dict(zip((_cell_id(v) for v in frame["fov_cell_index"]), frame["fov_cell_index"].tolist()))
            entry = {"trajectory": frame,
                     "div_indices": [values[a["fov_cell_index"]] for a in annotations if a["division_point"]],
                     "excluded_indices": {values[a["fov_cell_index"]] for a in annotations if a["excluded_point"]},
                     "original_div_indices": _original_divisions(annotations, values),
                     "source_csv_path": str(Path(source_dir) / t["filename"]) if source_dir is not None
                         else t["path"] or str(Path(db_path).resolve().parent / t["filename"]),
                     "_review_status": t["review_status"], "_reviewed_at": t["reviewed_at"],
                     "_saved_revision": result["revision"]}
            entries[key] = entry
            if t["selected"]:
                selected.add(key)
        result.update(trajectory_dict=entries, selected_keys=selected,
                      sources=[{k: v for k, v in dict(s).items() if k not in {"dataset_id", "source_id"} and v is not None} for s in connection.execute("SELECT * FROM source_files WHERE dataset_id=? ORDER BY filename", (dataset_id,))])
        _activate(connection, dataset_id)
        return result


@_guard
def save_changes(db_path, dataset_id, *, expected_revision: int,
                 trajectory_dict: dict, labeled_frames: dict,
                 selected_keys: set, current_key: str | None) -> int:
    """Commit a batch of annotations and full selection with revision CAS."""
    if set(trajectory_dict) != set(labeled_frames):
        raise CurationStoreError("Affected trajectory and labeled frame keys must match")
    if not isinstance(expected_revision, int) or isinstance(expected_revision, bool):
        raise CurationStoreError("expected_revision must be an integer")
    with _connect(db_path, write=True) as connection:
        summary = _summary(connection, dataset_id)
        if summary["revision"] != expected_revision:
            raise CurationConflictError(f"Dataset {dataset_id} changed: expected revision {expected_revision}, found {summary['revision']}")
        stored = {t["trajectory_key"]: t for t in connection.execute("""
            SELECT t.*, s.filename, m.dtype_json FROM trajectories t
            JOIN source_files s ON s.dataset_id=t.dataset_id AND s.source_id=t.source_id
            JOIN measurement_schemas m ON m.schema_id=t.schema_id WHERE t.dataset_id=?""", (dataset_id,))}
        _validate_membership(stored, selected_keys, current_key)
        if not set(trajectory_dict) <= set(stored):
            raise CurationStoreError("Changed batch contains an unknown trajectory")
        mapping = _column_map(connection)
        prepared = {}
        for key, entry in trajectory_dict.items():
            p = _prepare(key, entry, labeled_frames[key])
            old = stored[key]
            raw = _read_raw(connection, dataset_id, key, json.loads(old["dtype_json"]), mapping)
            _equal_measurements(raw, p["raw"])
            annotations = _annotation_rows(connection, dataset_id, key)
            values = dict(zip((_cell_id(v) for v in raw["fov_cell_index"]), raw["fov_cell_index"].tolist()))
            original = _original_divisions(annotations, values)
            if list(p["original_order"]) != [_cell_id(value) for value in original] or p["source_filename"] != old["filename"]:
                raise CurationStoreError("Original division baseline and source filename are immutable")
            prepared[key] = p
        for key, p in prepared.items():
            connection.executemany("""UPDATE observations SET _moma_label_order=?,
                _moma_division_state=(_moma_division_state & ~1) | ?,
                _moma_excluded_point=?, _moma_effective_division_point=?, _moma_cell_cycle_index=?
                WHERE _moma_dataset_id=? AND _moma_trajectory_key=? AND _moma_cell_id=?""",
                [(order, division, exclusion, effective, cycle, dataset_id, key, cell) for cell, order, division, exclusion, effective, cycle in p["annotations"]])
            connection.execute("UPDATE trajectories SET review_status=?, reviewed_at=? WHERE dataset_id=? AND trajectory_key=?", (p["status"], p["reviewed_at"], dataset_id, key))
        connection.executemany("UPDATE trajectories SET selected=? WHERE dataset_id=? AND trajectory_key=?", [(int(key in selected_keys), dataset_id, key) for key in stored])
        revision = expected_revision + 1
        connection.execute("UPDATE datasets SET revision=?, current_key=?, updated_at=? WHERE dataset_id=?", (revision, current_key, _now(), dataset_id))
        _activate(connection, dataset_id)
        return revision
