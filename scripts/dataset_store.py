"""Manage the local SQLite dataset store (phases 1-3: build / verify / export / backup).

The store lives at datasets/store/lineage.sqlite3 and holds the frozen verified
corpora as versioned dataset membership sets over de-duplicated observations;
training reads it directly (phase 2) and the legacy corpus CSVs were retired in
phase 3 (restorable byte-identically with `export`).

After phase 3, `build` skips a corpus whose CSV is missing only if that dataset
version is already in the store; a missing CSV for a dataset the store does not
hold is an error (restore it from datasets/store/retired_csv_manifest_20261017.json
first). `verify` accepts a missing CSV only when it is recorded as retired in
that manifest; `verify --require-legacy-csv` demands the pre-retirement state.

Usage:
    python scripts/dataset_store.py build
    python scripts/dataset_store.py verify [--require-legacy-csv] [--exports-dir dir]
    python scripts/dataset_store.py export --dataset verified_merged_20261017 --output out.csv
    python scripts/dataset_store.py backup
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "datasets" / "store" / "lineage.sqlite3"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from moma_lineage_analysis.trajectory_cleaning import dataset_store  # noqa: E402

IMPORT_ORDER = (
    "verified_20260916",
    "verified_20261013",
    "verified_20261016",
    "verified_merged_20261016",
    "verified_20261017",
    "verified_merged_20261017",
)

# Explicit corpus metadata. Do not infer kind/round by sniffing the role text:
# every role mentions the production corpus, which previously mis-classified
# batches and merged corpora, and two roles lack a round token.
CORPUS_METADATA = {
    "verified_20260916": ("production", "R004"),
    "verified_20261013": ("batch", "R006"),
    "verified_20261016": ("batch", "R007"),
    "verified_20261017": ("batch", "R009"),
    "verified_merged_20261016": ("merged", "R008"),
    "verified_merged_20261017": ("merged", "R010"),
}


def corpus_specs() -> list[dict]:
    """Derive the corpus import specification from datasets/manifest.json.

    Known corpora are imported in a stable documented order; any additional
    manifest entries (a newly curated batch) are appended in sorted order, so
    `build` is idempotent and only inserts genuinely new versions.
    """

    entries = dataset_store.load_manifest_entries(REPO_ROOT)
    directories = list(IMPORT_ORDER) + sorted(set(entries) - set(IMPORT_ORDER))
    specs: list[dict] = []
    for directory in directories:
        entry = entries.get(directory)
        if entry is None:
            raise SystemExit(f"manifest entry missing for {directory}")
        role = str(entry["role"])
        kind, round_name = CORPUS_METADATA.get(
            directory, ("batch", None)
        )
        if round_name is None:
            match = re.search(r"round (R\d+)", role)
            round_name = match.group(1) if match else None
        specs.append(
            {
                "dataset_id": entry["id"],
                "csv_path": REPO_ROOT / "datasets" / str(entry["path"]),
                "kind": kind,
                "label": role,
                "round_name": round_name,
                "verification_basis": entry.get("verification_basis"),
                "human_verified": bool(entry.get("human_verified", False)),
            }
        )
    return specs


def command_build(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    connection = dataset_store.connect(db_path)
    try:
        dataset_store.initialize(connection)
        summary: list[dict] = []
        for spec in corpus_specs():
            if not Path(spec["csv_path"]).is_file():
                existing = connection.execute(
                    "SELECT 1 FROM dataset_version WHERE dataset_id = ?",
                    (spec["dataset_id"],),
                ).fetchone()
                if existing is not None:
                    summary.append(
                        {
                            "dataset_id": spec["dataset_id"],
                            "status": "skipped_retired_csv",
                            "reason": (
                                "corpus CSV retired after phase 3 and already in "
                                "the store; restore it with `export` only if a "
                                "full rebuild is required"
                            ),
                        }
                    )
                    continue
                raise SystemExit(
                    f"corpus CSV missing and not in the store: {spec['csv_path']}; "
                    "restore it from datasets/store/retired_csv_manifest_20261017.json "
                    "and retry"
                )
            result = dataset_store.import_corpus(
                connection,
                repo_root=REPO_ROOT,
                csv_path=spec["csv_path"],
                dataset_id=spec["dataset_id"],
                kind=spec["kind"],
                label=spec["label"],
                round_name=spec["round_name"],
                verification_basis=spec["verification_basis"],
                human_verified=spec["human_verified"],
            )
            summary.append(result)
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "db": str(db_path),
                "imports": summary,
                "db_bytes": db_path.stat().st_size,
            },
            indent=2,
        )
    )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    expected = [
        dataset_store.version_lineage_id(spec["dataset_id"], 1)
        for spec in corpus_specs()
    ]
    retirement_manifest = REPO_ROOT / "datasets" / "store" / "retired_csv_manifest_20261017.json"
    if args.exports_dir:
        connection = dataset_store.connect_readonly(Path(args.db))
        try:
            result = dataset_store.verify_store(
                connection,
                repo_root=REPO_ROOT,
                export_dir=Path(args.exports_dir),
                expected_versions=expected,
                require_legacy_csv=args.require_legacy_csv,
                retirement_manifest=retirement_manifest,
            )
        finally:
            connection.close()
    else:
        # Export comparison needs a scratch directory; it re-materialises the
        # corpora (~330 MB) and is deleted again, so no cache is kept.
        with tempfile.TemporaryDirectory(prefix="lineage_store_verify_") as temp_dir:
            connection = dataset_store.connect_readonly(Path(args.db))
            try:
                result = dataset_store.verify_store(
                    connection,
                    repo_root=REPO_ROOT,
                    export_dir=Path(temp_dir),
                    expected_versions=expected,
                    require_legacy_csv=args.require_legacy_csv,
                    retirement_manifest=retirement_manifest,
                )
            finally:
                connection.close()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def command_export(args: argparse.Namespace) -> int:
    connection = dataset_store.connect_readonly(Path(args.db))
    try:
        row = connection.execute(
            """SELECT version_id FROM dataset_version
               WHERE dataset_id = ? AND version = ?""",
            (args.dataset, args.version),
        ).fetchone()
        if row is None:
            raise SystemExit(f"no such dataset version: {args.dataset} v{args.version}")
        result = dataset_store.export_version_csv(
            connection, int(row["version_id"]), Path(args.output)
        )
    finally:
        connection.close()
    print(json.dumps(result, indent=2))
    return 0


def command_backup(args: argparse.Namespace) -> int:
    result = dataset_store.backup_database(
        Path(args.db), REPO_ROOT / "datasets" / "store" / "backups"
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="create/import the dataset store")
    build.set_defaults(func=command_build)

    verify = subparsers.add_parser("verify", help="verify hashes, counts and exports")
    verify.add_argument(
        "--exports-dir",
        default=None,
        help=(
            "keep exported verification CSVs in this directory "
            "(default: temporary directory, deleted after verification)"
        ),
    )
    verify.add_argument(
        "--require-legacy-csv",
        action="store_true",
        help=(
            "fail if a legacy corpus CSV is missing (use before phase-3 "
            "retirement; after retirement the export hash is the proof)"
        ),
    )
    verify.set_defaults(func=command_verify)

    export = subparsers.add_parser("export", help="materialise one dataset version")
    export.add_argument("--dataset", required=True)
    export.add_argument("--version", type=int, default=1)
    export.add_argument("--output", required=True)
    export.set_defaults(func=command_export)

    backup = subparsers.add_parser("backup", help="VACUUM INTO + gzip the store")
    backup.set_defaults(func=command_backup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
