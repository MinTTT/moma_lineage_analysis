"""Add a brand-new attested curated batch to a store that holds review data.

Historical CSV build/import is disabled once the training-review app has
written review data (it could restore permanently deleted trajectories).  This
admin path admits only a NEW batch corpus: a new dataset id and key namespace,
no stored or deleted trajectory keys and no measurement-identical trajectory.
It is review-revision CAS-guarded, takes an online SQLite backup and records an
`import_batch` review event.  Metadata comes from the batch's entry in
datasets/manifest.json, so write that entry first (Workflow A step 8).

Usage:
    python scripts/import_batch.py --dataset-dir verified_20261109 \
        --expected-revision 567 \
        --reason "user: load new curated raw exports"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "datasets" / "store" / "lineage.sqlite3"
SYS_PATH = str(REPO_ROOT / "src")
if SYS_PATH not in sys.path:
    sys.path.insert(0, SYS_PATH)

from moma_lineage_analysis.trainset_review import store  # noqa: E402
from moma_lineage_analysis.trajectory_cleaning import dataset_store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Batch folder under datasets/ that has a manifest.json entry.",
    )
    parser.add_argument(
        "--expected-revision",
        type=int,
        required=True,
        help="Current global review revision (CAS precondition).",
    )
    parser.add_argument("--reason", required=True, help="Why the batch is added.")
    parser.add_argument("--actor", default="user")
    args = parser.parse_args(argv)

    entries = dataset_store.load_manifest_entries(REPO_ROOT)
    entry = entries.get(args.dataset_dir)
    if entry is None:
        raise SystemExit(f"no datasets/manifest.json entry for {args.dataset_dir}")
    if entry.get("store_removed") or entry.get("retired"):
        raise SystemExit(f"{args.dataset_dir} is retired/removed; not a new batch")
    if str(entry.get("kind", "batch")) != "batch":
        raise SystemExit(f"{args.dataset_dir} is not a batch corpus")
    result = store.import_batch(
        args.db,
        REPO_ROOT / "datasets" / str(entry["path"]),
        args.expected_revision,
        repo_root=REPO_ROOT,
        dataset_id=str(entry["id"]),
        label=str(entry["role"]),
        round_name=entry.get("round"),
        verification_basis=str(entry["verification_basis"]),
        human_verified=bool(entry.get("human_verified", False)),
        reason=args.reason,
        actor=args.actor,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
