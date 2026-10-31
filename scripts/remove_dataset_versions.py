"""Remove user-identified duplicate dataset versions from the training store.

Dataset-level counterpart of a per-trajectory delete in the training-review app:
whole dataset versions are removed only when every one of their trajectories is
still referenced by another version, so no observation loses all memberships.
The removal takes an online SQLite backup, bumps the global review revision and
records a review event.

Usage:
    python scripts/remove_dataset_versions.py --expected-revision 561 \
        --reason "user: merged corpora duplicate the five batch corpora" \
        --lineage-id verified_merged_20261016@v1 \
        --lineage-id verified_merged_20261017@v1 \
        --lineage-id verified_merged_20261029@v1
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument(
        "--lineage-id",
        action="append",
        required=True,
        help="Dataset lineage id to remove (repeat for several versions).",
    )
    parser.add_argument(
        "--expected-revision",
        type=int,
        required=True,
        help="Current global review revision (CAS precondition).",
    )
    parser.add_argument("--reason", required=True, help="Why the versions are removed.")
    parser.add_argument("--actor", default="user")
    args = parser.parse_args(argv)
    result = store.remove_dataset_versions(
        args.db,
        args.lineage_id,
        args.expected_revision,
        reason=args.reason,
        actor=args.actor,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
