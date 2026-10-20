"""Stage raw curated exports into canonical label form for corpus building.

Applies the contract's R002 label policy (registered in
METHODS_TRAJECTORY_CLEANING_MODEL.md as `R002_LABEL_POLICY`) to each raw
``*_selected_labeled.csv`` export:

  - rows whose raw division label is a ``coalesced_previous`` artifact have
    ``division_point`` dropped (the event is already labeled at the previous
    time point);
  - ``effective_division_point``-only rows (no raw division label) are kept as
    divisions — the effective boundary is the audited label;
  - ``needs_review`` and ``shifted_previous`` rows keep their raw
    ``division_point``;
  - ``excluded_point`` is carried through unchanged.

The division/exclusion conflict resolution is recomputed per file with the
packaged resolver (``resolve_division_exclusion_conflicts``), so no audit
columns are required in the export. Also reports, per trajectory, the longest
contiguous exclusion run so the R004 production-corpus filter (>= 5 frames)
impact can be assessed before merging.

Two division-label modes are supported (--division-mode):

  - ``r002_policy``: the contract's historical R002 staging policy described
    above (coalesced dropped, effective-only added, needs_review/shifted kept);
  - ``effective``: ``effective_division_point`` is the final true division
    label — division_point is set exactly to the resolver's effective
    boundaries. Excluded shifted sources and unresolved needs_review sources
    lose their raw label (flanking cycles merge).

An optional --rename-map JSON ({"old_key": "new_key"}) rewrites trajectory_key
in the staged output, e.g. to date-prefix keys that would otherwise duplicate
trajectories already present in another corpus. Raw exports are never
modified.

Usage:
    python scripts/stage_verified_labels.py <raw_dir> <staged_dir> [--report <json>]
        [--division-mode r002_policy|effective] [--rename-map <json>]

Writes one staged CSV per source (same filename) containing exactly:
trajectory_key, fov_cell_index, time_h, spine_length, division_point, excluded_point
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from moma_lineage_analysis.trajectory_cleaning.streamlit_api import (
    resolve_division_exclusion_conflicts,
)

OUTPUT_COLUMNS = (
    "trajectory_key",
    "fov_cell_index",
    "time_h",
    "spine_length",
    "division_point",
    "excluded_point",
)


def longest_exclusion_run(flags: pd.Series) -> int:
    """Length of the longest contiguous run of True in a boolean series."""
    longest = current = 0
    for flag in flags.astype(bool).tolist():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def stage_file(source: Path, staged_dir: Path, division_mode: str, rename_map: dict) -> dict:
    df = pd.read_csv(source, low_memory=False)
    df = df.sort_values("time_h", kind="stable").reset_index(drop=True)

    division_ids = df.loc[df["division_point"].astype(bool), "fov_cell_index"]
    excluded_ids = df.loc[df["excluded_point"].astype(bool), "fov_cell_index"]
    resolution = resolve_division_exclusion_conflicts(
        df[["fov_cell_index", "time_h", "spine_length"]],
        division_ids,
        excluded_ids,
        division_drop_ratio=0.75,
    )

    decisions = resolution.decisions
    status_counts = (
        decisions["division_resolution"].value_counts().to_dict()
        if not decisions.empty
        else {}
    )
    effective_ids = set(resolution.effective_division_indices)

    raw_dp = df["fov_cell_index"].isin(set(division_ids))
    edp = df["fov_cell_index"].isin(effective_ids)
    if division_mode == "effective":
        # effective_division_point is the final true division label.
        resolved_dp = edp
    else:  # r002_policy
        coalesced_ids = set(
            decisions.loc[
                decisions["division_resolution"] == "coalesced_previous",
                "source_index",
            ]
        )
        resolved_dp = raw_dp.copy()
        resolved_dp[df["fov_cell_index"].isin(coalesced_ids)] = False
        resolved_dp[~raw_dp & edp] = True

    staged = df.loc[:, list(OUTPUT_COLUMNS)].copy()
    original_key = str(staged["trajectory_key"].iloc[0])
    renamed_to = rename_map.get(original_key)
    if renamed_to is not None:
        staged["trajectory_key"] = renamed_to
    staged["division_point"] = resolved_dp.to_numpy(dtype=bool)
    staged["excluded_point"] = df["excluded_point"].astype(bool).to_numpy()

    out_path = staged_dir / source.name
    staged.to_csv(out_path, index=False)

    return {
        "file": source.name,
        "staged": str(out_path),
        "rows": int(len(staged)),
        "original_trajectory_key": original_key,
        "renamed_to": renamed_to,
        "division_mode": division_mode,
        "resolution_status_counts": {k: int(v) for k, v in status_counts.items()},
        "division_labels_removed": int((raw_dp & ~resolved_dp).sum()),
        "division_labels_added": int((~raw_dp & resolved_dp).sum()),
        "division_points_final": int(resolved_dp.sum()),
        "excluded_points": int(staged["excluded_point"].sum()),
        "longest_exclusion_run_frames": longest_exclusion_run(
            staged["excluded_point"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir")
    parser.add_argument("staged_dir")
    parser.add_argument("--report", help="Write per-file staging report JSON")
    parser.add_argument(
        "--division-mode",
        choices=("r002_policy", "effective"),
        default="r002_policy",
        help=(
            "r002_policy = historical staging policy; effective = "
            "effective_division_point is the final true division label"
        ),
    )
    parser.add_argument(
        "--rename-map",
        help="JSON object mapping original trajectory_key to the staged key",
    )
    args = parser.parse_args()

    rename_map = {}
    if args.rename_map:
        rename_map = json.loads(Path(args.rename_map).read_text(encoding="utf-8"))
        if not isinstance(rename_map, dict):
            raise ValueError("rename map must be a JSON object of old->new keys")

    raw_dir = Path(args.raw_dir)
    staged_dir = Path(args.staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*_selected_labeled.csv"))
    if not files:
        print(f"ERROR: no *_selected_labeled.csv files in {raw_dir}")
        return 1

    records = [
        stage_file(path, staged_dir, args.division_mode, rename_map)
        for path in files
    ]
    totals = {
        "files": len(records),
        "division_mode": args.division_mode,
        "renamed_keys": {
            r["original_trajectory_key"]: r["renamed_to"]
            for r in records
            if r["renamed_to"] is not None
        },
        "rows": sum(r["rows"] for r in records),
        "division_points_final": sum(r["division_points_final"] for r in records),
        "division_labels_removed": sum(r["division_labels_removed"] for r in records),
        "division_labels_added": sum(r["division_labels_added"] for r in records),
        "excluded_points": sum(r["excluded_points"] for r in records),
        "status_counts": {},
        "r004_filter_would_drop": [
            r["file"]
            for r in records
            if r["longest_exclusion_run_frames"] >= 5
        ],
    }
    for rec in records:
        for status, count in rec["resolution_status_counts"].items():
            totals["status_counts"][status] = (
                totals["status_counts"].get(status, 0) + count
            )

    print(json.dumps(totals, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"totals": totals, "files": records}, indent=2),
            encoding="utf-8",
        )
        print(f"Staging report written: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
