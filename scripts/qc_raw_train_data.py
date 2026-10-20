"""Technical QC for raw curated trajectory exports before corpus building.

Checks every ``*_selected_labeled.csv`` in a directory against the input
requirements of ``trajectory_cleaning.prepare_verified_training_data``:

  1. required columns present (trajectory_key, fov_cell_index, time_h,
     spine_length, division_point, excluded_point)
  2. exactly one trajectory_key per file, matching the filename stem
  3. label columns strictly boolean-parseable (bool / 0-1 / true-false)
  4. no missing values in required columns
  5. time_h finite; spine_length finite and > 0
  6. no duplicated (trajectory_key, fov_cell_index) rows
  7. label-policy divergence: rows where division_point != effective_division_point
     (when the column exists) — these need a staging-policy decision
  8. cross-file trajectory_key collisions (same key exported twice)
  9. overlap of raw keys with an existing corpus (biological duplication risk)

Usage:
    python scripts/qc_raw_train_data.py <raw_dir> [--corpus <existing_corpus.csv>]
        [--report <qc_report.json>]

Exit code 0 = all hard gates passed (items 1-6, 8); 1 = gate failures.
Items 7 and 9 are reported as WARNINGS, not failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "trajectory_key",
    "fov_cell_index",
    "time_h",
    "spine_length",
    "division_point",
    "excluded_point",
)

BOOL_MAP = {
    "true": True, "false": False, "1": True, "0": False, "yes": False, "no": False,
}


def check_boolean_parseable(series: pd.Series) -> list[str]:
    """Return a list of problems; empty list means strictly parseable."""
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            return ["contains missing values"]
        return []
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            return ["contains missing/non-numeric values"]
        invalid = ~numeric.isin([0, 1])
        if invalid.any():
            examples = series.loc[invalid].astype(str).unique()[:5].tolist()
            return [f"non-boolean values: {examples}"]
        return []
    normalized = series.astype("string").str.strip().str.lower()
    if normalized.isna().any():
        return ["contains missing values"]
    invalid = ~normalized.isin(["true", "false", "1", "0", "yes", "no"])
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:5].tolist()
        return [f"non-boolean values: {examples}"]
    return []


def qc_file(path: Path) -> dict:
    record: dict = {"file": path.name, "failures": [], "warnings": []}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - report any parse failure as a gate failure
        record["failures"].append(f"cannot read CSV: {exc}")
        return record

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        record["failures"].append(f"missing required columns: {missing}")
        return record

    # Gate 2: single key matching the filename stem
    keys = df["trajectory_key"].dropna().unique().tolist()
    expected = path.name[: -len("_selected_labeled.csv")] if path.name.endswith(
        "_selected_labeled.csv"
    ) else path.stem
    if len(keys) != 1:
        record["failures"].append(f"expected 1 trajectory_key, found {len(keys)}: {keys[:5]}")
    elif str(keys[0]) != expected:
        record["failures"].append(
            f"trajectory_key {keys[0]!r} does not match filename stem {expected!r}"
        )

    # Gate 3: boolean labels
    for col in ("division_point", "excluded_point"):
        problems = check_boolean_parseable(df[col])
        for problem in problems:
            record["failures"].append(f"{col}: {problem}")

    # Gate 4: missing values
    for col in REQUIRED_COLUMNS:
        n_na = int(df[col].isna().sum())
        if n_na:
            record["failures"].append(f"{col}: {n_na} missing value(s)")

    # Gate 5: finite features, positive length
    time_h = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
    length = pd.to_numeric(df["spine_length"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(time_h).all():
        record["failures"].append(f"time_h: {int((~np.isfinite(time_h)).sum())} non-finite")
    if not np.isfinite(length).all():
        record["failures"].append(f"spine_length: {int((~np.isfinite(length)).sum())} non-finite")
    elif not (length > 0).all():
        record["failures"].append(f"spine_length: {int((length <= 0).sum())} non-positive")

    # Gate 6: duplicate observation keys
    n_dup = int(df.duplicated(subset=["trajectory_key", "fov_cell_index"]).sum())
    if n_dup:
        record["failures"].append(f"{n_dup} duplicated (trajectory_key, fov_cell_index) rows")

    # Stats
    dp = df["division_point"].astype(str).str.lower().isin(["true", "1"])
    ex = df["excluded_point"].astype(str).str.lower().isin(["true", "1"])
    record["rows"] = int(len(df))
    record["trajectory_key"] = str(keys[0]) if len(keys) == 1 else None
    record["division_points"] = int(dp.sum())
    record["excluded_points"] = int(ex.sum())
    record["duration_h"] = round(float(np.nanmax(time_h) - np.nanmin(time_h)), 3) if len(df) else 0.0
    record["spine_length_range"] = [
        round(float(np.nanmin(length)), 3),
        round(float(np.nanmax(length)), 3),
    ]

    # Item 7 (warning): label-policy divergence vs effective_division_point
    if "effective_division_point" in df.columns:
        edp = df["effective_division_point"].astype(str).str.lower().isin(["true", "1"])
        n_div = int((dp & ~edp).sum())
        n_edp_only = int((edp & ~dp).sum())
        record["division_not_effective"] = n_div
        record["effective_not_division"] = n_edp_only
        if n_div or n_edp_only:
            record["warnings"].append(
                f"label-policy divergence: {n_div} division_point row(s) not effective, "
                f"{n_edp_only} effective-only row(s)"
            )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", help="Directory of *_selected_labeled.csv files")
    parser.add_argument("--corpus", help="Existing corpus CSV for overlap check")
    parser.add_argument("--report", help="Write full QC results to this JSON path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*_selected_labeled.csv"))
    if not files:
        print(f"ERROR: no *_selected_labeled.csv files in {raw_dir}")
        return 1

    records = [qc_file(path) for path in files]

    # Gate 8: cross-file key collisions
    key_to_files: dict[str, list[str]] = {}
    for rec in records:
        if rec.get("trajectory_key"):
            key_to_files.setdefault(rec["trajectory_key"], []).append(rec["file"])
    collisions = {k: v for k, v in key_to_files.items() if len(v) > 1}
    for key, names in collisions.items():
        for rec in records:
            if rec["file"] in names:
                rec["failures"].append(f"trajectory_key {key!r} also exported in {names}")

    # Item 9 (warning): overlap with existing corpus
    overlap_report = None
    if args.corpus:
        corpus = pd.read_csv(args.corpus, low_memory=False)
        corpus_raw_keys = set(corpus["source_trajectory_key"].astype(str)) | set(
            corpus["trajectory_key"].astype(str)
        )
        new_keys = {rec["trajectory_key"] for rec in records if rec.get("trajectory_key")}
        overlap = sorted(new_keys & corpus_raw_keys)
        overlap_report = {"corpus": args.corpus, "overlapping_keys": overlap}
        if overlap:
            print(
                f"WARNING: {len(overlap)} new key(s) already exist in corpus "
                f"(duplication at merge time): {overlap[:10]}"
            )

    n_fail = sum(1 for r in records if r["failures"])
    n_warn = sum(1 for r in records if r["warnings"])
    totals = {
        "files": len(records),
        "files_failed": n_fail,
        "files_with_warnings": n_warn,
        "rows": sum(r.get("rows", 0) for r in records),
        "trajectories": len(key_to_files),
        "division_points": sum(r.get("division_points", 0) for r in records),
        "excluded_points": sum(r.get("excluded_points", 0) for r in records),
        "division_not_effective": sum(r.get("division_not_effective", 0) for r in records),
        "effective_not_division": sum(r.get("effective_not_division", 0) for r in records),
    }

    print(json.dumps(totals, indent=2))
    for rec in records:
        for failure in rec["failures"]:
            print(f"FAIL  {rec['file']}: {failure}")
        for warning in rec["warnings"]:
            print(f"WARN  {rec['file']}: {warning}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {"totals": totals, "overlap": overlap_report, "files": records},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"QC report written: {report_path}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
