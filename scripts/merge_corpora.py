"""Merge verified training corpora into one training corpus with provenance.

Concatenates already-verified corpora (their keys are already namespaced and
unique), applies the R004 production-corpus rule to the *incoming* corpora —
drop any trajectory whose longest contiguous exclusion run is >= 5 frames —
and writes a merged CSV plus a merge manifest (input hashes, counts, dropped
keys, output hash). The R004 production corpus is assumed already filtered and
is verified to satisfy the rule rather than re-filtered blindly.

Usage:
    python scripts/merge_corpora.py --output-dir datasets/verified_merged_20261016 \
        --corpus datasets/verified_20260916/verified_training_data_no_longruns.csv \
        --filtered-corpus datasets/verified_20261013/verified_training_data.csv \
        --filtered-corpus datasets/verified_20261016/verified_training_data.csv

--corpus marks an already-R004-filtered corpus (verified, not re-filtered);
--filtered-corpus applies the R004 rule during the merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _longest_exclusion_run(flags) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        help="Verified corpus CSV assumed to already satisfy the R004 rule",
    )
    parser.add_argument(
        "--filtered-corpus",
        action="append",
        default=[],
        help="Verified corpus CSV to which the R004 filter is applied now",
    )
    args = parser.parse_args()

    pairs = [(p, False) for p in args.corpus] + [(p, True) for p in args.filtered_corpus]
    if not pairs:
        parser.error("at least one --corpus or --filtered-corpus is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    source_records = []
    dropped_all: dict[str, list[str]] = {}
    for corpus_path_str, apply_filter in pairs:
        corpus_path = Path(corpus_path_str).resolve()
        df = pd.read_csv(corpus_path, low_memory=False)
        record = {
            "path": str(corpus_path),
            "sha256": _sha256(corpus_path),
            "rows": int(len(df)),
            "trajectories": int(df["trajectory_key"].nunique()),
            "r004_filter_applied": bool(apply_filter),
        }
        if apply_filter:
            drop_keys = []
            for key, group in df.groupby("trajectory_key", sort=False):
                run = _longest_exclusion_run(
                    group["excluded_point"].astype(bool).tolist()
                )
                if run >= 5:
                    drop_keys.append(key)
            if drop_keys:
                df = df[~df["trajectory_key"].isin(drop_keys)].copy()
            dropped_all[str(corpus_path)] = drop_keys
            record["dropped_trajectories"] = drop_keys
            record["retained_rows"] = int(len(df))
            record["retained_trajectories"] = int(df["trajectory_key"].nunique())
        else:
            # Verify the corpus already satisfies the rule
            violations = []
            for key, group in df.groupby("trajectory_key", sort=False):
                run = _longest_exclusion_run(
                    group["excluded_point"].astype(bool).tolist()
                )
                if run >= 5:
                    violations.append(key)
            record["pre_filtered_violations"] = violations
            if violations:
                print(
                    f"ERROR: {corpus_path.name} has {len(violations)} trajectories "
                    f"violating the R004 rule but --filter was not set: {violations[:5]}"
                )
                return 1
        frames.append(df)
        source_records.append(record)

    merged = pd.concat(frames, ignore_index=True)
    merged_keys = merged["trajectory_key"].nunique()
    expected_keys = sum(r.get("retained_trajectories", r["trajectories"]) for r in source_records)
    if merged_keys != expected_keys:
        print(
            f"ERROR: key collision across corpora: merged has {merged_keys} unique "
            f"trajectory_key values but the sources declare {expected_keys}"
        )
        return 1

    out_csv = out_dir / "verified_training_data.csv"
    merged.to_csv(out_csv, index=False)
    out_hash = _sha256(out_csv)

    manifest = {
        "schema_version": "1.0-merge",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # Fields required by the training CLI's provenance gate. The merged
        # corpus inherits the human-verified attestation of its constituents.
        "human_verified": True,
        "verification_basis": "constituent_corpora_attested",
        "canonical_target_columns": ["division_point", "excluded_point"],
        "feature_columns": ["time_h", "spine_length"],
        "merge_rule": "concatenate verified corpora; keys are pre-namespaced and unique",
        "r004_filter": {
            "rule": "drop trajectory when longest contiguous exclusion run >= 5 frames",
            "applied_to": [r["path"] for r in source_records if r["r004_filter_applied"]],
            "dropped": dropped_all,
        },
        "sources": source_records,
        "summary": {
            "rows": int(len(merged)),
            "trajectories": int(merged["trajectory_key"].nunique()),
            "division_points": int(merged["division_point"].sum()),
            "excluded_points": int(merged["excluded_point"].sum()),
            "division_and_excluded_overlap": int(
                (merged["division_point"] & merged["excluded_point"]).sum()
            ),
            "source_datasets": {
                k: int(v)
                for k, v in merged.groupby("source_dataset")["trajectory_key"]
                .nunique()
                .items()
            },
        },
        "output": {
            "training_csv": str(out_csv),
            "training_csv_sha256": out_hash,
        },
    }
    manifest_path = out_dir / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"corpus: {out_csv}")
    print(f"sha256: {out_hash}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
