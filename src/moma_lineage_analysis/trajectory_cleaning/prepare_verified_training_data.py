"""Build a provenance-preserving training CSV from human-verified sources.

Only the canonical ``division_point`` and ``excluded_point`` columns are used
as targets.  Prediction, AI, effective-label, and division-audit columns from
source files are deliberately not copied to the output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
NAMESPACE_DELIMITER = "::"
REQUIRED_COLUMNS = (
    "trajectory_key",
    "fov_cell_index",
    "time_h",
    "spine_length",
    "division_point",
    "excluded_point",
)
TARGET_COLUMNS = ("division_point", "excluded_point")
OUTPUT_COLUMNS = (
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
)


@dataclass(frozen=True)
class PreparedVerifiedDataset:
    """The prepared dataframe and the manifest describing its provenance."""

    dataframe: pd.DataFrame
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_verification_basis(value: str) -> str:
    basis = str(value).strip()
    if not basis:
        raise ValueError(
            "verification_basis is required; it is the explicit attestation that "
            "the canonical labels were human verified"
        )
    return basis


def _parse_boolean_labels(series: pd.Series, column: str) -> pd.Series:
    """Parse explicit boolean encodings without using Python truthiness."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{column!r} contains missing values")
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{column!r} contains missing values")
        invalid = ~numeric.isin([0, 1])
        if invalid.any():
            examples = series.loc[invalid].astype(str).unique()[:5].tolist()
            raise ValueError(f"{column!r} contains non-boolean values: {examples}")
        return numeric.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        examples = series.loc[parsed.isna()].astype(str).unique()[:5].tolist()
        raise ValueError(f"{column!r} contains non-boolean values: {examples}")
    return parsed.astype(bool)


def _derive_dataset_ids(paths: Sequence[Path], hashes: Sequence[str]) -> list[str]:
    bases = [path.stem.strip().replace(NAMESPACE_DELIMITER, "__") for path in paths]
    if any(not value for value in bases):
        raise ValueError("Every source filename must have a non-empty stem")

    counts: dict[str, int] = defaultdict(int)
    for value in bases:
        counts[value] += 1
    # If stems collide, suffix every member rather than privileging input order.
    return [
        f"{base}__{digest[:12]}" if counts[base] > 1 else base
        for base, digest in zip(bases, hashes)
    ]


def _validate_dataset_ids(
    dataset_ids: Sequence[str] | None,
    paths: Sequence[Path],
    hashes: Sequence[str],
) -> list[str]:
    if dataset_ids is None:
        result = _derive_dataset_ids(paths, hashes)
    else:
        if len(dataset_ids) != len(paths):
            raise ValueError(
                "dataset_ids must be omitted or contain exactly one ID per source"
            )
        result = [str(value).strip() for value in dataset_ids]
        if any(not value for value in result):
            raise ValueError("dataset_ids cannot contain blank values")
        if any(NAMESPACE_DELIMITER in value for value in result):
            raise ValueError(
                f"dataset_ids cannot contain the namespace delimiter "
                f"{NAMESPACE_DELIMITER!r}"
            )
    if len(set(result)) != len(result):
        raise ValueError(f"dataset_ids must be unique; received {result}")
    return result


def _validate_supersessions(
    supersessions: Sequence[str | tuple[str, str]] | None,
    dataset_ids: Sequence[str],
) -> list[tuple[str, str]]:
    """Validate explicit older=preferred trajectory-source precedence rules."""

    parsed: list[tuple[str, str]] = []
    for value in supersessions or ():
        if isinstance(value, str):
            if value.count("=") != 1:
                raise ValueError(
                    "supersession rules must use the form <superseded>=<preferred>"
                )
            superseded, preferred = (part.strip() for part in value.split("=", 1))
        else:
            if len(value) != 2:
                raise ValueError("supersession tuples must contain exactly two IDs")
            superseded, preferred = (str(part).strip() for part in value)
        if not superseded or not preferred:
            raise ValueError("supersession dataset IDs cannot be blank")
        if superseded == preferred:
            raise ValueError("a dataset cannot supersede itself")
        missing = sorted({superseded, preferred}.difference(dataset_ids))
        if missing:
            raise ValueError(
                f"supersession rules reference unknown dataset IDs: {missing}"
            )
        parsed.append((superseded, preferred))

    superseded_ids = [superseded for superseded, _ in parsed]
    if len(set(superseded_ids)) != len(superseded_ids):
        raise ValueError("each dataset can be superseded by at most one source")
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate supersession rules are not allowed")

    graph = dict(parsed)
    for start in graph:
        visited: set[str] = set()
        current = start
        while current in graph:
            if current in visited:
                raise ValueError("supersession rules cannot contain a cycle")
            visited.add(current)
            current = graph[current]
    return parsed


def _label_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    division = dataframe["division_point"].astype(bool)
    excluded = dataframe["excluded_point"].astype(bool)
    return {
        "division_point": int(division.sum()),
        "excluded_point": int(excluded.sum()),
        "division_and_excluded_overlap": int((division & excluded).sum()),
    }


def _ignored_label_like_columns(columns: Sequence[Any]) -> list[str]:
    ignored: list[str] = []
    canonical = set(TARGET_COLUMNS)
    for value in columns:
        column = str(value)
        lowered = column.lower()
        if lowered in canonical:
            continue
        if (
            lowered.startswith(("pred_", "ai_", "effective_"))
            or "division" in lowered
            or "exclusion" in lowered
            or "excluded" in lowered
        ):
            ignored.append(column)
    return sorted(set(ignored))


def _json_value(value: Any) -> Any:
    """Represent a raw invalid value without emitting non-standard JSON NaN."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return str(value)


def _validate_source_identifiers(dataframe: pd.DataFrame, source_file: Path) -> None:
    for column in ("trajectory_key", "fov_cell_index"):
        if dataframe[column].isna().any():
            raise ValueError(f"{source_file}: {column!r} contains missing values")
    empty_trajectory = dataframe["trajectory_key"].astype(str).str.strip().eq("")
    if empty_trajectory.any():
        raise ValueError(f"{source_file}: 'trajectory_key' contains blank values")


def _validate_unique_coordinates(dataframe: pd.DataFrame, source_file: Path) -> None:
    for trajectory_key, group in dataframe.groupby(
        "source_trajectory_key", sort=False, dropna=False
    ):
        duplicate_ids = group["source_fov_cell_index"].duplicated(keep=False)
        if duplicate_ids.any():
            examples = (
                group.loc[duplicate_ids, "source_fov_cell_index"]
                .astype(str)
                .unique()[:5]
                .tolist()
            )
            raise ValueError(
                f"{source_file}: trajectory {trajectory_key!r} has duplicate "
                f"fov_cell_index values: {examples}"
            )
        duplicate_times = group["time_h"].duplicated(keep=False)
        if duplicate_times.any():
            examples = (
                group.loc[duplicate_times, "time_h"]
                .astype(str)
                .unique()[:5]
                .tolist()
            )
            raise ValueError(
                f"{source_file}: trajectory {trajectory_key!r} has duplicate "
                f"time_h values: {examples}"
            )


def build_verified_training_dataset(
    sources: Sequence[str | Path],
    *,
    verification_basis: str,
    dataset_ids: Sequence[str] | None = None,
    supersessions: Sequence[str | tuple[str, str]] | None = None,
) -> PreparedVerifiedDataset:
    """Validate and combine explicitly listed human-verified CSV files.

    ``verification_basis`` is mandatory because the files themselves cannot
    prove whether a person reviewed their canonical labels.  Rows with a
    nonnumeric/nonfinite time, or a nonnumeric/nonfinite/nonpositive length,
    are omitted and described in the returned manifest.  Other schema and
    uniqueness problems raise ``ValueError``.
    """

    basis = _require_verification_basis(verification_basis)
    paths = [Path(value).expanduser().resolve() for value in sources]
    if not paths:
        raise ValueError("At least one explicit source CSV is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Source CSV does not exist: {path}")

    hashes = [_sha256(path) for path in paths]
    paths_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path, digest in zip(paths, hashes):
        paths_by_hash[digest].append(path)
    duplicates = {key: value for key, value in paths_by_hash.items() if len(value) > 1}
    if duplicates:
        digest, duplicate_paths = next(iter(duplicates.items()))
        rendered = ", ".join(str(path) for path in duplicate_paths)
        raise ValueError(
            "Duplicate source content would be double-counted: "
            f"SHA-256 {digest} is shared by {rendered}"
        )

    resolved_dataset_ids = _validate_dataset_ids(dataset_ids, paths, hashes)
    resolved_supersessions = _validate_supersessions(
        supersessions, resolved_dataset_ids
    )
    path_by_dataset_id = dict(zip(resolved_dataset_ids, paths))
    supersession_by_dataset: dict[str, tuple[str, set[str]]] = {}
    for superseded, preferred in resolved_supersessions:
        preferred_path = path_by_dataset_id[preferred]
        try:
            preferred_keys = pd.read_csv(
                preferred_path, usecols=["trajectory_key"]
            )["trajectory_key"]
        except ValueError as error:
            raise ValueError(
                f"{preferred_path}: missing required columns: ['trajectory_key']"
            ) from error
        supersession_by_dataset[superseded] = (
            preferred,
            set(preferred_keys.dropna().astype(str)),
        )
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    frames: list[pd.DataFrame] = []
    source_summaries: list[dict[str, Any]] = []
    invalid_details: list[dict[str, Any]] = []
    raw_key_sources: dict[str, set[str]] = defaultdict(set)

    for path, digest, dataset_id in zip(paths, hashes, resolved_dataset_ids):
        source = pd.read_csv(path)
        missing = sorted(set(REQUIRED_COLUMNS).difference(source.columns))
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")
        if source.empty:
            raise ValueError(f"{path}: source CSV is empty")

        _validate_source_identifiers(source, path)
        source = source.copy()
        for label_column in TARGET_COLUMNS:
            source[label_column] = _parse_boolean_labels(
                source[label_column], label_column
            )

        raw_trajectory_key = source["trajectory_key"].astype(str)
        eligible_rows = np.ones(len(source), dtype=bool)
        superseded_by: str | None = None
        superseded_trajectory_keys: list[str] = []
        if dataset_id in supersession_by_dataset:
            superseded_by, preferred_keys = supersession_by_dataset[dataset_id]
            superseded_trajectory_keys = sorted(
                set(raw_trajectory_key.unique()).intersection(preferred_keys)
            )
            eligible_rows = ~raw_trajectory_key.isin(
                superseded_trajectory_keys
            ).to_numpy()
        for key in raw_trajectory_key.loc[eligible_rows].unique():
            raw_key_sources[key].add(dataset_id)

        numeric_time = pd.to_numeric(source["time_h"], errors="coerce")
        numeric_length = pd.to_numeric(source["spine_length"], errors="coerce")
        time_values = numeric_time.to_numpy(dtype=np.float64)
        length_values = numeric_length.to_numpy(dtype=np.float64)
        valid_time = np.isfinite(time_values)
        finite_length = np.isfinite(length_values)
        positive_length = length_values > 0
        valid_features = valid_time & finite_length & positive_length

        retained_mask = eligible_rows & valid_features
        invalid_positions = np.flatnonzero(eligible_rows & ~valid_features)
        source_invalid_details: list[dict[str, Any]] = []
        for position in invalid_positions:
            reasons: list[str] = []
            if not valid_time[position]:
                reasons.append("time_h_not_finite_numeric")
            if not finite_length[position]:
                reasons.append("spine_length_not_finite_numeric")
            elif not positive_length[position]:
                reasons.append("spine_length_not_positive")
            detail = {
                "source_dataset": dataset_id,
                "source_file": str(path),
                "source_row_index": int(position),
                "csv_line_number": int(position) + 2,
                "source_trajectory_key": str(raw_trajectory_key.iloc[position]),
                "source_fov_cell_index": _json_value(
                    source["fov_cell_index"].iloc[position]
                ),
                "raw_time_h": _json_value(source["time_h"].iloc[position]),
                "raw_spine_length": _json_value(
                    source["spine_length"].iloc[position]
                ),
                "reasons": reasons,
            }
            source_invalid_details.append(detail)
            invalid_details.append(detail)

        retained_source = source.loc[retained_mask].copy()
        if retained_source.empty and eligible_rows.any():
            raise ValueError(f"{path}: no valid feature rows remain")
        retained_source["time_h"] = numeric_time.loc[retained_mask].astype(float)
        retained_source["spine_length"] = numeric_length.loc[retained_mask].astype(
            float
        )
        retained_source["source_trajectory_key"] = raw_trajectory_key.loc[
            retained_mask
        ].to_numpy()
        retained_source["source_fov_cell_index"] = retained_source[
            "fov_cell_index"
        ]
        retained_source["trajectory_key"] = (
            dataset_id
            + NAMESPACE_DELIMITER
            + retained_source["source_trajectory_key"].astype(str)
        )
        retained_source["source_dataset"] = dataset_id
        retained_source["source_file"] = str(path)
        retained_source["source_file_sha256"] = digest
        retained_source = retained_source.loc[:, OUTPUT_COLUMNS]

        _validate_unique_coordinates(retained_source, path)
        if not retained_source.empty:
            frames.append(retained_source)
        source_summaries.append(
            {
                "dataset_id": dataset_id,
                "source_file": str(path),
                "source_file_sha256": digest,
                "raw_rows": int(len(source)),
                "eligible_rows_after_supersession": int(eligible_rows.sum()),
                "retained_rows": int(len(retained_source)),
                "omitted_superseded_rows": int((~eligible_rows).sum()),
                "omitted_superseded_trajectories": len(
                    superseded_trajectory_keys
                ),
                "omitted_invalid_feature_rows": int(len(source_invalid_details)),
                "raw_trajectories": int(raw_trajectory_key.nunique()),
                "eligible_trajectories_after_supersession": int(
                    raw_trajectory_key.loc[eligible_rows].nunique()
                ),
                "retained_trajectories": int(
                    retained_source["trajectory_key"].nunique()
                ),
                "raw_label_counts": _label_counts(source),
                "eligible_label_counts_after_supersession": _label_counts(
                    source.loc[eligible_rows]
                ),
                "retained_label_counts": _label_counts(retained_source),
                "superseded_by": superseded_by,
                "superseded_trajectory_keys": superseded_trajectory_keys,
                "ignored_label_or_audit_columns": _ignored_label_like_columns(
                    source.columns
                ),
                "invalid_feature_rows": source_invalid_details,
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    if list(combined.columns) != list(OUTPUT_COLUMNS):
        raise AssertionError("Prepared output schema changed unexpectedly")
    forbidden_output = _ignored_label_like_columns(combined.columns)
    if forbidden_output:
        raise AssertionError(
            f"Prepared output contains forbidden label/audit columns: {forbidden_output}"
        )

    collided_keys = sorted(
        key for key, source_ids in raw_key_sources.items() if len(source_ids) > 1
    )
    short_trajectories = (
        combined.groupby("trajectory_key", sort=False)
        .size()
        .loc[lambda values: values < 2]
    )
    supersession_rules = [
        {
            "superseded_dataset": item["dataset_id"],
            "preferred_dataset": item["superseded_by"],
            "omitted_rows": item["omitted_superseded_rows"],
            "omitted_trajectories": item["omitted_superseded_trajectories"],
            "trajectory_keys": item["superseded_trajectory_keys"],
        }
        for item in source_summaries
        if item["superseded_by"] is not None
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "human_verified": True,
        "verification_basis": basis,
        "canonical_target_columns": list(TARGET_COLUMNS),
        "feature_columns": ["time_h", "spine_length"],
        "output_columns": list(OUTPUT_COLUMNS),
        "label_policy": (
            "Only canonical final division_point and excluded_point values are "
            "targets; AI predictions, effective labels, and audit fields are ignored."
        ),
        "namespace": {
            "delimiter": NAMESPACE_DELIMITER,
            "format": "<source_dataset>::<source_trajectory_key>",
            "raw_keys_present_in_multiple_sources": len(collided_keys),
            "colliding_raw_keys": collided_keys,
            "collision_handling": (
                "After explicit supersession, remaining raw-key collisions are "
                "resolved by prefixing every raw key with its unique dataset ID."
            ),
        },
        "supersession": {
            "policy": (
                "For each explicit older=preferred rule, an older trajectory is "
                "omitted in full when its raw key exists in the preferred source."
            ),
            "rules": supersession_rules,
        },
        "sources": source_summaries,
        "summary": {
            "source_files": len(paths),
            "raw_rows": int(sum(item["raw_rows"] for item in source_summaries)),
            "retained_rows": int(len(combined)),
            "omitted_superseded_rows": int(
                sum(item["omitted_superseded_rows"] for item in source_summaries)
            ),
            "omitted_superseded_trajectories": int(
                sum(item["omitted_superseded_trajectories"] for item in source_summaries)
            ),
            "omitted_invalid_feature_rows": int(len(invalid_details)),
            "retained_trajectories": int(combined["trajectory_key"].nunique()),
            "retained_label_counts": _label_counts(combined),
            "trajectories_with_fewer_than_two_retained_points": short_trajectories.index.astype(
                str
            ).tolist(),
        },
        "invalid_feature_rows": invalid_details,
    }
    return PreparedVerifiedDataset(dataframe=combined, manifest=manifest)


def prepare_verified_training_data(
    sources: Sequence[str | Path],
    *,
    output_csv: str | Path,
    manifest_path: str | Path,
    verification_basis: str,
    dataset_ids: Sequence[str] | None = None,
    supersessions: Sequence[str | tuple[str, str]] | None = None,
) -> PreparedVerifiedDataset:
    """Build a verified dataset, write its CSV and JSON manifest, and return it."""

    source_paths = [Path(value).expanduser().resolve() for value in sources]
    output_path = Path(output_csv).expanduser().resolve()
    manifest_output = Path(manifest_path).expanduser().resolve()
    if output_path == manifest_output:
        raise ValueError("output_csv and manifest_path must be different files")
    for source_path in source_paths:
        if output_path == source_path or manifest_output == source_path:
            raise ValueError("Output paths cannot overwrite a source CSV")

    prepared = build_verified_training_dataset(
        source_paths,
        verification_basis=verification_basis,
        dataset_ids=dataset_ids,
        supersessions=supersessions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    csv_temporary = output_path.with_name(f".{output_path.name}.tmp")
    manifest_temporary = manifest_output.with_name(f".{manifest_output.name}.tmp")
    try:
        prepared.dataframe.to_csv(csv_temporary, index=False)
        csv_temporary.replace(output_path)
        manifest = dict(prepared.manifest)
        manifest["output"] = {
            "training_csv": str(output_path),
            "training_csv_sha256": _sha256(output_path),
            "manifest": str(manifest_output),
        }
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_temporary.replace(manifest_output)
    finally:
        csv_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)

    return PreparedVerifiedDataset(dataframe=prepared.dataframe, manifest=manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine explicit human-verified trajectory CSVs while ignoring all "
            "AI/prediction/audit label columns."
        )
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        required=True,
        help="Explicit source CSV path; repeat once per source (globs are not expanded).",
    )
    parser.add_argument(
        "--dataset-id",
        dest="dataset_ids",
        action="append",
        help=(
            "Optional source namespace ID; if used, repeat in the same order and "
            "number as --source. Filename stems are used when omitted."
        ),
    )
    parser.add_argument(
        "--supersede",
        dest="supersessions",
        action="append",
        help=(
            "Optional trajectory-level source precedence in the form "
            "<superseded_dataset_id>=<preferred_dataset_id>. Repeat for "
            "multiple explicit relationships."
        ),
    )
    parser.add_argument("--output-csv", required=True, help="Combined training CSV")
    parser.add_argument("--manifest", required=True, help="JSON provenance manifest")
    parser.add_argument(
        "--verification-basis",
        required=True,
        help=(
            "Required human-verification attestation, for example "
            "'user_attested_2026-07-17'."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prepared = prepare_verified_training_data(
        args.sources,
        output_csv=args.output_csv,
        manifest_path=args.manifest,
        verification_basis=args.verification_basis,
        dataset_ids=args.dataset_ids,
        supersessions=args.supersessions,
    )
    summary = prepared.manifest["summary"]
    print(
        json.dumps(
            {
                "training_csv": prepared.manifest["output"]["training_csv"],
                "manifest": prepared.manifest["output"]["manifest"],
                **summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
