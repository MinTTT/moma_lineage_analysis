"""Phase 2 tests: training reads the SQLite dataset store and records lineage ids."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from moma_lineage_analysis.trajectory_cleaning import cli, dataset_store
from moma_lineage_analysis.trajectory_cleaning.data import load_trajectory_csv


def _frame(trajectories=("T1", "T2", "T3", "T4", "T5", "T6")) -> pd.DataFrame:
    rows = []
    for index, key in enumerate(trajectories):
        for step in range(5):
            rows.append(
                {
                    "trajectory_key": key,
                    "fov_cell_index": 100 * (index + 1) + step,
                    "time_h": 0.5 * step,
                    "spine_length": 6.0 - 0.4 * step + 0.05 * index,
                    "division_point": step == 2,
                    "excluded_point": step == 4,
                    "source_dataset": "synthetic",
                    "source_file": "synthetic.csv",
                    "source_file_sha256": "deadbeef" * 8,
                    "source_trajectory_key": key,
                    "source_fov_cell_index": 100 * (index + 1) + step,
                }
            )
    return pd.DataFrame(rows, columns=dataset_store.OUTPUT_COLUMNS)


def _build_store(
    tmp_path: Path,
    *,
    frame: pd.DataFrame | None = None,
    dataset_id: str = "synthetic_a",
    human_verified: bool = True,
    csv_name: str = "corpus.csv",
) -> tuple[Path, Path]:
    frame = _frame() if frame is None else frame
    csv_path = tmp_path / csv_name
    frame.to_csv(csv_path, index=False)
    store_path = tmp_path / "store.sqlite3"
    connection = dataset_store.connect(store_path)
    try:
        dataset_store.initialize(connection)
        dataset_store.import_corpus(
            connection,
            repo_root=tmp_path,
            csv_path=csv_path,
            dataset_id=dataset_id,
            kind="batch",
            label="synthetic test corpus",
            round_name="R999",
            verification_basis="synthetic",
            human_verified=human_verified,
        )
    finally:
        connection.close()
    return store_path, csv_path


def _load(store_path: Path, lineage_ids, *, require_human_verified=True):
    connection = dataset_store.connect_readonly(store_path)
    try:
        return dataset_store.load_training_frame(
            connection,
            lineage_ids,
            require_human_verified=require_human_verified,
            store_path=store_path,
        )
    finally:
        connection.close()


def test_store_frame_matches_csv_training_contract(tmp_path: Path) -> None:
    store_path, csv_path = _build_store(tmp_path)
    frame, block = _load(store_path, ["synthetic_a@v1"])
    csv_frame = load_trajectory_csv(csv_path)
    assert_frame_equal(frame, csv_frame, check_dtype=True)
    assert block["source"] == "sqlite_dataset_store"
    assert block["schema_version"] == 2
    assert block["composition"]["lineage_ids"] == ["synthetic_a@v1"]
    assert block["composition"]["rows"] == len(frame)
    assert block["composition"]["trajectories"] == 6
    lineage = block["lineages"][0]
    assert lineage["lineage_id"] == "synthetic_a@v1"
    assert lineage["content_ref"].startswith("synthetic_a@v1#")
    assert lineage["human_verified"] is True


def test_unknown_and_mismatched_lineage_ids_are_rejected(tmp_path: Path) -> None:
    store_path, _ = _build_store(tmp_path)
    connection = dataset_store.connect_readonly(store_path)
    try:
        for bad_id in ("nope@v1", "synthetic_a@v9", "synthetic_a@v1#000000"):
            try:
                dataset_store.resolve_lineage(connection, bad_id)
            except ValueError:
                pass
            else:  # pragma: no cover
                raise AssertionError(f"{bad_id} should be rejected")
        row = dataset_store.resolve_lineage(connection, "synthetic_a@v1")
        prefix = row["canonical_hash"][:12]
        dataset_store.resolve_lineage(connection, f"synthetic_a@v1#{prefix}")
    finally:
        connection.close()


def test_require_human_verified_is_enforced(tmp_path: Path) -> None:
    store_path, _ = _build_store(tmp_path, human_verified=False)
    try:
        _load(store_path, ["synthetic_a@v1"])
    except ValueError as exc:
        assert "human verified" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unverified store dataset must be rejected")
    frame, block = _load(store_path, ["synthetic_a@v1"], require_human_verified=False)
    assert len(frame) == 30
    assert block["lineages"][0]["human_verified"] is False


def test_composed_versions_collapse_identical_observations(tmp_path: Path) -> None:
    store_path, _ = _build_store(tmp_path, csv_name="first.csv")
    extended = pd.concat(
        [
            _frame(),
            _frame(("T7",)).assign(
                fov_cell_index=lambda frame: frame["fov_cell_index"] + 900,
                source_fov_cell_index=lambda frame: frame["fov_cell_index"] + 900,
            ),
        ],
        ignore_index=True,
    )
    second_csv = tmp_path / "second.csv"
    extended.to_csv(second_csv, index=False)
    connection = dataset_store.connect(store_path)
    try:
        dataset_store.import_corpus(
            connection,
            repo_root=tmp_path,
            csv_path=second_csv,
            dataset_id="synthetic_b",
            kind="batch",
            label="synthetic second",
            round_name="R999",
            verification_basis="synthetic",
            human_verified=True,
        )
    finally:
        connection.close()

    frame, block = _load(store_path, ["synthetic_a@v1", "synthetic_b@v1"])
    assert len(frame) == 35
    assert frame["trajectory_key"].nunique() == 7
    assert block["composition"]["duplicate_observation_rows_collapsed"] == 30
    assert block["composition"]["lineage_ids"] == [
        "synthetic_a@v1",
        "synthetic_b@v1",
    ]


def test_cli_training_from_store_records_lineage_metadata(tmp_path: Path) -> None:
    store_path, _ = _build_store(tmp_path)
    output_dir = tmp_path / "run"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset-lineage",
            "synthetic_a@v1",
            "--store",
            str(store_path),
            "--require-human-verified",
            "--model",
            "cnn",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    args.function(args)

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training_data"]["source"] == "sqlite_dataset_store"
    assert metrics["training_data"]["lineage_ids"] == ["synthetic_a@v1"]
    assert metrics["dataset_lineage"]["lineages"][0]["lineage_id"] == "synthetic_a@v1"
    assert (
        metrics["dataset_lineage"]["composition"]["trajectory_keys_sha256"]
        == _load(store_path, ["synthetic_a@v1"])[1]["composition"][
            "trajectory_keys_sha256"
        ]
    )
    splits = json.loads((output_dir / "splits.json").read_text(encoding="utf-8"))
    assert set(splits) == {"train", "validation", "test"}
    assert sum(len(keys) for keys in splits.values()) == 6
    assert (output_dir / "best_model.pt").is_file()
