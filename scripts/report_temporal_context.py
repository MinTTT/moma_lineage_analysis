"""Read-only R016 reporting: figures, validation diagnostics, control evaluation.

summary is CPU-only. diagnostics/control/all require an explicit --device;
normally use cuda only after the Lead authorizes GPU inference. No training,
model selection, checkpoint mutation, or dataset writes are performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/context_20261029"
ARCHITECTURES = ("cnn", "resnet")
LABELS = ("division_point", "excluded_point")
SEEDS = (42, 43, 44)
POPULATIONS = ("test", "common_historical_test", "new_batch_test")
SCRIPT_PATH = Path(__file__).resolve()
COLORS = {"cnn": "#0072B2", "resnet": "#D55E00"}
MARKERS = {42: "o", 43: "s", 44: "^"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return relative(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mean_sd(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    require(values, "Cannot summarize an empty sequence")
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def point_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {"mean_f1": (float(metrics["division_event"]["f1"])
                        + float(metrics["point"]["excluded_point"]["f1"])) / 2,
            "division_event_f1": float(metrics["division_event"]["f1"]),
            "exclusion_f1": float(metrics["point"]["excluded_point"]["f1"]),
            "exclusion_ap": float(metrics["point"]["excluded_point"]["average_precision"])}


def output_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


class Artifacts:
    """Verify immutable inputs; a lock is evidence, never a new ranking API."""
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.protocol_path = run_dir / "protocol.json"
        self.protocol = read_json(self.protocol_path)
        self.protocol_hash = sha256_file(self.protocol_path)
        self.splits_path = ROOT / self.protocol["dataset"]["frozen_splits"]
        self.splits = read_json(self.splits_path)
        self.splits_hash = sha256_file(self.splits_path)
        self.contexts = list(self.protocol["architecture"]["contexts"])
        self.validation = self._read_validation()

    def _read_validation(self) -> dict[str, dict[str, Any]]:
        screen = read_json(self.run_dir / "screen_results.json")
        require(screen["protocol_sha256"] == self.protocol_hash, "Screen protocol pin changed")
        require(screen["splits_sha256"] == self.splits_hash, "Screen split pin changed")
        require(screen["test_evaluated"] is False, "Screen used test data")
        screen_seed = int(self.protocol["training"]["screen_seed"])
        expected = {f"{name}/context_{c:03d}/seed_{screen_seed}"
                    for name in ARCHITECTURES for c in self.contexts}
        # Only the three shortlisted contexts per architecture have repeat seeds:
        # 16 screen runs + 12 repeat runs + 1 original-ResNet control = 29.
        for name in ARCHITECTURES:
            contexts = screen["shortlist"][name]
            require(len(contexts) == 3 and len(set(contexts)) == 3
                    and set(contexts) <= set(self.contexts), "Invalid frozen shortlist")
            for context in contexts:
                for seed in self.protocol["training"]["repeat_seeds"]:
                    expected.add(f"{name}/context_{context:03d}/seed_{seed}")
        expected.add("resnet_original_control/context_121/seed_42")
        rows = {}
        training_sources = None
        # Includes resnet_original_control, unlike an architecture-only glob.
        for path in sorted(self.run_dir.glob("*/context_*/seed_*/validation_metrics.json")):
            row = read_json(path)
            run_id = row["run_id"]
            require(run_id not in rows, f"Duplicate run id: {run_id}")
            require(path == self.run_dir / f"{run_id}/validation_metrics.json",
                    f"Run id does not match artifact location: {path}")
            require(row["protocol_sha256"] == self.protocol_hash, f"Protocol pin changed: {run_id}")
            require(row["splits_sha256"] == self.splits_hash, f"Split pin changed: {run_id}")
            if training_sources is None:
                training_sources = dict(row["source_sha256"])
            require(row["source_sha256"] == training_sources, f"Training source pins differ: {run_id}")
            require(row["test_evaluated"] is False, f"Validation record contains test use: {run_id}")
            require(row["checkpoint_sha256"] == sha256_file(path.with_name("best_model.pt")),
                    f"Checkpoint pin changed: {run_id}")
            rows[run_id] = row
        require(set(rows) == expected,
                f"Incomplete/unexpected validation artifacts: expected {len(expected)}, found {len(rows)}; "
                f"missing={sorted(expected - set(rows))}; unexpected={sorted(set(rows) - expected)}")
        require(len(rows) == int(self.protocol["selection"]["expected_training_runs"]),
                "Validation count differs from frozen protocol")
        for source, digest in training_sources.items():
            require(sha256_file(ROOT / source) == digest, f"Training source changed: {source}")
        return rows

    def lock(self) -> dict[str, Any]:
        lock = read_json(self.run_dir / "selection_lock.json")
        require(lock["protocol_sha256"] == self.protocol_hash, "Lock protocol pin changed")
        require(lock["split_sha256"] == self.splits_hash, "Lock split pin changed")
        require(lock["screen_sha256"] == sha256_file(self.run_dir / "screen_results.json"),
                "Screen results changed after lock")
        require(lock["repeats_sha256"] == sha256_file(self.run_dir / "repeat_results.json"),
                "Repeat results changed after lock")
        require(lock["test_evaluated_before_lock"] is False, "Test was evaluated before lock")
        require(lock["representative_seed_selected"] is False, "A representative seed was selected")
        require(set(lock["winners"]) == set(ARCHITECTURES), "Need one locked context per architecture")
        return lock

    def evaluation(self) -> dict[str, Any]:
        report = read_json(self.run_dir / "evaluation.json")
        require(report["selection_lock_sha256"] == sha256_file(self.run_dir / "selection_lock.json"),
                "Evaluation does not reference the current selection lock")
        require(report["evaluation_started_after_lock"] is True
                and report["selection_changed_after_test"] is False,
                "Evaluation/selection chronology is invalid")
        return report


def validation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"run_id": row["run_id"], "architecture": row["architecture"],
            "context_frames": row["context_frames"], "seed": row["seed"], "control": row["control"],
            "split_seed": row["split_seed"], "validation_mean_f1": row["selection_score"]["mean_f1"],
            "validation_division_event_f1": row["selection_score"]["division_event_f1"],
            "validation_exclusion_f1": row["selection_score"]["exclusion_f1"],
            "validation_exclusion_ap": row["validation"]["point"]["excluded_point"]["average_precision"],
            "minimum_validation_loss": row["minimum_validation_loss"],
            "parameter_count": row["parameter_count"], "epochs": row["epochs"],
            "best_epoch": row["best_epoch"], "seconds": row["seconds"],
            "division_threshold": row["thresholds"]["division_point"],
            "exclusion_threshold": row["thresholds"]["excluded_point"],
            "effective_dilations": json.dumps(row["effective_dilations"]),
            "checkpoint_sha256": row["checkpoint_sha256"], "protocol_sha256": row["protocol_sha256"],
            "splits_sha256": row["splits_sha256"],
            "source_sha256": json.dumps(row["source_sha256"], sort_keys=True)}


def screen_table(artifacts: Artifacts) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # screen_results.json is canonical for both screen.log and screen_aggregate.log.
    screen = read_json(artifacts.run_dir / "screen_results.json")
    require(screen["protocol_sha256"] == artifacts.protocol_hash, "Screen protocol pin changed")
    require(screen["splits_sha256"] == artifacts.splits_hash, "Screen split pin changed")
    require(screen["test_evaluated"] is False, "Screen used test data")
    expected = {f"{name}/context_{c:03d}/seed_42" for name in ARCHITECTURES for c in artifacts.contexts}
    require({row["run_id"] for row in screen["rows"]} == expected,
            "Screen table must contain all 16 non-control candidates")
    require(screen["control"]["run_id"] == "resnet_original_control/context_121/seed_42",
            "Original ResNet control is not separate")
    return [validation_row(artifacts.validation[run_id]) for run_id in sorted(expected)], screen


def repeat_table(artifacts: Artifacts, lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeat = read_json(artifacts.run_dir / "repeat_results.json")
    require(repeat["protocol_sha256"] == artifacts.protocol_hash, "Repeat protocol pin changed")
    require(repeat["test_evaluated"] is False, "Repeat stage used test data")
    require(len(repeat["groups"]) == 6, "Expected six repeated context groups")
    grouped, raw, seen = [], [], set()
    for group in repeat["groups"]:
        name, context = group["architecture"], int(group["context_frames"])
        require(name in ARCHITECTURES and context in artifacts.contexts, "Unexpected repeated context")
        require((name, context) not in seen, "Duplicate repeated context")
        seen.add((name, context))
        runs = [artifacts.validation[f"{name}/context_{context:03d}/seed_{seed}"] for seed in SEEDS]
        require(group["runs"] == [row["run_id"] for row in runs] and group["seeds"] == list(SEEDS),
                "Repeat group must contain all three frozen seeds")
        result = {"architecture": name, "context_frames": context, "seeds": list(SEEDS),
                  "run_ids": group["runs"], "n_training_seeds": 3,
                  "parameter_count": runs[0]["parameter_count"],
                  "wall_time_seconds_total": sum(row["seconds"] for row in runs)}
        scores = [row["selection_score"] for row in runs]
        for key in ("mean_f1", "division_event_f1", "exclusion_f1", "minimum_validation_loss"):
            stats = mean_sd(score[key] for score in scores)
            result[f"validation_{key}_mean"] = stats["mean"]
            result[f"validation_{key}_sample_sd"] = stats["sample_sd"]
            require(math.isclose(group["selection_score"][key], stats["mean"], rel_tol=0, abs_tol=1e-12),
                    f"Repeat aggregate differs: {name}/{context}/{key}")
            require(math.isclose(group["sample_sd"][key], stats["sample_sd"], rel_tol=0, abs_tol=1e-12),
                    f"Repeat sample SD differs: {name}/{context}/{key}")
        ap = mean_sd(row["validation"]["point"]["excluded_point"]["average_precision"] for row in runs)
        result["validation_exclusion_ap_mean"] = ap["mean"]
        result["validation_exclusion_ap_sample_sd"] = ap["sample_sd"]
        grouped.append(result)
        raw.extend(validation_row(row) for row in runs)
    require(len(seen) == 6 and all(sum(name == arch for name, _ in seen) == 3 for arch in ARCHITECTURES),
            "Each architecture must repeat exactly three contexts")
    for name, winner in lock["winners"].items():
        require(any(row["architecture"] == name and row["context_frames"] == winner["context_frames"]
                    and row["run_ids"] == winner["runs"] for row in grouped),
                f"Locked {name} winner is not a repeated context")
    return grouped, raw


def test_row(model: dict[str, Any], population: str) -> dict[str, Any]:
    return {"label": model["label"], "architecture": model["architecture"],
            "context_frames": model["context_frames"], "seed": model["seed"],
            "selected": model["selected"], "control": not model["selected"],
            "population": population, **point_values(model["populations"][population])}


def evaluation_tables(artifacts: Artifacts, lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = artifacts.evaluation()
    models = evaluation["models"]
    for model in models:
        require(model["checkpoint_sha256"] == sha256_file(output_path(Path(model["checkpoint"]))),
                f"Evaluated checkpoint changed: {model['label']}")
    selected = []
    for name, winner in lock["winners"].items():
        found = [model for model in models if model["selected"] and model["architecture"] == name
                 and model["context_frames"] == winner["context_frames"] and model["seed"] in SEEDS]
        require(len(found) == 3 and {model["seed"] for model in found} == set(SEEDS),
                f"Evaluation must retain all seeds for locked {name} context")
        selected.extend(found)
    require(len([model for model in models if model["selected"]]) == 6, "Unexpected selected model")
    raw = [test_row(model, population) for model in selected for population in POPULATIONS]
    aggregate = []
    for name in ARCHITECTURES:
        context = lock["winners"][name]["context_frames"]
        for population in POPULATIONS:
            rows = [row for row in raw if row["architecture"] == name and row["population"] == population]
            result = {"architecture": name, "context_frames": context, "population": population,
                      "seeds": sorted(row["seed"] for row in rows), "n_training_seeds": len(rows),
                      "selection_status": "locked_before_evaluation"}
            for key in ("mean_f1", "division_event_f1", "exclusion_f1", "exclusion_ap"):
                stats = mean_sd(row[key] for row in rows)
                result[f"{key}_mean"] = stats["mean"]
                result[f"{key}_sample_sd"] = stats["sample_sd"]
            result["population_support"] = {key: evaluation["populations"][population][key]
                for key in ("trajectories", "rows", "division_points", "excluded_points")}
            aggregate.append(result)
    require(len(aggregate) == 6, "Selected test table must be two contexts by three populations")
    return aggregate, raw

def control_table(artifacts: Artifacts, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in evaluation["models"]:
        if model["selected"]:
            continue
        require(model["seed"] == 42, "Descriptive control must not be seed-selected")
        for population in POPULATIONS:
            row = test_row(model, population)
            row.update({"role": "descriptive_single_seed_control", "n_training_seeds": 1,
                        "mean_f1_mean": row["mean_f1"], "mean_f1_sample_sd": None,
                        "exclusion_ap_mean": row["exclusion_ap"], "exclusion_ap_sample_sd": None,
                        "division_event_f1_mean": row["division_event_f1"],
                        "exclusion_f1_mean": row["exclusion_f1"]})
            rows.append(row)
    require(len(rows) == 9, "Expected three runner controls across three populations")
    supplement_path = artifacts.run_dir / "control_evaluation.json"
    if supplement_path.exists():
        supplement = read_json(supplement_path)
        require(supplement["protocol_sha256"] == artifacts.protocol_hash,
                "Control supplement protocol pin changed")
        require(supplement["selection_lock_sha256"] == sha256_file(artifacts.run_dir / "selection_lock.json"),
                "Control supplement lock pin changed")
        model = supplement["model"]
        require(model["checkpoint_sha256"] == sha256_file(output_path(Path(model["checkpoint"]))),
                "Original ResNet control checkpoint changed")
        for population in POPULATIONS:
            row = test_row(model, population)
            row.update({"role": "descriptive_original_resnet_control", "n_training_seeds": 1,
                        "mean_f1_mean": row["mean_f1"], "mean_f1_sample_sd": None,
                        "exclusion_ap_mean": row["exclusion_ap"], "exclusion_ap_sample_sd": None,
                        "division_event_f1_mean": row["division_event_f1"],
                        "exclusion_f1_mean": row["exclusion_f1"]})
            rows.append(row)
    return rows


def save_figure(fig: plt.Figure, stem: Path, description: str, data_csv: str) -> dict[str, Any]:
    paths = []
    for suffix in (".png", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    plt.close(fig)
    from matplotlib.image import imread
    pixels = imread(paths[0]).shape
    return {"description": description, "data_csv": data_csv,
            "files": [{"path": relative(path), "format": path.suffix[1:],
                       "sha256": sha256_file(path), "dpi": 300 if path.suffix == ".png" else None,
                       "pixel_width": int(pixels[1]) if path.suffix == ".png" else None,
                       "pixel_height": int(pixels[0]) if path.suffix == ".png" else None}
                      for path in paths]}


def make_figures(artifacts: Artifacts, screen: list[dict[str, Any]], repeated: list[dict[str, Any]],
                 repeated_raw: list[dict[str, Any]], test_aggregate: list[dict[str, Any]],
                 controls: list[dict[str, Any]]) -> dict[str, Any]:
    out = artifacts.run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    pd.DataFrame(screen).to_csv(out / "screen_context_data.csv", index=False)
    pd.DataFrame(repeated_raw).to_csv(out / "repeated_validation_seed_data.csv", index=False)
    pd.DataFrame(repeated).to_csv(out / "repeated_validation_summary.csv", index=False)
    test_all = [{"label": f"Locked {row['architecture'].upper()} RF{row['context_frames']}",
                 **row, "role": "locked_context", "seed": None} for row in test_aggregate]
    test_all.extend(controls)
    pd.DataFrame(test_all).to_csv(out / "test_comparison_data.csv", index=False)
    pd.DataFrame([validation_row(row) for row in artifacts.validation.values()]).to_csv(
        out / "validation_run_provenance.csv", index=False)
    manifests = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, name in zip(axes, ARCHITECTURES):
        rows = sorted((row for row in screen if row["architecture"] == name),
                      key=lambda row: row["context_frames"])
        x = [row["context_frames"] for row in rows]
        ax.plot(x, [row["validation_mean_f1"] for row in rows], color=COLORS[name],
                marker="o", label="Mean of division-event F1 and exclusion F1")
        ax.plot(x, [row["validation_exclusion_ap"] for row in rows], color="#009E73",
                marker="s", linestyle="--", label="Exclusion average precision")
        ax.set(title=f"{name.upper()} (seed 42)", xlabel="Theoretical receptive field (frames)",
               ylabel="Validation metric", ylim=(0, 1), xticks=x)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="lower center")
    fig.suptitle("R016 validation screen: observed contexts joined by straight segments")
    manifests["screen_context"] = save_figure(fig, out / "screen_context",
        "Two panels of all 16 seed-42 screen candidates. Piecewise lines connect observed contexts only; "
        "no interpolation, optimum estimation, or test-based selection. Metrics use a common 0–1 scale.",
        "screen_context_data.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, name in zip(axes, ARCHITECTURES):
        rows = sorted((row for row in repeated if row["architecture"] == name),
                      key=lambda row: row["context_frames"])
        x = [row["context_frames"] for row in rows]
        for seed in SEEDS:
            seed_rows = [row for row in repeated_raw if row["architecture"] == name and row["seed"] == seed]
            by_context = {row["context_frames"]: row for row in seed_rows}
            ax.scatter(x, [by_context[c]["validation_mean_f1"] for c in x],
                       marker=MARKERS[seed], s=34, alpha=0.85, label=f"Seed {seed}", zorder=3)
        ax.errorbar(x, [row["validation_mean_f1_mean"] for row in rows],
                    yerr=[row["validation_mean_f1_sample_sd"] for row in rows],
                    color=COLORS[name], marker="D", linestyle="--", capsize=4,
                    linewidth=1.4, label="Mean ± sample SD (n=3 seeds)")
        ax.set(title=name.upper(), xlabel="Theoretical receptive field (frames)",
               ylabel="Validation mean F1", ylim=(0, 1), xticks=x)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="lower center")
    fig.suptitle("Repeated validation: every seed shown; only complete context means are connected")
    manifests["repeated_validation"] = save_figure(fig, out / "repeated_validation",
        "All three optimization seeds are displayed for each of six shortlisted contexts. Diamonds and "
        "error bars are arithmetic means ± sample SD across three training seeds, not generalization CIs.",
        "repeated_validation_seed_data.csv; repeated_validation_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), layout="constrained")
    # Only the frozen R015 test population belongs in this comparison;
    # all three populations remain available in summary.json and the CSV.
    rows = [row for row in test_aggregate if row["population"] == "test"]
    rows += [row for row in controls if row["population"] == "test"]
    control_labels = {
        "R015_cnn_rf17": "R015\nCNN RF17\nseed 42",
        "R015_resnet_rf121": "R015\nResNet RF121\nseed 42",
        "production_R010_cnn_rf17": "R010\nproduction\nCNN RF17\nseed 42",
        "R016_original_resnet_k5_rf121_seed42": "R016 control\noriginal ResNet\nk5 RF121\nseed 42",
    }
    labels = [f"Locked\n{row['architecture'].upper()} RF{row['context_frames']}\nn=3 seeds"
              if "selection_status" in row else control_labels.get(row["label"], row["label"])
              for row in rows]
    require(len(rows) in (5, 6), "Test plot requires two locked contexts and three/four controls")
    x = np.arange(len(rows))
    for ax, metric, title in zip(axes, ("mean_f1", "exclusion_ap"),
            ("Mean of division-event F1 and exclusion F1", "Exclusion average precision")):
        y = [row[f"{metric}_mean"] for row in rows]
        sd = [row[f"{metric}_sample_sd"] for row in rows]
        ax.errorbar(x, y, yerr=[0 if value is None else value for value in sd], fmt="none",
                    ecolor="#333333", capsize=4, elinewidth=1.1, zorder=2)
        for index, row in enumerate(rows):
            selected = "selection_status" in row
            ax.scatter(index, y[index], marker="D" if selected else "o", s=58,
                       color=COLORS[row["architecture"]] if selected else "#666666",
                       edgecolor="black", linewidth=0.5, zorder=3)
        ax.set(title=title, ylim=(0, 1), xticks=x,
               xticklabels=labels,
               ylabel="Frozen R015 test metric")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Locked contexts (mean ± sample SD, n=3 training seeds) and single-seed controls")
    fig.get_layout_engine().set(rect=(0.0, 0.12, 1.0, 0.88))
    fig.text(0.5, 0.017, "Error bars describe optimization variation, not new-FOV generalization or a confidence interval. "
             "Controls have no repeat-seed uncertainty.", ha="center", fontsize=8)
    manifests["test_comparison"] = save_figure(fig, out / "test_comparison",
        "Frozen R015 test metrics for two locked contexts across all three seeds and descriptive "
        "single-seed R015/R010 controls, with the newly trained original-ResNet control when available. "
        "Gray circles are controls; colored diamonds are locked contexts. No best seed is chosen.",
        "test_comparison_data.csv")
    return manifests


def make_summary(artifacts: Artifacts) -> dict[str, Any]:
    lock = artifacts.lock()
    screen, screen_artifact = screen_table(artifacts)
    repeated, repeated_raw = repeat_table(artifacts, lock)
    selected_test, selected_test_raw = evaluation_tables(artifacts, lock)
    evaluation = artifacts.evaluation()
    controls = control_table(artifacts, evaluation)
    figures = make_figures(artifacts, screen, repeated, repeated_raw, selected_test, controls)
    figure_dir = artifacts.run_dir / "figures"
    source_paths = [artifacts.protocol_path, artifacts.splits_path, SCRIPT_PATH,
                    artifacts.run_dir / "screen_results.json", artifacts.run_dir / "repeat_results.json",
                    artifacts.run_dir / "selection_lock.json", artifacts.run_dir / "evaluation.json"]
    if (artifacts.run_dir / "control_evaluation.json").exists():
        source_paths.append(artifacts.run_dir / "control_evaluation.json")
    sources = {relative(path): sha256_file(path) for path in source_paths}
    for run_id, row in artifacts.validation.items():
        sources[relative(artifacts.run_dir / run_id / "validation_metrics.json")] = sha256_file(
            artifacts.run_dir / run_id / "validation_metrics.json")
        sources[relative(artifacts.run_dir / run_id / "best_model.pt")] = row["checkpoint_sha256"]
    for path in sorted(figure_dir.glob("*.csv")):
        sources[relative(path)] = sha256_file(path)
    manifest = {"report": "R016 figure and data provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_artifacts": sources,
        "training_source_artifacts": artifacts.validation[next(iter(artifacts.validation))]["source_sha256"],
        "transformations": ["Arithmetic mean and sample SD (ddof=1) across training seeds 42, 43, 44",
            "No smoothing or interpolated optimum; straight segments only between observed contexts",
            "No selection, best-seed selection, or model promotion performed by this reporter"],
        "uncertainty": "Sample SD across n=3 training seeds; optimization variation, not a generalization CI. Controls have n=1.",
        "missing_data": "None accepted: all 16 screen, six repeat and six selected evaluations required.",
        "publication_status": "Provisional scientific figures; journal and final-width requirements unspecified.",
        "figures": figures}
    write_json(figure_dir / "figure_manifest.json", manifest)
    summary = {"report": "R016 temporal-context summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": artifacts.protocol_hash, "splits_sha256": artifacts.splits_hash,
        "selection_lock_sha256": sha256_file(artifacts.run_dir / "selection_lock.json"),
        "selection_policy": "Copied from selection_lock.json; never re-ranks contexts or seeds.",
        "screen_shortlist_from_frozen_runner": screen_artifact["shortlist"],
        "screen": screen, "screen_control": validation_row(screen_artifact["control"]),
        "repeated_contexts": repeated, "repeated_seed_metrics": repeated_raw,
        "locked_winners": lock["winners"], "selected_test": selected_test,
        "selected_test_seed_metrics": selected_test_raw, "control_test_descriptive": controls,
        "populations": evaluation["populations"], "figure_manifest": relative(figure_dir / "figure_manifest.json"),
        "limitations": artifacts.protocol["reporting"]["limitations"],
        "selection_changed": False, "promotion_performed": False}
    write_json(artifacts.run_dir / "summary.json", summary)
    return summary


def resolve_device(device: str) -> Any:
    import torch
    require(device in ("cpu", "cuda"), "Inference device must be explicitly cpu or cuda")
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; not falling back to CPU")
        return torch.device("cuda")
    return torch.device("cpu")


def load_experiment_data(artifacts: Artifacts) -> tuple[pd.DataFrame, dict[str, Any]]:
    from moma_lineage_analysis.trajectory_cleaning import dataset_store as ds
    from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryDataset, TrajectoryNormalizer
    store = ROOT / artifacts.protocol["dataset"]["store"]
    connection = ds.connect_readonly(store)
    try:
        frame, lineage = ds.load_training_frame(connection, [artifacts.protocol["dataset"]["lineage"]],
            require_human_verified=True, store_path=store)
    finally:
        connection.close()
    require(lineage["lineages"][0]["canonical_hash"] == artifacts.protocol["dataset"]["canonical_hash"],
            "Frozen dataset content changed")
    datasets = {split: TrajectoryDataset(frame, keys=keys, normalizer=TrajectoryNormalizer())
                for split, keys in artifacts.splits.items()}
    return frame, datasets

def population_definitions(artifacts: Artifacts, frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, Any]]:
    from moma_lineage_analysis.trajectory_cleaning import dataset_store as ds
    original = read_json(ROOT / "runs/merged_20261017/cnn/splits.json")
    common = sorted(set(original["test"]) & set(artifacts.splits["test"]))
    connection = ds.connect_readonly(ROOT / artifacts.protocol["dataset"]["store"])
    try:
        content = dict(connection.execute("SELECT trajectory_key,content_hash FROM trajectory"))
    finally:
        connection.close()
    seen = {content[key] for split in (original, artifacts.splits)
            for partition in ("train", "validation") for key in split[partition]}
    require(not any(content[key] in seen for key in common),
            "Common historical test content overlaps either train/validation partition")
    keys = {"test": artifacts.splits["test"], "common_historical_test": common,
            "new_batch_test": [key for key in artifacts.splits["test"] if key.startswith("curated_20261029::")]}
    populations = {}
    for name, values in keys.items():
        mask = frame.trajectory_key.isin(values)
        populations[name] = {"keys": values, "trajectories": len(values), "rows": int(mask.sum()),
            "division_points": int(frame.loc[mask, "division_point"].sum()),
            "excluded_points": int(frame.loc[mask, "excluded_point"].sum())}
    return keys, {"populations": populations, "no_common_content_overlap_with_either_train_validation": True}


def checkpoint_lock(artifacts: Artifacts) -> dict[str, Any]:
    lock = artifacts.lock()
    require(artifacts.evaluation()["selection_lock_sha256"] == sha256_file(artifacts.run_dir / "selection_lock.json"),
            "Control reporting requires completed locked evaluation")
    return lock


def evaluate_control(artifacts: Artifacts, device: str) -> dict[str, Any]:
    import torch
    from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryDataset
    from moma_lineage_analysis.trajectory_cleaning.training import evaluate_model, load_checkpoint
    selected_device = resolve_device(device)
    checkpoint_lock(artifacts)
    frame, _ = load_experiment_data(artifacts)
    keys, population_report = population_definitions(artifacts, frame)
    row = artifacts.validation["resnet_original_control/context_121/seed_42"]
    require(row["control"] is True and row["context_frames"] == 121 and row["seed"] == 42,
            "Original ResNet control identity changed")
    require(row["model_config"]["kernel_size"] == 5 and row["model_config"]["conv_dilations"] is None,
            "Original ResNet control must retain k5/default dilations")
    checkpoint = artifacts.run_dir / row["run_id"] / "best_model.pt"
    model, normalizer, thresholds, _ = load_checkpoint(checkpoint, device=selected_device)
    datasets = {name: TrajectoryDataset(frame, keys=values, normalizer=normalizer)
                for name, values in keys.items()}
    results = {name: evaluate_model(model, dataset, thresholds=thresholds, batch_size=16,
                                    device=selected_device, division_tolerance_frames=1)
               for name, dataset in datasets.items()}
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
    report = {"report": "R016 newly trained original-ResNet control evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": artifacts.protocol_hash,
        "selection_lock_sha256": sha256_file(artifacts.run_dir / "selection_lock.json"),
        "role": "Descriptive control only; excluded from shortlist and locked-winner selection",
        "device": str(selected_device), "torch": torch.__version__,
        "model": {"label": "R016_original_resnet_k5_rf121_seed42", "architecture": "resnet",
                  "context_frames": 121, "seed": 42, "selected": False, "control": True,
                  "checkpoint": relative(checkpoint), "checkpoint_sha256": row["checkpoint_sha256"],
                  "thresholds": dict(zip(LABELS, map(float, thresholds))), "populations": results},
        **population_report, "selection_changed": False, "promotion_performed": False}
    write_json(artifacts.run_dir / "control_evaluation.json", report)
    return report


def record_metadata(frame: pd.DataFrame, keys: Iterable[str]) -> list[dict[str, Any]]:
    selected = frame[frame.trajectory_key.isin(set(keys))]
    records = []
    for key, group in selected.groupby("trajectory_key", sort=False):
        group = group.sort_values("time_h", kind="mergesort")
        time = group.time_h.to_numpy(dtype=float)
        require(np.isfinite(time).all() and len(time) > 0 and np.all(np.diff(time) >= 0),
                f"Invalid/non-monotonic time for diagnostic trajectory {key}")
        records.append({"key": key, "time": time, "source": str(group.source_dataset.iloc[0]),
                        "targets": group.loc[:, LABELS].to_numpy(dtype=np.float32)})
    return records


def distribution(values: np.ndarray) -> dict[str, Any]:
    require(len(values) > 0, "No valid full temporal windows")
    return {"windows": len(values), "mean_h": float(np.mean(values)),
            "sample_sd_h": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min_h": float(np.min(values)), "p05_h": float(np.quantile(values, 0.05)),
            "p25_h": float(np.quantile(values, 0.25)), "median_h": float(np.median(values)),
            "p75_h": float(np.quantile(values, 0.75)), "p95_h": float(np.quantile(values, 0.95)),
            "max_h": float(np.max(values))}


def temporal_coverage(records: list[dict[str, Any]], context: int, split: str) -> dict[str, Any]:
    radius = (context - 1) // 2
    span_groups = defaultdict(list)
    coverage_groups = defaultdict(lambda: defaultdict(int))
    for record in records:
        n, targets = len(record["time"]), record["targets"]
        for source in ("ALL", record["source"]):
            counts = coverage_groups[source]
            counts["trajectories"] += 1
            counts["rows"] += n
            counts["division_points"] += int(targets[:, 0].sum())
            counts["excluded_points"] += int(targets[:, 1].sum())
            if n >= context:
                counts["full_window_trajectories"] += 1
                counts["full_window_rows"] += n - 2 * radius
                centers = np.arange(radius, n - radius)
                counts["full_window_division_points"] += int(targets[centers, 0].sum())
                counts["full_window_excluded_points"] += int(targets[centers, 1].sum())
                span_groups[source].append(record["time"][centers + radius] - record["time"][centers - radius])
    coverage = []
    for source, values in sorted(coverage_groups.items()):
        counts = dict(values)
        for key in ("full_window_trajectories", "full_window_rows",
                    "full_window_division_points", "full_window_excluded_points"):
            counts.setdefault(key, 0)
        coverage.append({"source_dataset": source, **counts,
                         "full_window_row_fraction": counts["full_window_rows"] / counts["rows"]})
    spans = [{"source_dataset": source, **distribution(np.concatenate(pieces))}
             for source, pieces in sorted(span_groups.items())]
    return {"split": split, "context_frames": context, "radius_frames": radius,
            "coverage": coverage, "elapsed_span_h": spans,
            "definition": "Full windows have observed frames at center±radius; elapsed span uses actual time_h. "
                          "Coverage counts row centers, not padded predictions."}


def diagnostics(artifacts: Artifacts, device: str) -> dict[str, Any]:
    import torch
    from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryNormalizer
    from moma_lineage_analysis.trajectory_cleaning.training import _point_metrics, collect_predictions, load_checkpoint
    selected_device = resolve_device(device)
    checkpoint_lock(artifacts)
    frame, datasets = load_experiment_data(artifacts)
    default_normalizer = TrajectoryNormalizer().to_dict()
    metadata = {record["key"]: record for record in record_metadata(frame, artifacts.splits["validation"])}
    run_reports = []
    for run_id, row in sorted(artifacts.validation.items()):
        checkpoint = artifacts.run_dir / run_id / "best_model.pt"
        model, normalizer, thresholds, _ = load_checkpoint(checkpoint, device=selected_device)
        require(normalizer.to_dict() == default_normalizer,
                f"Diagnostic normalizer differs from experiment default: {run_id}")
        predictions = collect_predictions(model, datasets["validation"], batch_size=16, device=selected_device)
        grouped = defaultdict(list)
        cohort_counts = defaultdict(set)
        for prediction in predictions:
            n = len(prediction["targets"])
            indices = np.arange(n)
            common_interior = (indices >= 98) & (indices < n - 98)
            masks = {"all_rows": np.ones(n, dtype=bool),
                     "common_interior_radius98": common_interior,
                     "common_boundary_radius98": ~common_interior,
                     "trajectory_shorter_than197": np.full(n, n < 197),
                     "trajectory_at_least197": np.full(n, n >= 197)}
            source = metadata[prediction["key"]]["source"]
            for cohort, mask in masks.items():
                if mask.any():
                    for source_name in ("ALL", source):
                        grouped[(cohort, source_name)].append(
                            (prediction["targets"][mask], prediction["probabilities"][mask]))
                        cohort_counts[(cohort, source_name)].add(prediction["key"])
        cohorts = []
        for (cohort, source), pieces in sorted(grouped.items()):
            targets = np.concatenate([piece[0] for piece in pieces])
            probabilities = np.concatenate([piece[1] for piece in pieces])
            cohorts.append({"cohort": cohort, "source_dataset": source,
                            "trajectories": len(cohort_counts[(cohort, source)]), "rows": len(targets),
                            "point": _point_metrics(targets, probabilities, thresholds)})
        validation_rows = int(frame.trajectory_key.isin(artifacts.splits["validation"]).sum())
        require(sum(item["rows"] for item in cohorts if item["cohort"] == "all_rows"
                    and item["source_dataset"] == "ALL") == validation_rows,
                f"Validation prediction coverage is incomplete: {run_id}")
        run_reports.append({"run_id": run_id, "architecture": row["architecture"],
                            "context_frames": row["context_frames"], "seed": row["seed"],
                            "control": row["control"], "checkpoint_sha256": row["checkpoint_sha256"],
                            "checkpoint": relative(checkpoint), "cohorts": cohorts})
        print(f"DIAGNOSTIC {run_id}", flush=True)
        del model, predictions
        if device == "cuda":
            torch.cuda.empty_cache()
    temporal = [temporal_coverage(record_metadata(frame, keys), context, split)
                for split, keys in artifacts.splits.items() for context in artifacts.contexts]
    report = {"report": "R016 validation temporal diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": artifacts.protocol_hash,
        "selection_lock_sha256": sha256_file(artifacts.run_dir / "selection_lock.json"),
        "device": str(selected_device), "torch": torch.__version__, "runs": run_reports,
        "prediction_passes": "One collect_predictions call per checkpoint on complete validation trajectories",
        "cohort_definition": "Fixed masks on complete trajectories; common interior is 98<=index<n-98; "
            "common boundary is its complement; length strata are n<197 and n>=197. Per-source rows use "
            "the same masks. Zero-row cohorts are absent, not zero-valued metrics.",
        "metrics": "Point precision/recall/F1/AP at stored thresholds; support is label-positive rows. "
                   "No division events are re-clustered on cropped tracks.",
        "temporal_coverage": temporal,
        "context_note": "RF is a normalized-input property; whole-trajectory normalization remains unchanged.",
        "selection_changed": False, "promotion_performed": False}
    write_json(artifacts.run_dir / "validation_diagnostics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("summary", "diagnostics", "control", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="R016 run directory (default: runs/context_20261029)")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="Explicit inference device; required for diagnostics/control/all. No CPU fallback.")
    args = parser.parse_args()
    if args.stage != "summary" and not args.device:
        parser.error("--device is required for diagnostics, control, and all (use cuda after Lead authorization)")
    if args.stage == "summary" and args.device:
        parser.error("summary is CPU-only; omit --device")
    artifacts = Artifacts(output_path(args.output_dir))
    if args.stage in ("diagnostics", "all"):
        report = diagnostics(artifacts, args.device)
        print(f"Wrote {len(report['runs'])} validation diagnostic runs and "
              f"{len(report['temporal_coverage'])} temporal summaries to "
              f"{artifacts.run_dir / 'validation_diagnostics.json'}")
    if args.stage in ("control", "all"):
        report = evaluate_control(artifacts, args.device)
        print(f"Wrote descriptive original-ResNet control report to {artifacts.run_dir / 'control_evaluation.json'}")
        for name, metrics in report["model"]["populations"].items():
            print(f"  {name}: {point_values(metrics)}")
    if args.stage in ("summary", "all"):
        summary = make_summary(artifacts)
        print(f"Screen: {len(summary['screen'])} candidates; repeats: {len(summary['repeated_contexts'])} contexts; "
              f"locked test summaries: {len(summary['selected_test'])} (2 contexts x 3 populations).")
        rows = [{"architecture": row["architecture"], "RF": row["context_frames"],
                 "population": row["population"], "n_seeds": row["n_training_seeds"],
                 "mean_F1": row["mean_f1_mean"], "exclusion_AP": row["exclusion_ap_mean"]}
                for row in summary["selected_test"]]
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print("Locked contexts copied without re-selection; no best-seed selection or promotion.")
        print(f"Summary: {artifacts.run_dir / 'summary.json'}")
        print(f"Figures/provenance: {artifacts.run_dir / 'figures'}")


if __name__ == "__main__":
    main()


