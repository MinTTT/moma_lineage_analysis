"""Generate reproducible figures for the trajectory-cleaning model report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ("cnn", "resnet", "lstm")
MODEL_LABELS = {"cnn": "CNN", "resnet": "ResNet", "lstm": "BiLSTM"}
MODEL_COLORS = {
    "cnn": "#0072B2",
    "resnet": "#E69F00",
    "lstm": "#009E73",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _load_runs(run_directory: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        model_directory = run_directory / model_name
        with (model_directory / "metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        history = pd.read_csv(model_directory / "history.csv")
        with (model_directory / "splits.json").open(encoding="utf-8") as handle:
            splits = json.load(handle)
        runs[model_name] = {
            "metrics": metrics,
            "history": history,
            "splits": splits,
        }
    return runs


def plot_training_loss(runs: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), sharey=True)
    for axis, model_name in zip(axes, MODEL_ORDER, strict=True):
        history = runs[model_name]["history"]
        best_row = history.loc[history["validation_loss"].idxmin()]
        color = MODEL_COLORS[model_name]
        axis.plot(
            history["epoch"],
            history["train_loss"],
            color=color,
            linewidth=1.8,
            label="Training",
        )
        axis.plot(
            history["epoch"],
            history["validation_loss"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label="Validation",
        )
        axis.scatter(
            [best_row["epoch"]],
            [best_row["validation_loss"]],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            marker="D",
            s=42,
            zorder=3,
        )
        axis.annotate(
            f"best epoch {int(best_row['epoch'])}\n{best_row['validation_loss']:.3f}",
            (best_row["epoch"], best_row["validation_loss"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.set_ylim(0.003, 0.65)
        axis.legend(loc="upper right")
    axes[0].set_ylabel("Masked focal BCE loss (log scale)")
    figure.suptitle("Training histories and validation-selected checkpoints", y=1.02)
    figure.savefig(output)
    plt.close(figure)


def _metric(metrics: dict, split: str, metric_name: str) -> float:
    if metric_name == "Division event":
        return float(metrics[split]["division_event"]["f1"])
    return float(metrics[split]["point"]["excluded_point"]["f1"])


def plot_f1_comparison(runs: dict[str, dict], output: Path) -> None:
    facets = (
        ("validation", "Division event"),
        ("validation", "Noise point"),
        ("test", "Division event"),
        ("test", "Noise point"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.1), sharey=True)
    for axis, (split, metric_name) in zip(axes.flat, facets, strict=True):
        values = [
            _metric(runs[model_name]["metrics"], split, metric_name)
            for model_name in MODEL_ORDER
        ]
        bars = axis.bar(
            np.arange(len(MODEL_ORDER)),
            values,
            color=[MODEL_COLORS[name] for name in MODEL_ORDER],
            width=0.68,
        )
        axis.set_xticks(
            np.arange(len(MODEL_ORDER)),
            [MODEL_LABELS[name] for name in MODEL_ORDER],
        )
        axis.set_ylim(0, 1.08)
        axis.set_title(f"{split.title()} · {metric_name} F1")
        axis.set_ylabel("F1 score")
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Model comparison on trajectory-grouped splits", y=1.01)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_label_balance(
    dataframe: pd.DataFrame,
    splits: dict[str, list],
    output: Path,
) -> None:
    total_rows = len(dataframe)
    division_count = int(dataframe["division_point"].sum())
    excluded_count = int(dataframe["excluded_point"].sum())
    overlap_count = int(
        (dataframe["division_point"] & dataframe["excluded_point"]).sum()
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    categories = ["Division", "Noise / excluded", "Both labels"]
    counts = [division_count, excluded_count, overlap_count]
    colors = ["#0072B2", "#D55E00", "#CC79A7"]
    bars = axes[0].barh(categories, counts, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Positive points")
    axes[0].set_title("Overall label prevalence")
    for bar, count in zip(bars, counts, strict=True):
        axes[0].text(
            count + 8,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({100 * count / total_rows:.2f}%)",
            va="center",
            fontsize=8,
        )
    axes[0].set_xlim(0, max(counts) * 1.34)

    split_order = ("train", "validation", "test")
    division_by_split = []
    excluded_by_split = []
    for split_name in split_order:
        subset = dataframe[
            dataframe["trajectory_key"].isin(splits[split_name])
        ]
        division_by_split.append(int(subset["division_point"].sum()))
        excluded_by_split.append(int(subset["excluded_point"].sum()))
    x_positions = np.arange(len(split_order))
    width = 0.34
    division_bars = axes[1].bar(
        x_positions - width / 2,
        division_by_split,
        width,
        color="#0072B2",
        label="Division",
    )
    exclusion_bars = axes[1].bar(
        x_positions + width / 2,
        excluded_by_split,
        width,
        color="#D55E00",
        label="Noise / excluded",
    )
    axes[1].set_xticks(x_positions, [name.title() for name in split_order])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Positive points (log scale)")
    axes[1].set_title("Positive-label support by split")
    axes[1].legend()
    for bars_for_label in (division_bars, exclusion_bars):
        for bar in bars_for_label:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    positive_split_counts = division_by_split + excluded_by_split
    axes[1].set_ylim(
        max(0.5, min(positive_split_counts) / 1.8),
        max(positive_split_counts) * 1.65,
    )
    figure.suptitle("Label imbalance and evaluation support", y=1.02)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repository_root / "trajectory_cleaning_run",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root
        / "cells_statistics"
        / "labeled_trajectories_example_data"
        / "0-8h_selected_labeled.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    runs = _load_runs(args.run_dir)
    dataframe = pd.read_csv(args.input)
    plot_training_loss(runs, args.output_dir / "training_loss_curves.png")
    plot_f1_comparison(runs, args.output_dir / "model_f1_comparison.png")
    plot_label_balance(
        dataframe,
        runs["cnn"]["splits"],
        args.output_dir / "dataset_label_balance.png",
    )
    for figure_path in sorted(args.output_dir.glob("*.png")):
        print(figure_path)


if __name__ == "__main__":
    main()
