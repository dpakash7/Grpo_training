import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.common import load_config, results_dir
from src.eval import MODELS, wilson_interval

SURFACE = "#fcfcfb"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
COLORS = {"grpo": "#2a78d6", "sft": "#eb6834", "base": "#1baf7a"}
LABELS = {"grpo": "GRPO", "sft": "SFT", "base": "Base"}
TRAINED = ("grpo", "sft")
SMOOTHING_WINDOW = 10
REWARD_PANELS = (
    ("reward", "Total weighted reward"),
    ("rewards/correctness_reward/mean", "Correct answer"),
    ("rewards/format_reward/mean", "Valid format"),
    ("rewards/length_penalty/mean", "Length penalty (0 is best)"),
)
COMPARISON_PANELS = (
    ("accuracy", "Strict accuracy"),
    ("lenient_accuracy", "Lenient accuracy"),
    ("format_valid_rate", "Valid format"),
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.labelcolor": SECONDARY_INK,
        "axes.titlecolor": "#0b0b0b",
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "text.color": SECONDARY_INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    }
)


def read_metrics(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Maps each metric to its (step, value) points. A missing file means the run has not produced it yet."""
    if not path.exists():
        return {}
    series = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series[row["metric"]].append((int(row["step"]), float(row["value"])))
    return dict(series)


def rolling_mean(values: list[float], window: int) -> list[float]:
    averages = []
    for i in range(len(values)):
        recent = values[max(0, i + 1 - window) : i + 1]
        averages.append(sum(recent) / len(recent))
    return averages


def plot_noisy(ax, points, color, label=None) -> None:
    steps, values = zip(*points)
    ax.plot(steps, values, color=color, linewidth=1, alpha=0.25)
    ax.plot(steps, rolling_mean(values, SMOOTHING_WINDOW), color=color, linewidth=2, label=label)


def plot_validation(ax, points, color, label=None) -> None:
    steps, values = zip(*points)
    ax.plot(
        steps,
        values,
        color=color,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label=label,
    )


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_rewards(metrics: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True, layout="constrained")
    for ax, (metric, title) in zip(axes.flat, REWARD_PANELS):
        plot_noisy(ax, metrics[metric], COLORS["grpo"])
        ax.set_title(title)
    fig.suptitle("GRPO rewards during training", x=0.01, ha="left", fontweight="semibold", color="#0b0b0b")
    fig.supxlabel(
        f"Training step. Thin line: each log. Thick line: mean of the last {SMOOTHING_WINDOW} logs.", fontsize=10
    )
    save(fig, out / "rewards.png")


def plot_accuracy(runs: dict, out: Path) -> None:
    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.5 * len(runs), 3.8), sharey=True, squeeze=False, layout="constrained"
    )
    for ax, (name, metrics) in zip(axes.flat, runs.items()):
        points = metrics["val/accuracy"]
        totals = [int(n) for _, n in metrics["val/n"]]
        steps, values = zip(*points)
        bounds = [wilson_interval(round(v * n), n) for v, n in zip(values, totals)]
        ax.fill_between(steps, *zip(*bounds), color=COLORS[name], alpha=0.1, linewidth=0)
        plot_validation(ax, points, COLORS[name])
        # Ties go to the earliest step, matching the checkpoint the monitor keeps.
        best_step, best = max(points, key=lambda point: point[1])
        align = "left" if best_step == steps[0] else "right" if best_step == steps[-1] else "center"
        ax.annotate(f"Best {best:.1%}", (best_step, best), xytext=(0, 10), textcoords="offset points", ha=align)
        ax.set_title(LABELS[name])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.suptitle(
        "Greedy accuracy on the validation set", x=0.01, ha="left", fontweight="semibold", color="#0b0b0b"
    )
    fig.supxlabel("Training step (steps differ in size between methods). Band: Wilson 95% interval.", fontsize=10)
    save(fig, out / "accuracy.png")


def plot_length(runs: dict, out: Path) -> None:
    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.5 * len(runs), 3.8), sharey=True, squeeze=False, layout="constrained"
    )
    for ax, (name, metrics) in zip(axes.flat, runs.items()):
        if name == "grpo":
            plot_noisy(
                ax,
                metrics["completions/mean_length"],
                MUTED_INK,
                f"Sampled in training (mean of last {SMOOTHING_WINDOW} logs)",
            )
            plot_validation(ax, metrics["val/mean_length_tokens"], COLORS[name], "Greedy validation")
            ax.legend(loc="best", fontsize=9)
        else:
            plot_validation(ax, metrics["val/mean_length_tokens"], COLORS[name])
        ax.set_title(LABELS[name])
    axes.flat[0].set_ylabel("Mean reply length (tokens)")
    fig.suptitle("Reply length during training", x=0.01, ha="left", fontweight="semibold", color="#0b0b0b")
    fig.supxlabel("Training step (steps differ in size between methods)", fontsize=10)
    save(fig, out / "length.png")


def plot_comparison(report: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(COMPARISON_PANELS), figsize=(9, 3.8), sharey=True, layout="constrained")
    models = report["models"]
    for ax, (metric, title) in zip(axes, COMPARISON_PANELS):
        for position, name in enumerate(MODELS):
            value, total = models[name][metric], models[name]["n"]
            low, high = wilson_interval(round(value * total), total)
            ax.bar(position, value, width=0.5, color=COLORS[name])
            ax.errorbar(
                position,
                value,
                # Rounding can put a bound a hair past the value at 0% or 100%, and matplotlib rejects negatives.
                yerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                color=SECONDARY_INK,
                linewidth=1,
                capsize=3,
            )
            ax.annotate(
                f"{value:.1%}", (position, high), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9
            )
        ax.set_xticks(range(len(MODELS)), [LABELS[name] for name in MODELS], color=SECONDARY_INK)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    total = models[MODELS[0]]["n"]
    fig.suptitle("Test-set results", x=0.01, ha="left", fontweight="semibold", color="#0b0b0b")
    fig.supxlabel(f"{total:,} problems. Error bars: Wilson 95% intervals.", fontsize=10)
    save(fig, out / "comparison.png")


def make_plots(source: Path, out: Path) -> None:
    """Draws whatever the run has produced so far, and never fills a gap with placeholder data."""
    out.mkdir(parents=True, exist_ok=True)
    runs = {name: read_metrics(source / f"metrics_{name}.csv") for name in TRAINED}
    runs = {name: metrics for name, metrics in runs.items() if metrics}
    if "grpo" in runs and all(metric in runs["grpo"] for metric, _ in REWARD_PANELS):
        plot_rewards(runs["grpo"], out)
    if runs:
        plot_accuracy(runs, out)
        plot_length(runs, out)
    comparison = source / "comparison.json"
    if comparison.exists():
        plot_comparison(json.loads(comparison.read_text(encoding="utf-8")), out)
    else:
        print("no comparison.json yet, skipping the comparison chart")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = results_dir(load_config(parser.parse_args().config))
    make_plots(source, source / "plots")


if __name__ == "__main__":
    main()
