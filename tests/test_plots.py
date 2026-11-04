import csv
import json

import pytest

from src.plots import make_plots, read_metrics, rolling_mean


def write_metrics(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "training_seconds", "metric", "value"])
        for step, metric, value in rows:
            writer.writerow([step, step * 20.0, metric, value])


def validation_rows(step, accuracy):
    return [(step, "val/n", 200), (step, "val/accuracy", accuracy), (step, "val/mean_length_tokens", 110)]


def grpo_rows():
    rows = validation_rows(0, 0.5) + validation_rows(10, 0.6)
    for step in (5, 10):
        for metric in (
            "reward",
            "rewards/correctness_reward/mean",
            "rewards/format_reward/mean",
            "rewards/length_penalty/mean",
            "completions/mean_length",
        ):
            rows.append((step, metric, 0.5))
    return rows


def comparison_report():
    summary = {"n": 100, "accuracy": 0.5, "lenient_accuracy": 0.6, "format_valid_rate": 1.0}
    return {"models": {name: summary for name in ("base", "sft", "grpo")}, "pairs": {}}


def test_rolling_mean_uses_only_past_values():
    assert rolling_mean([1.0, 3.0, 5.0, 7.0], 2) == pytest.approx([1.0, 2.0, 4.0, 6.0])


def test_read_metrics_groups_points_by_metric(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics(path, [(0, "reward", 0.1), (5, "kl", 0.2), (5, "reward", 0.3)])
    assert read_metrics(path) == {"reward": [(0, 0.1), (5, 0.3)], "kl": [(5, 0.2)]}


def test_read_metrics_treats_a_missing_file_as_no_data(tmp_path):
    assert read_metrics(tmp_path / "missing.csv") == {}


def test_nothing_is_drawn_for_an_empty_run(tmp_path):
    make_plots(tmp_path, tmp_path / "plots")
    assert list((tmp_path / "plots").iterdir()) == []


def test_a_run_in_progress_gets_only_the_charts_its_data_supports(tmp_path):
    write_metrics(tmp_path / "metrics_grpo.csv", grpo_rows())
    make_plots(tmp_path, tmp_path / "plots")
    drawn = {path.name for path in (tmp_path / "plots").iterdir()}
    assert drawn == {"rewards.png", "accuracy.png", "length.png"}


def test_a_finished_run_gets_every_chart(tmp_path):
    write_metrics(tmp_path / "metrics_grpo.csv", grpo_rows())
    write_metrics(tmp_path / "metrics_sft.csv", validation_rows(0, 0.5))
    (tmp_path / "comparison.json").write_text(json.dumps(comparison_report()), encoding="utf-8")
    make_plots(tmp_path, tmp_path / "plots")
    drawn = {path.name for path in (tmp_path / "plots").iterdir()}
    assert drawn == {"rewards.png", "accuracy.png", "length.png", "comparison.png"}
