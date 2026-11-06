from types import SimpleNamespace

import pytest

from src.common import TrainingMonitor


class FakeModel:
    def __init__(self):
        self.saved = 0

    def train(self):
        pass

    def save_pretrained(self, path):
        self.saved += 1


def run_monitor(monkeypatch, tmp_path, accuracies, patience, budget_minutes=1000):
    """Validates at the start and after every step, stopping when the monitor says so."""
    monkeypatch.chdir(tmp_path)
    cfg = {"run_name": "test", "budget_minutes": budget_minutes, "validate_every_minutes": 0}
    scores = iter(accuracies)
    monitor = TrainingMonitor(cfg, "sft", lambda model, tok: {"accuracy": next(scores)}, patience)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace(should_training_stop=False)
    model = FakeModel()
    monitor.on_train_begin(None, state, control, model=model)
    while not control.should_training_stop and state.global_step < len(accuracies) - 1:
        state.global_step += 1
        monitor.on_step_end(None, state, control, model=model)
    monitor.on_train_end(None, state, control, model=model)
    return monitor, state, model


def test_stops_after_patience_validations_without_improvement(monkeypatch, tmp_path):
    monitor, state, _ = run_monitor(monkeypatch, tmp_path, [0.5, 0.6, 0.6, 0.4, 0.9], patience=2)
    assert state.global_step == 3
    assert monitor.stopped_early
    assert monitor.best_step == 1


def test_improvement_resets_the_count(monkeypatch, tmp_path):
    monitor, state, _ = run_monitor(monkeypatch, tmp_path, [0.5, 0.4, 0.6, 0.4, 0.7], patience=2)
    assert state.global_step == 4
    assert not monitor.stopped_early
    assert monitor.best_accuracy == pytest.approx(0.7)


def test_no_patience_never_stops_early(monkeypatch, tmp_path):
    monitor, state, _ = run_monitor(monkeypatch, tmp_path, [0.5, 0.4, 0.4, 0.4], patience=None)
    assert state.global_step == 3
    assert not monitor.stopped_early


def test_a_tie_does_not_replace_the_best_adapter(monkeypatch, tmp_path):
    monitor, _, model = run_monitor(monkeypatch, tmp_path, [0.5, 0.5, 0.5], patience=None)
    assert monitor.best_step == 0
    assert model.saved == 1


def test_budget_still_stops_when_early_stopping_is_off(monkeypatch, tmp_path):
    monitor, state, _ = run_monitor(monkeypatch, tmp_path, [0.5, 0.6, 0.7], None, budget_minutes=0)
    assert state.global_step == 1
    assert not monitor.stopped_early
