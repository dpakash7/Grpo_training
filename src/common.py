import csv
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, TrainerCallback

PACKAGES = ["torch", "trl", "transformers", "peft", "datasets", "accelerate"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir(cfg: dict) -> Path:
    path = Path("results") / cfg["run_name"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(cfg: dict, name: str) -> Path:
    return Path("checkpoints") / cfg["run_name"] / name


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    return tokenizer


def precision() -> dict:
    """bf16 where the GPU supports it, otherwise fp32 weights with fp16 autocast (T4)."""
    if not torch.cuda.is_available():
        return {"dtype": "float32", "bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"dtype": "bfloat16", "bf16": True, "fp16": False}
    return {"dtype": "float32", "bf16": False, "fp16": True}


def record_environment(cfg: dict, name: str) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    info = {
        "python": platform.python_version(),
        "packages": {package: version(package) for package in PACKAGES},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "config": cfg,
    }
    path = results_dir(cfg) / f"env_{name}.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


class TrainingMonitor(TrainerCallback):
    """Logs metrics to CSV, validates on a shared time schedule, keeps the best adapter, enforces the budget."""

    def __init__(self, cfg, name, evaluate, patience=None):
        self.budget_seconds = cfg["budget_minutes"] * 60
        self.validate_every_seconds = cfg["validate_every_minutes"] * 60
        self.evaluate = evaluate
        self.patience = patience
        self.best_dir = checkpoint_dir(cfg, name) / "best"
        self.metrics_path = results_dir(cfg) / f"metrics_{name}.csv"
        self.best_accuracy = -1.0
        self.best_step = 0
        self.validation_seconds = 0.0
        self.last_validated_step = None
        self.validations_since_best = 0
        self.stopped_early = False

    def training_seconds(self) -> float:
        # Validation time is excluded so it cannot eat into either model's training budget.
        return time.monotonic() - self.started - self.validation_seconds

    def _write(self, step: int, metric: str, value: float) -> None:
        self.writer.writerow([step, round(self.training_seconds(), 1), metric, value])
        self.file.flush()

    def _validate(self, state, model, tokenizer) -> None:
        started = time.monotonic()
        summary = self.evaluate(model, tokenizer)
        model.train()
        # Validation's larger batches would otherwise stay reserved and push training out of VRAM.
        torch.cuda.empty_cache()
        if summary["accuracy"] > self.best_accuracy:
            self.best_accuracy = summary["accuracy"]
            self.best_step = state.global_step
            self.validations_since_best = 0
            model.save_pretrained(self.best_dir)
        else:
            self.validations_since_best += 1
        self.validation_seconds += time.monotonic() - started
        self.last_validated_step = state.global_step
        self.next_validation = self.training_seconds() + self.validate_every_seconds
        for metric, value in summary.items():
            self._write(state.global_step, f"val/{metric}", value)

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        self.file = open(self.metrics_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "training_seconds", "metric", "value"])
        self.started = time.monotonic()
        self._validate(state, model, processing_class)

    def on_log(self, args, state, control, logs=None, **kwargs):
        for metric, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                self._write(state.global_step, metric, value)

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self.training_seconds() >= self.next_validation:
            self._validate(state, model, processing_class)
        if self.patience is not None and self.validations_since_best >= self.patience:
            self.stopped_early = True
            control.should_training_stop = True
        if self.training_seconds() >= self.budget_seconds:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self.last_validated_step != state.global_step:
            self._validate(state, model, processing_class)
        self.file.close()

    def summary(self, state) -> dict:
        return {
            "steps": state.global_step,
            "training_minutes": round(self.training_seconds() / 60, 2),
            "validation_minutes": round(self.validation_seconds / 60, 2),
            "best_step": self.best_step,
            "best_val_accuracy": self.best_accuracy,
            "stopped_early": self.stopped_early,
        }
