"""Post-hoc checks after the full run, kept apart from the pre-planned comparison.

The base model scored far higher on validation than on test, and selection kept SFT's untrained step 0.
This measures whether the gap follows GSM8K's train/test split, and how a trained SFT adapter really scores.
Run from the repo root: python -m scripts.diagnose --config configs/default.yaml --sft-checkpoint checkpoint-1500
"""
import argparse
import json

from peft import PeftModel
from transformers import set_seed

from src.common import checkpoint_dir, load_config, load_tokenizer, record_environment, results_dir
from src.data import load_splits
from src.eval import compare, evaluate_dataset, load_model, summarize, wilson_interval

TRAIN_SAMPLE_SIZE = 1000


def describe(rows: list[dict]) -> dict:
    low, high = wilson_interval(sum(r["correct"] for r in rows), len(rows))
    return {**summarize(rows), "accuracy_ci95": [low, high]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    record_environment(cfg, "diagnose")

    out = results_dir(cfg)
    splits = load_splits(cfg["val_size"], cfg["seed"])
    tokenizer = load_tokenizer(cfg["model"])
    train_sample = splits["train"].shuffle(seed=cfg["seed"]).select(range(TRAIN_SAMPLE_SIZE))
    report = {"sft_checkpoint": args.sft_checkpoint, "train_sample_size": TRAIN_SAMPLE_SIZE}

    def checkpoint() -> None:
        (out / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    model = load_model(cfg, "base")
    report["base_validation"] = describe(evaluate_dataset(model, tokenizer, splits["validation"], cfg))
    checkpoint()
    report["base_train_sample"] = describe(evaluate_dataset(model, tokenizer, train_sample, cfg))
    checkpoint()

    model = PeftModel.from_pretrained(model, checkpoint_dir(cfg, "sft") / args.sft_checkpoint)
    report["sft_checkpoint_validation"] = describe(evaluate_dataset(model, tokenizer, splits["validation"], cfg))
    checkpoint()
    sft_test = evaluate_dataset(model, tokenizer, splits["test"], cfg)
    report["sft_checkpoint_test"] = describe(sft_test)

    rows_by_model = {"sft_checkpoint": sft_test}
    for name in ("base", "grpo"):
        with open(out / f"eval_{name}.jsonl", encoding="utf-8") as f:
            rows_by_model[name] = [json.loads(line) for line in f]
    report["test_pairs"] = compare(rows_by_model)["pairs"]
    checkpoint()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
