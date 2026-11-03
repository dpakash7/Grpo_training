import argparse
import json
import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from src.common import (
    checkpoint_dir,
    load_config,
    load_tokenizer,
    precision,
    record_environment,
    results_dir,
)
from src.data import load_splits, render_prompt
from src.rewards import PREFILL, is_correct, is_format_valid, is_lenient_correct

MODELS = ("base", "sft", "grpo")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Two-sided exact McNemar p-value from the problems where the two models disagree."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a_correct, only_b_correct) + 1))
    return min(1.0, 2 * tail / 2**n)


def score(completions: list[str], lengths: list[int], clipped: list[bool], golds: list[str]) -> list[dict]:
    return [
        {
            "completion": completion,
            "gold": gold,
            "correct": is_correct(completion, gold),
            "lenient_correct": is_lenient_correct(completion, gold),
            "format_valid": is_format_valid(completion),
            "length_tokens": length,
            "clipped": was_clipped,
        }
        for completion, length, was_clipped, gold in zip(completions, lengths, clipped, golds)
    ]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "lenient_accuracy": sum(r["lenient_correct"] for r in rows) / n,
        "format_valid_rate": sum(r["format_valid"] for r in rows) / n,
        "mean_length_tokens": sum(r["length_tokens"] for r in rows) / n,
        "clipped_rate": sum(r["clipped"] for r in rows) / n,
    }


def compare(rows_by_model: dict[str, list[dict]]) -> dict:
    report = {"models": {}, "pairs": {}}
    for name, rows in rows_by_model.items():
        summary = summarize(rows)
        correct = sum(r["correct"] for r in rows)
        low, high = wilson_interval(correct, len(rows))
        report["models"][name] = {**summary, "accuracy_ci95": [low, high]}
    names = list(rows_by_model)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows_a, rows_b = rows_by_model[a], rows_by_model[b]
            if [r["gold"] for r in rows_a] != [r["gold"] for r in rows_b]:
                raise ValueError(f"{a} and {b} were not scored on the same problems")
            only_a = sum(x["correct"] and not y["correct"] for x, y in zip(rows_a, rows_b))
            only_b = sum(y["correct"] and not x["correct"] for x, y in zip(rows_a, rows_b))
            report["pairs"][f"{a}_vs_{b}"] = {
                f"only_{a}_correct": only_a,
                f"only_{b}_correct": only_b,
                "mcnemar_p": mcnemar_exact(only_a, only_b),
            }
    return report


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens, batch_size):
    model.eval()
    device = next(model.parameters()).device
    stop_ids = torch.tensor([tokenizer.pad_token_id, tokenizer.eos_token_id], device=device)
    completions, lengths, clipped = [], [], []
    for start in range(0, len(prompts), batch_size):
        texts = [render_prompt(tokenizer, p) for p in prompts[start : start + batch_size]]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # Qwen ships sampling defaults and a repetition penalty; greedy must override all of them.
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        is_stop = torch.isin(new_tokens, stop_ids)
        completions += [PREFILL + text for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
        lengths += (~is_stop).sum(dim=1).tolist()
        clipped += (~is_stop.any(dim=1)).tolist()
    return completions, lengths, clipped


def evaluate_dataset(model, tokenizer, dataset, cfg) -> list[dict]:
    completions, lengths, clipped = generate(
        model, tokenizer, dataset["prompt"], cfg["max_completion_tokens"], cfg["eval_batch_size"]
    )
    return score(completions, lengths, clipped, dataset["answer"])


def make_validator(dataset, cfg):
    def evaluate(model, tokenizer) -> dict:
        return summarize(evaluate_dataset(model, tokenizer, dataset, cfg))

    return evaluate


def load_model(cfg: dict, name: str):
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=precision()["dtype"])
    if name != "base":
        model = PeftModel.from_pretrained(model, checkpoint_dir(cfg, name) / "best")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def run_eval(cfg: dict, name: str) -> dict:
    set_seed(cfg["seed"])
    record_environment(cfg, f"eval_{name}")
    test = load_splits(cfg["val_size"], cfg["seed"])["test"]
    if cfg["eval_limit"] is not None:
        test = test.select(range(cfg["eval_limit"]))
    rows = evaluate_dataset(load_model(cfg, name), load_tokenizer(cfg["model"]), test, cfg)
    out = results_dir(cfg)
    with open(out / f"eval_{name}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = summarize(rows)
    (out / f"eval_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_compare(cfg: dict) -> dict:
    out = results_dir(cfg)
    rows_by_model = {}
    for name in MODELS:
        with open(out / f"eval_{name}.jsonl", encoding="utf-8") as f:
            rows_by_model[name] = [json.loads(line) for line in f]
    report = compare(rows_by_model)
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", choices=MODELS)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.compare:
        print(json.dumps(run_compare(cfg), indent=2))
    elif args.name:
        print(json.dumps(run_eval(cfg, args.name), indent=2))
    else:
        parser.error("pass --name or --compare")


if __name__ == "__main__":
    main()
