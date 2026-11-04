"""Reproduces how the prompt was chosen: five variants on the base model, validation split only.
Run from the repo root: python -m scripts.compare_prompts --config configs/default.yaml
"""
import argparse
import json

import torch
from transformers import set_seed

from src.common import load_config, load_tokenizer, record_environment, results_dir
from src.data import EXAMPLE_QUESTION, EXAMPLE_REPLY, SYSTEM_PROMPT, load_splits
from src.eval import load_model, score, summarize
from src.rewards import PREFILL


def variants(question: str) -> dict[str, list[dict]]:
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user = {"role": "user", "content": question}
    prefill = {"role": "assistant", "content": PREFILL}
    example = [
        {"role": "user", "content": EXAMPLE_QUESTION},
        {"role": "assistant", "content": EXAMPLE_REPLY},
    ]
    in_system = f"{SYSTEM_PROMPT}\n\nExample question: {EXAMPLE_QUESTION}\nExample reply:\n{EXAMPLE_REPLY}"
    return {
        "A_instruction_only": [system, user],
        "B_example_in_system": [{"role": "system", "content": in_system}, user],
        "C_example_as_turn": [system, *example, user],
        "D_prefill_only": [system, user, prefill],
        "E_example_turn_and_prefill": [system, *example, user, prefill],
    }


@torch.inference_mode()
def run(model, tokenizer, conversations, cfg) -> tuple[list[str], list[int], list[bool]]:
    prefilled = conversations[0][-1]["role"] == "assistant"
    texts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=prefilled, add_generation_prompt=not prefilled
        )
        for messages in conversations
    ]
    stop_ids = torch.tensor([tokenizer.pad_token_id, tokenizer.eos_token_id], device=model.device)
    completions, lengths, clipped = [], [], []
    for start in range(0, len(texts), cfg["eval_batch_size"]):
        inputs = tokenizer(
            texts[start : start + cfg["eval_batch_size"]], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=cfg["max_completion_tokens"],
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
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        completions += [PREFILL + text if prefilled else text for text in decoded]
        lengths += (~is_stop).sum(dim=1).tolist()
        clipped += (~is_stop.any(dim=1)).tolist()
    return completions, lengths, clipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    record_environment(cfg, "prompts")
    validation = load_splits(cfg["val_size"], cfg["seed"])["validation"]
    questions = [messages[-2]["content"] for messages in validation["prompt"]]
    tokenizer = load_tokenizer(cfg["model"])
    model = load_model(cfg, "base").eval()

    report = {}
    for name in variants(""):
        conversations = [variants(question)[name] for question in questions]
        rows = score(*run(model, tokenizer, conversations, cfg), validation["answer"])
        report[name] = summarize(rows)
        print(name, json.dumps(report[name]), flush=True)
        path = results_dir(cfg) / "prompt_selection.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
