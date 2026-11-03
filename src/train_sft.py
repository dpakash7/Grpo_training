import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

from src.common import (
    TrainingMonitor,
    checkpoint_dir,
    load_config,
    load_tokenizer,
    precision,
    record_environment,
    results_dir,
)
from src.data import load_splits
from src.eval import make_validator

IGNORE_INDEX = -100


def tokenize_example(tokenizer, prompt: list[dict], target: str, max_length: int) -> dict:
    """Labels cover only the assistant reply, so the model is never trained to reproduce the prompt."""
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + target_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def make_collator(pad_token_id: int):
    def collate(features: list[dict]) -> dict:
        longest = max(len(f["input_ids"]) for f in features)

        def pad(key: str, value: int):
            return torch.tensor([f[key] + [value] * (longest - len(f[key])) for f in features])

        return {
            "input_ids": pad("input_ids", pad_token_id),
            "labels": pad("labels", IGNORE_INDEX),
            "attention_mask": pad("attention_mask", 0),
        }

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    sft, lora = cfg["sft"], cfg["lora"]

    set_seed(cfg["seed"])
    record_environment(cfg, "sft")
    splits = load_splits(cfg["val_size"], cfg["seed"])
    tokenizer = load_tokenizer(cfg["model"])
    mode = precision()

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=mode["dtype"])
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    train = splits["train"].map(
        lambda row: tokenize_example(tokenizer, row["prompt"], row["target"], sft["max_length"]),
        remove_columns=splits["train"].column_names,
    )
    args = TrainingArguments(
        output_dir=str(checkpoint_dir(cfg, "sft")),
        seed=cfg["seed"],
        bf16=mode["bf16"],
        fp16=mode["fp16"],
        learning_rate=sft["learning_rate"],
        lr_scheduler_type="constant",
        per_device_train_batch_size=sft["per_device_train_batch_size"],
        gradient_accumulation_steps=sft["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        max_grad_norm=sft["max_grad_norm"],
        max_steps=sft["max_steps"],
        logging_steps=sft["logging_steps"],
        save_steps=sft["save_steps"],
        save_total_limit=2,
        report_to="none",
    )
    monitor = TrainingMonitor(cfg, "sft", make_validator(splits["validation"], cfg))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        data_collator=make_collator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[monitor],
    )
    trainer.train()

    summary = monitor.summary(trainer.state)
    summary["examples_trained"] = (
        summary["steps"] * sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
    )
    (results_dir(cfg) / "run_sft.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
