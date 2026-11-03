import argparse
import json

from peft import LoraConfig
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from src.common import (
    TrainingMonitor,
    checkpoint_dir,
    load_config,
    load_tokenizer,
    precision,
    record_environment,
    results_dir,
)
from src.data import load_splits, render_prompt
from src.eval import make_validator
from src.rewards import PREFILL, correctness_reward, format_reward, make_length_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    grpo, rewards, lora = cfg["grpo"], cfg["rewards"], cfg["lora"]

    set_seed(cfg["seed"])
    record_environment(cfg, "grpo")
    splits = load_splits(cfg["val_size"], cfg["seed"])
    tokenizer = load_tokenizer(cfg["model"])
    mode = precision()

    args = GRPOConfig(
        output_dir=str(checkpoint_dir(cfg, "grpo")),
        seed=cfg["seed"],
        model_init_kwargs={"dtype": mode["dtype"]},
        bf16=mode["bf16"],
        fp16=mode["fp16"],
        learning_rate=grpo["learning_rate"],
        lr_scheduler_type="constant",
        num_generations=grpo["num_generations"],
        per_device_train_batch_size=grpo["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo["gradient_accumulation_steps"],
        max_completion_length=cfg["max_completion_tokens"],
        beta=grpo["beta"],
        scale_rewards=grpo["scale_rewards"],
        temperature=grpo["temperature"],
        reward_weights=[rewards["correctness"], rewards["format"], rewards["length"]],
        max_grad_norm=grpo["max_grad_norm"],
        max_steps=grpo["max_steps"],
        logging_steps=grpo["logging_steps"],
        save_steps=grpo["save_steps"],
        save_total_limit=2,
        report_to="none",
    )
    # TRL renders message prompts with add_generation_prompt=True, which would close the pre-filled
    # <think> turn. Rendered strings are tokenized as-is, so training samples from the prompt eval uses.
    train = splits["train"].remove_columns("target").map(
        lambda row: {"prompt": render_prompt(tokenizer, row["prompt"])}
    )
    if not train[0]["prompt"].endswith(PREFILL):
        raise RuntimeError("GRPO prompt does not end with the pre-filled tag")
    monitor = TrainingMonitor(cfg, "grpo", make_validator(splits["validation"], cfg))
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[
            correctness_reward,
            format_reward,
            make_length_penalty(rewards["free_chars"], rewards["max_chars"]),
        ],
        args=args,
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
        callbacks=[monitor],
    )
    trainer.train()

    summary = monitor.summary(trainer.state)
    summary["completions_sampled"] = (
        summary["steps"] * grpo["per_device_train_batch_size"] * grpo["gradient_accumulation_steps"]
    )
    (results_dir(cfg) / "run_grpo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
