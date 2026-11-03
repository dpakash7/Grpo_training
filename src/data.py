import re

from src.rewards import PREFILL, normalize_number

SYSTEM_PROMPT = (
    "You solve math word problems. Reason step by step inside <think></think>, "
    "then give only the final number inside <answer></answer>."
)
# Hand-written, not from GSM8K. Chosen on validation data: without it and the pre-filled tag the base model almost never uses the format.
EXAMPLE_QUESTION = (
    "A shop packs 12 pens in each box. It sells 3 boxes and gives away 5 pens "
    "from a fourth box. How many pens left the shop?"
)
EXAMPLE_REPLY = (
    "<think>3 boxes hold 3 * 12 = 36 pens. Adding the 5 given away makes 36 + 5 = 41 pens.</think>\n"
    "<answer>41</answer>"
)
_CALCULATOR_RE = re.compile(r"<<.*?>>")


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_QUESTION},
        {"role": "assistant", "content": EXAMPLE_REPLY},
        {"role": "user", "content": question},
        {"role": "assistant", "content": PREFILL},
    ]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


def split_solution(solution: str) -> tuple[str, str]:
    """Split a GSM8K solution into (reasoning, canonical final answer)."""
    reasoning, separator, final = solution.rpartition("####")
    answer = normalize_number(final)
    if not separator or answer is None:
        raise ValueError(f"Unparseable GSM8K solution: {solution!r}")
    return reasoning.strip(), answer


def to_sft_target(solution: str) -> str:
    """The reply as it continues after the pre-filled <think> tag."""
    reasoning, answer = split_solution(solution)
    reasoning = _CALCULATOR_RE.sub("", reasoning)
    return f"{reasoning}</think>\n<answer>{answer}</answer>"


def _to_example(row: dict) -> dict:
    return {
        "prompt": build_messages(row["question"]),
        "answer": split_solution(row["answer"])[1],
        "target": to_sft_target(row["answer"]),
    }


def load_splits(val_size: int, seed: int) -> dict:
    """Train and validation come from GSM8K train; the test split is only for final evaluation."""
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main")
    split = raw["train"].train_test_split(test_size=val_size, seed=seed)

    def prepare(dataset):
        # The cache keys on _to_example alone, so a prompt change would silently reuse stale prompts.
        return dataset.map(_to_example, remove_columns=dataset.column_names, load_from_cache_file=False)

    return {
        "train": prepare(split["train"]),
        "validation": prepare(split["test"]),
        "test": prepare(raw["test"]),
    }
