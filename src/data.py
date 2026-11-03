import re

from src.rewards import normalize_number

SYSTEM_PROMPT = (
    "You solve math word problems. Reason step by step inside <think></think>, "
    "then give only the final number inside <answer></answer>."
)
_CALCULATOR_RE = re.compile(r"<<.*?>>")


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def split_solution(solution: str) -> tuple[str, str]:
    """Split a GSM8K solution into (reasoning, canonical final answer)."""
    reasoning, separator, final = solution.rpartition("####")
    answer = normalize_number(final)
    if not separator or answer is None:
        raise ValueError(f"Unparseable GSM8K solution: {solution!r}")
    return reasoning.strip(), answer


def to_sft_target(solution: str) -> str:
    reasoning, answer = split_solution(solution)
    reasoning = _CALCULATOR_RE.sub("", reasoning)
    return f"<think>{reasoning}</think>\n<answer>{answer}</answer>"


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
    columns = raw["train"].column_names
    split = raw["train"].train_test_split(test_size=val_size, seed=seed)
    return {
        "train": split["train"].map(_to_example, remove_columns=columns),
        "validation": split["test"].map(_to_example, remove_columns=columns),
        "test": raw["test"].map(_to_example, remove_columns=columns),
    }
