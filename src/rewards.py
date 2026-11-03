import re
from decimal import Decimal, InvalidOperation

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FORMAT_RE = re.compile(
    r"\s*<think>(?P<think>.+?)</think>\s*<answer>(?P<answer>.+?)</answer>\s*", re.DOTALL
)
_PLAIN_NUMBER_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)", re.ASCII)
_GROUPED_NUMBER_RE = re.compile(r"[+-]?\d{1,3}(,\d{3})+(\.\d*)?", re.ASCII)
_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def normalize_number(raw: str) -> str | None:
    """Return a canonical number string, or None if raw is not purely a number."""
    text = raw.strip()
    # Accept a currency sign on either side of the sign: "$-5" and "-$5".
    text = re.sub(r"^([+-]?)\$", r"\1", text)
    text = re.sub(r"^\$([+-])", r"\1", text)
    if _GROUPED_NUMBER_RE.fullmatch(text):
        text = text.replace(",", "")
    elif not _PLAIN_NUMBER_RE.fullmatch(text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def parse_answer(completion: str) -> str | None:
    """Normalized number from the last <answer> tag. Last-wins means listing several guesses earns nothing extra."""
    matches = _ANSWER_RE.findall(completion)
    return normalize_number(matches[-1]) if matches else None


def is_format_valid(completion: str) -> bool:
    match = _FORMAT_RE.fullmatch(completion)
    if match is None or any(completion.count(tag) != 1 for tag in _TAGS):
        return False
    if not match["think"].strip():
        return False
    return normalize_number(match["answer"]) is not None


def is_correct(completion: str, gold: str) -> bool:
    predicted = parse_answer(completion)
    return predicted is not None and predicted == normalize_number(gold)


def _text(completion) -> str:
    # TRL passes plain strings for string prompts and message lists for conversational ones.
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    return [float(is_correct(_text(c), gold)) for c, gold in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    return [float(is_format_valid(_text(c))) for c in completions]


def make_length_penalty(free_chars: int, max_chars: int):
    """Zero up to free_chars, falling linearly to -1 at max_chars and staying there."""
    span = max_chars - free_chars
    if span <= 0:
        raise ValueError("max_chars must be greater than free_chars")

    def length_penalty(completions, **kwargs) -> list[float]:
        return [
            -min(max(len(_text(c)) - free_chars, 0) / span, 1.0) for c in completions
        ]

    return length_penalty
