import pytest

from src.data import SYSTEM_PROMPT, build_messages, split_solution, to_sft_target
from src.rewards import is_correct, is_format_valid

SOLUTION = "She sold 48/2 = <<48/2=24>>24 clips.\nTotal is 48+24 = <<48+24=72>>72.\n#### 72"


def test_split_solution():
    reasoning, answer = split_solution(SOLUTION)
    assert answer == "72"
    assert "####" not in reasoning


def test_split_solution_normalizes_commas():
    assert split_solution("Work.\n#### 1,000")[1] == "1000"


@pytest.mark.parametrize("bad", ["no separator 72", "Work.\n#### seventy"])
def test_split_solution_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        split_solution(bad)


def test_sft_target_strips_calculator_annotations():
    target = to_sft_target(SOLUTION)
    assert "<<" not in target and ">>" not in target


def test_sft_target_earns_full_marks_from_the_rewards():
    target = to_sft_target(SOLUTION)
    assert is_format_valid(target)
    assert is_correct(target, "72")


def test_build_messages_uses_the_fixed_prompt():
    messages = build_messages("What is 2+2?")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "What is 2+2?"}
