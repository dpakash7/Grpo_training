import pytest

from src.data import (
    EXAMPLE_REPLY,
    SYSTEM_PROMPT,
    build_messages,
    load_splits,
    render_prompt,
    split_solution,
    to_sft_target,
)
from src.rewards import PREFILL, format_reward, correctness_reward, is_correct, is_format_valid

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


def test_sft_target_continues_after_the_prefill():
    target = to_sft_target(SOLUTION)
    assert not target.startswith(PREFILL)
    assert is_format_valid(PREFILL + target)
    assert is_correct(PREFILL + target, "72")


def test_rewards_score_an_sft_target_as_if_it_were_generated():
    target = to_sft_target(SOLUTION)
    assert format_reward([target]) == [1.0]
    assert correctness_reward([target], answer=["72"]) == [1.0]


def test_build_messages_ends_with_the_question_and_prefill():
    messages = build_messages("What is 2+2?")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-2] == {"role": "user", "content": "What is 2+2?"}
    assert messages[-1] == {"role": "assistant", "content": PREFILL}


def test_the_worked_example_obeys_the_format_it_teaches():
    assert is_format_valid(EXAMPLE_REPLY)
    assert is_correct(EXAMPLE_REPLY, "41")


@pytest.fixture(scope="module")
def splits():
    return load_splits(val_size=50, seed=42)


def test_loaded_splits_use_the_current_prompt_not_a_stale_cache(splits):
    for name in ("train", "validation", "test"):
        row = splits[name][0]
        expected = build_messages(row["prompt"][-2]["content"])
        assert row["prompt"] == expected
        assert not row["target"].startswith(PREFILL)


def test_splits_do_not_overlap(splits):
    def questions(name):
        return {messages[-2]["content"] for messages in splits[name]["prompt"]}

    train, validation, test = questions("train"), questions("validation"), questions("test")
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(splits["validation"]) == 50


def test_render_prompt_continues_the_final_message():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, continue_final_message):
            assert tokenize is False and continue_final_message is True
            return "rendered"

    assert render_prompt(Tokenizer(), []) == "rendered"
