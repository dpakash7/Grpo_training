import pytest

from src.eval import compare, mcnemar_exact, score, summarize, wilson_interval
from src.train_sft import IGNORE_INDEX, make_collator, tokenize_example


def row(correct, gold="1"):
    return {
        "gold": gold,
        "correct": correct,
        "lenient_correct": correct,
        "format_valid": True,
        "length_tokens": 10,
        "clipped": False,
    }


def test_wilson_matches_a_known_value():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_wilson_stays_in_range_at_the_extremes():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    "a, b, expected",
    [(0, 0, 1.0), (5, 5, 1.0), (0, 5, 0.0625), (1, 9, 0.021484375), (2, 1, 1.0)],
)
def test_mcnemar_known_values(a, b, expected):
    assert mcnemar_exact(a, b) == pytest.approx(expected)


def test_score_and_summarize():
    rows = score(
        ["<think>x</think><answer>4</answer>", "<answer>5</answer>"],
        [10, 20],
        [False, True],
        ["4", "4"],
    )
    summary = summarize(rows)
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["format_valid_rate"] == 0.5
    assert summary["mean_length_tokens"] == 15
    assert summary["clipped_rate"] == 0.5


def test_compare_counts_only_the_disagreements():
    report = compare(
        {
            "a": [row(True), row(True), row(False)],
            "b": [row(True), row(False), row(False)],
        }
    )
    pair = report["pairs"]["a_vs_b"]
    assert pair["only_a_correct"] == 1
    assert pair["only_b_correct"] == 0
    assert pair["mcnemar_p"] == 1.0


def test_compare_refuses_different_problem_sets():
    with pytest.raises(ValueError):
        compare({"a": [row(True, gold="1")], "b": [row(True, gold="2")]})


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, continue_final_message):
        return "p p p"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) % 50 for c in text.split()]}


def test_sft_labels_mask_the_prompt_and_end_with_eos():
    example = tokenize_example(FakeTokenizer(), [], "a b", max_length=100)
    assert len(example["input_ids"]) == 6
    assert example["labels"][:3] == [IGNORE_INDEX] * 3
    assert example["labels"][3:5] == example["input_ids"][3:5]
    assert example["labels"][-1] == 99


def test_sft_truncation_keeps_inputs_and_labels_aligned():
    example = tokenize_example(FakeTokenizer(), [], "a b c d", max_length=5)
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"]) == 5


def test_collator_pads_right_and_ignores_padding_in_the_loss():
    batch = make_collator(0)(
        [
            {"input_ids": [1, 2], "labels": [IGNORE_INDEX, 2], "attention_mask": [1, 1]},
            {"input_ids": [3], "labels": [3], "attention_mask": [1]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 2], [3, IGNORE_INDEX]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
