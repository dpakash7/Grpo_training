import pytest

from src.rewards import (
    PREFILL,
    correctness_reward,
    format_reward,
    is_correct,
    is_format_valid,
    is_lenient_correct,
    last_number,
    make_length_penalty,
    normalize_number,
    parse_answer,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("18", "18"),
        ("  18  ", "18"),
        ("\n18\n", "18"),
        ("1,000", "1000"),
        ("1,234,567", "1234567"),
        ("1,000.50", "1000.5"),
        ("-5", "-5"),
        ("+5", "5"),
        ("-1,000", "-1000"),
        ("$18", "18"),
        ("-$5", "-5"),
        ("$-5", "-5"),
        ("18.", "18"),
        ("18.00", "18"),
        ("18.50", "18.5"),
        (".5", "0.5"),
        ("100", "100"),
        ("0", "0"),
        ("-0", "0"),
        ("0.0", "0"),
        ("007", "7"),
    ],
)
def test_normalize_number_accepts(raw, expected):
    assert normalize_number(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        ".",
        "abc",
        "18 dollars",
        "x = 18",
        "18 + 2",
        "1,2,3",
        "1,00",
        "10,00,000",
        "1e5",
        "--5",
        "18..",
        "$",
        "١٨",
        "NaN",
        "Infinity",
    ],
)
def test_normalize_number_rejects(raw):
    assert normalize_number(raw) is None


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("<think>a</think><answer>18</answer>", "18"),
        ("<answer> 1,000 </answer>", "1000"),
        ("<answer>\n-5\n</answer>", "-5"),
        ("<answer>1</answer> then <answer>2</answer>", "2"),
        ("<answer>1</answer><answer>not a number</answer>", None),
        ("no tags at all, the answer is 18", None),
        ("<answer>18", None),
        ("18</answer>", None),
        ("<answer></answer>", None),
        ("<ANSWER>18</ANSWER>", None),
    ],
)
def test_parse_answer(completion, expected):
    assert parse_answer(completion) == expected


@pytest.mark.parametrize(
    "completion",
    [
        "<think>2+2=4</think><answer>4</answer>",
        "  <think>2+2=4</think>\n\n<answer> 4 </answer>\n",
        "<think>line one\nline two</think>\n<answer>1,000</answer>",
    ],
)
def test_format_valid(completion):
    assert is_format_valid(completion)


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "4",
        "<answer>4</answer>",
        "<think>x</think>",
        "<think></think><answer>4</answer>",
        "<think>   </think><answer>4</answer>",
        "<answer>4</answer><think>x</think>",
        "<think>x</think><answer>four</answer>",
        "<think>x</think><answer>1</answer><answer>2</answer>",
        "<think>a</think><think>b</think><answer>4</answer>",
        "<think>a<answer>3</answer></think><answer>4</answer>",
        "preamble <think>x</think><answer>4</answer>",
        "<think>x</think><answer>4</answer> trailing",
        "<think>x</think>middle<answer>4</answer>",
    ],
)
def test_format_invalid(completion):
    assert not is_format_valid(completion)


def test_is_correct_compares_numbers_not_strings():
    assert is_correct("<answer>1,000.00</answer>", "1000")
    assert is_correct("<answer>$18</answer>", "18")
    assert not is_correct("<answer>19</answer>", "18")
    assert not is_correct("the answer is 18", "18")


def test_hedging_does_not_earn_credit_unless_last_guess_is_right():
    assert not is_correct("<answer>18</answer><answer>19</answer>", "18")
    assert is_correct("<answer>19</answer><answer>18</answer>", "18")


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("So it takes 3 bolts of fiber.", "3"),
        ("First 2, then 1,000 total.", "1000"),
        ("The cost is $18.50.", "18.5"),
        ("<think>2+2</think><answer>4</answer>", "4"),
        ("no digits here", None),
    ],
)
def test_last_number(completion, expected):
    assert last_number(completion) == expected


def test_lenient_scoring_credits_right_maths_in_the_wrong_format():
    completion = "So it takes a total of 3 bolts."
    assert is_lenient_correct(completion, "3")
    assert not is_correct(completion, "3")


def test_rewards_restore_the_prefilled_tag_before_scoring():
    generated = "2+2=4</think><answer>4</answer>"
    assert not is_format_valid(generated)
    assert is_format_valid(PREFILL + generated)
    assert format_reward([generated]) == [1.0]


def test_a_doubled_think_tag_is_not_rewarded():
    assert format_reward(["<think>2+2=4</think><answer>4</answer>"]) == [0.0]


def test_correctness_reward_batch():
    completions = ["<answer>1</answer>", "<answer>3</answer>", "nothing"]
    assert correctness_reward(completions, answer=["1", "2", "3"]) == [1.0, 0.0, 0.0]


def test_rewards_accept_conversational_completions():
    completion = [{"role": "assistant", "content": "x</think><answer>4</answer>"}]
    assert correctness_reward([completion], answer=["4"]) == [1.0]
    assert format_reward([completion]) == [1.0]


def test_rewards_ignore_extra_trainer_kwargs():
    assert format_reward(["x"], prompts=["p"], trainer_state=None, question=["q"]) == [0.0]


def test_format_reward_alone_cannot_mark_a_wrong_answer_correct():
    wrong = "x</think><answer>5</answer>"
    assert format_reward([wrong]) == [1.0]
    assert correctness_reward([wrong], answer=["4"]) == [0.0]


def test_length_penalty_shape():
    # The pre-filled tag counts toward length, so sizes here are offset by its length.
    offset = len(PREFILL)
    penalty = make_length_penalty(free_chars=10 + offset, max_chars=20 + offset)
    assert penalty.__name__ == "length_penalty"
    scores = penalty(["a" * 5, "a" * 10, "a" * 15, "a" * 20, "a" * 500])
    assert scores == [0.0, 0.0, -0.5, -1.0, -1.0]


def test_length_penalty_rejects_bad_range():
    with pytest.raises(ValueError):
        make_length_penalty(free_chars=20, max_chars=20)
