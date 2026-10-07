"""
Grading tests for FinQA. Needs the dataset JSON files in ./data (train.json,
dev.json, test.json); no sandbox or API key is used.
"""

import asyncio

import pytest

from finqa import FINQA_DATA, FinQA, SubmitAnswerInput, format_full_prompt, validate_numerical_answer


def _task(task_id: str) -> dict:
    return next(t for t in FINQA_DATA["train"] if t["id"] == task_id)


# (task id, answer, expected reward). The references are the tasks' exe_ans:
# LKQ/2009/page_66.pdf-2 -> 0.11852 (a percentage stored as a fraction),
# LKQ/2016/page_48.pdf-4 -> 2.60192 (a ratio).
CASES = [
    ("LKQ/2009/page_66.pdf-2", "11.85%", 1.0),
    ("LKQ/2009/page_66.pdf-2", "11.85", 1.0),
    ("LKQ/2009/page_66.pdf-2", "11.9%", 1.0),
    ("LKQ/2009/page_66.pdf-2", "0.1185", 1.0),
    ("LKQ/2009/page_66.pdf-2", "0.11852", 1.0),
    ("LKQ/2009/page_66.pdf-2", "11.95%", 1.0),  # 0.8% off: inside the 1% window
    ("LKQ/2009/page_66.pdf-2", "11.7%", 0.0),  # 1.3% off
    ("LKQ/2009/page_66.pdf-2", "12.85%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "-11.85%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "0.12", 0.0),
    ("LKQ/2009/page_66.pdf-2", "12%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "0", 0.0),
    ("LKQ/2016/page_48.pdf-4", "2.60", 1.0),
    ("LKQ/2016/page_48.pdf-4", "2.6", 1.0),
    ("LKQ/2016/page_48.pdf-4", "2.6019", 1.0),
    ("LKQ/2016/page_48.pdf-4", "2.61", 1.0),  # 0.3% off
    ("LKQ/2016/page_48.pdf-4", "2.63", 0.0),  # 1.1% off
    ("LKQ/2016/page_48.pdf-4", "3", 0.0),
    ("LKQ/2016/page_48.pdf-4", "26.0", 0.0),
    ("LKQ/2016/page_48.pdf-4", "about two and a half", 0.0),
]


@pytest.mark.parametrize("task_id,answer,expected", CASES)
def test_submit_answer_grading(task_id: str, answer: str, expected: float):
    env = FinQA(task_spec={"id": task_id, "split": "train"}, secrets={"api_key": ""})
    out = asyncio.run(env.submit_answer(SubmitAnswerInput(answer=answer)))
    assert out.reward == expected
    assert out.finished is True
    reference = str(_task(task_id)["qa"]["exe_ans"])
    assert reference not in out.blocks[0].text + str(out.metadata) or reference == answer


@pytest.mark.parametrize("submitted,expected,ok", [
    ("1,234.5", "1234.5", True),
    ("$1,234.5", "1234.5", True),
    ("3.54%", "3.53735", True),  # reference already in percent units
    ("3.5%", "3.53735", False),  # rounded to 1.06% off
    ("3.537", "3.53735", True),
    ("6.58%", "0.0658", True),
    ("6.6%", "0.0658", True),
    ("7%", "0.0658", False),  # 1 significant figure, 6.4% off
    ("0.07", "0.0658", False),
    ("3,380", "3376", True),  # 3 significant figures with trailing zero
    ("3,300", "3376", False),  # 2.3% off
    ("2", "2.0", True),  # exact small integer
    ("0.0", "0.0", True),
    ("0.1", "0.0", False),
    ("yes", "yes", True),
    ("No", "no", True),
    ("yes", "no", False),
])
def test_validate_numerical_answer(submitted: str, expected: str, ok: bool):
    assert validate_numerical_answer(submitted, expected) is ok


def _sig_fig(x: float, k: int) -> str:
    return f"{x:.{k}g}" if abs(x) < 10 ** k else str(round(x))


# References spanning fractions, percents already scaled, ratios and amounts.
COARSE_REFS = [0.11852, 0.0658, 0.01445, 3.53735, 2.60192, 0.63636, 1.04938, 497.9854, 3376.0, -0.03842]


@pytest.mark.parametrize("reference", COARSE_REFS)
def test_precise_answers_are_not_penalised_against_coarse_ones(reference: float):
    """The acceptance window does not depend on how many digits the answer gives."""
    for as_percent in (False, True):
        suffix = "%" if as_percent else ""
        scale = 100 if as_percent else 1
        # A correct value passes at every precision from 3 significant figures up.
        for k in (6, 4, 3):
            assert validate_numerical_answer(_sig_fig(reference * scale, k) + suffix, str(reference))
        # A value 0.8% off passes when written precisely, so rounding it to 1-2
        # significant figures can never be the better choice.
        for value in (reference * 1.008, reference * 0.992):
            assert validate_numerical_answer(_sig_fig(value * scale, 6) + suffix, str(reference))
        # A value 3% off fails however precisely it is written.
        for k in (6, 4, 3):
            assert not validate_numerical_answer(_sig_fig(reference * 1.03 * scale, k) + suffix, str(reference))


@pytest.mark.parametrize("coarse", ["0", "1", "10%", "0.1", "0%"])
def test_coarse_constant_answers_score_zero(coarse: str):
    reference = "0.11852"
    assert validate_numerical_answer(coarse, reference) is False


def test_prompt_states_required_precision():
    prompt = format_full_prompt(_task("LKQ/2009/page_66.pdf-2"))
    assert "3 significant figures" in prompt and "within 1%" in prompt
