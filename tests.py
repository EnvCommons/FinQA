"""
Grading tests for FinQA. Needs the dataset JSON files in ./data (train.json,
dev.json, test.json); no sandbox or API key is used.
"""

import asyncio

import pytest

from finqa import (
    FINQA_DATA,
    FinQA,
    SubmitAnswerInput,
    format_full_prompt,
    has_consistent_reference,
    stated_units,
    validate_numerical_answer,
)


def _task(task_id: str) -> dict:
    return next(t for t in FINQA_DATA["train"] if t["id"] == task_id)


# (task id, answer, expected reward). The references are the tasks' exe_ans:
# LKQ/2009/page_66.pdf-2 -> 0.11852 (a percentage stored as a fraction),
# LKQ/2016/page_48.pdf-4 -> 2.60192 (a ratio),
# HUM/2012/page_109.pdf-2 -> 95.0 (an amount in a table stated "in millions").
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
    ("HUM/2012/page_109.pdf-2", "95", 1.0),
    ("HUM/2012/page_109.pdf-2", "95 million", 1.0),
    ("HUM/2012/page_109.pdf-2", "$95 Million", 1.0),
    ("HUM/2012/page_109.pdf-2", "95M", 1.0),
    ("HUM/2012/page_109.pdf-2", "0.095 billion", 1.0),
    ("HUM/2012/page_109.pdf-2", "95,000 thousand", 1.0),
    ("HUM/2012/page_109.pdf-2", "95 billion", 0.0),  # a scale the document does not use
    ("HUM/2012/page_109.pdf-2", "95 thousand", 0.0),
    ("HUM/2012/page_109.pdf-2", "96.5 million", 0.0),  # 1.6% off
    ("HUM/2012/page_109.pdf-2", "95000000", 0.0),  # a bare number is read in the document's units
    ("HUM/2012/page_109.pdf-2", "95 million dollars", 0.0),
    ("LKQ/2016/page_48.pdf-4", "2.6 million", 0.0),  # the document states thousands
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


MILLIONS = (1e6,)


@pytest.mark.parametrize("submitted,expected,units,ok", [
    ("95 million", "95000000.0", (), True),  # the reference is a full amount
    ("1.2bn", "1200000000.0", (), True),
    ("95 million", "95.0", (), False),  # no stated unit to read the reference in
    ("95 million", "95.0", MILLIONS, True),
    ("95 millions", "95.0", MILLIONS, True),
    ("95 mm", "95.0", MILLIONS, True),
    ("-95 million", "-95.0", MILLIONS, True),
    ("-95 million", "95.0", MILLIONS, False),
    ("95 billion", "95.0", MILLIONS, False),
    ("95 thousand", "95.0", MILLIONS, False),
    ("95 thousand", "95.0", (1e3, 1e6), True),  # both scales are stated
    ("11.85 million", "0.11852", MILLIONS, False),  # no percent scaling with a scale word
    ("11.85% million", "0.11852", MILLIONS, False),
    ("95 million", "0.0", MILLIONS, False),
    ("0 million", "0.0", MILLIONS, True),
    ("95 miles", "95.0", MILLIONS, False),
    ("million", "95.0", MILLIONS, False),
])
def test_scale_words(submitted: str, expected: str, units: tuple, ok: bool):
    assert validate_numerical_answer(submitted, expected, units) is ok


def test_stated_units():
    assert stated_units(_task("HUM/2012/page_109.pdf-2")) == (1e6,)
    assert stated_units(_task("LKQ/2016/page_48.pdf-4")) == (1e3,)
    assert stated_units(_task("HII/2018/page_103.pdf-2")) == ()


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


@pytest.mark.parametrize("answer,exe_ans,ok", [
    ("35.72%", 0.35715, True),
    ("35.72%", 0.73684, False),  # program divides the wrong way round
    ("11.85%", 0.11852, True),
    ("7%", 0.0658, True),  # coarsely rounded
    ("7.7%", 7.78443, True),  # truncated
    ("14%", 0.13174, False),  # neither rounded nor truncated, 6% off
    ("16.7%", -0.16667, False),  # opposite sign
    ("2.58", 0.00258, False),  # different units
    ("0.1", 0.01, False),
    ("$ 13 million", 13.0, True),
    ("1,234", 1234.0, True),
    ("", 22929.0, True),  # no written answer to check against
    ("yes", "yes", False),
    ("no", "no", False),
])
def test_has_consistent_reference(answer, exe_ans, ok: bool):
    assert has_consistent_reference({"answer": answer, "exe_ans": exe_ans}) is ok


# STT/2006/page_92.pdf-4: exe_ans 0.73684 from a wrong program, written answer 35.72%.
# SLG/2018/page_45.pdf-1: a yes/no question.
@pytest.mark.parametrize("task_id", ["STT/2006/page_92.pdf-4", "SLG/2018/page_45.pdf-1"])
def test_inconsistent_tasks_are_not_served(task_id: str):
    assert task_id not in {t["id"] for t in FinQA.list_tasks("train")}
    with pytest.raises(ValueError):
        FinQA(task_spec={"id": task_id, "split": "train"}, secrets={"api_key": ""})


def test_served_tasks_have_consistent_numeric_references():
    counts = {split: len(FinQA.list_tasks(split)) for split in FinQA.list_splits()}
    assert counts == {"train": 5672, "dev": 815, "test": 1047}
    for split in FinQA.list_splits():
        for task in FINQA_DATA[split]:
            assert isinstance(task["qa"]["exe_ans"], float)
            assert has_consistent_reference(task["qa"])
