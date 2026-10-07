"""
Grading tests for FinQA. Needs the dataset JSON files in ./data (train.json,
dev.json, test.json); no sandbox or API key is used.
"""

import asyncio

import pytest

from finqa import FINQA_DATA, FinQA, SubmitAnswerInput, validate_numerical_answer


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
    ("LKQ/2009/page_66.pdf-2", "11.95%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "12.85%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "-11.85%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "0.12", 0.0),
    ("LKQ/2009/page_66.pdf-2", "12%", 0.0),
    ("LKQ/2009/page_66.pdf-2", "0", 0.0),
    ("LKQ/2016/page_48.pdf-4", "2.60", 1.0),
    ("LKQ/2016/page_48.pdf-4", "2.6", 1.0),
    ("LKQ/2016/page_48.pdf-4", "2.6019", 1.0),
    ("LKQ/2016/page_48.pdf-4", "2.61", 0.0),
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
    ("0.0", "0.0", True),
    ("0.1", "0.0", False),
    ("yes", "yes", True),
    ("No", "no", True),
    ("yes", "no", False),
])
def test_validate_numerical_answer(submitted: str, expected: str, ok: bool):
    assert validate_numerical_answer(submitted, expected) is ok
