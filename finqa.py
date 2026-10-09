import json
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from cli_environment import CLIEnvironment
from openreward.environments import JSONObject, ToolOutput, TextBlock, tool
from openreward import AsyncOpenReward, SandboxSettings

from constants import ENV_PATH


# ====================
# Data Loading
# ====================

# Largest relative error accepted between an answer and the reference. The window
# does not depend on how many digits the answer gives, so rounding more coarsely
# never widens it; any answer with 3 significant figures falls inside it.
MAX_RELATIVE_ERROR = 0.01

_WRITTEN_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")

SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6, "mm": 1e6, "mn": 1e6,
    "billion": 1e9, "b": 1e9, "bn": 1e9,
}

# A number followed by a scale word, e.g. "95 million", "1.2bn", "450K".
_SCALED_NUMBER = re.compile(
    r"(?P<num>-?(?:\d+\.?\d*|\.\d+))\s*(?P<word>(?:thousand|million|billion)s?|k|mm|mn|m|bn|b)",
    re.IGNORECASE,
)

# The unit a document states its amounts in, e.g. "( in millions )", "$ in thousands".
_STATED_UNIT = re.compile(r"\bin (thousand|million|billion)s\b", re.IGNORECASE)


def has_consistent_reference(qa: dict) -> bool:
    """
    Whether a task's graded reference (exe_ans) is a number that agrees with the
    dataset's written answer.

    exe_ans is the result of the task's annotated program. When that program is
    wrong (wrong operands, reversed division, wrong units or sign), exe_ans
    disagrees with the written answer and the task cannot be graded reliably,
    so it is not served. The two agree when the written answer is within
    MAX_RELATIVE_ERROR of exe_ans, or is exe_ans rounded or truncated to the
    written answer's last digit, after percent scaling either way.

    Yes/no questions (a non-numeric exe_ans) are not served either: this env
    asks for a numerical answer, and a binary answer can be guessed.

    A task without a written number is kept: there is nothing to check.
    """
    exe = qa["exe_ans"]
    if isinstance(exe, str):
        return False
    match = _WRITTEN_NUMBER.search(str(qa.get("answer", "")).replace("$", ""))
    if match is None:
        return True
    text = match.group(0).replace(",", "")
    written = float(text)
    # One unit of the written answer's last digit, with slack for float error.
    unit = 10.0 ** -len(text.partition(".")[2]) * (1 + 1e-9)
    for target in (exe, exe * 100, exe / 100):
        error = abs(written - target)
        rounded = error <= unit / 2
        truncated = (written >= 0) == (target >= 0) and 0 <= abs(target) - abs(written) < unit
        if rounded or truncated or (target != 0 and error / abs(target) <= MAX_RELATIVE_ERROR):
            return True
    return False


def load_finqa_data():
    """Load all FinQA splits at module import time, keeping the tasks with a consistent reference."""
    data_dir = ENV_PATH / "data"

    with open(data_dir / "train.json", "r") as f:
        train_tasks = [t for t in json.load(f) if has_consistent_reference(t["qa"])]

    with open(data_dir / "dev.json", "r") as f:
        dev_tasks = [t for t in json.load(f) if has_consistent_reference(t["qa"])]

    with open(data_dir / "test.json", "r") as f:
        test_tasks = [t for t in json.load(f) if has_consistent_reference(t["qa"])]

    return {
        "train": train_tasks,
        "dev": dev_tasks,
        "test": test_tasks
    }

FINQA_DATA = load_finqa_data()


# ====================
# Pydantic Schemas
# ====================

# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class FinQATaskSpec(BaseModel):
    id: str
    split: str

class SubmitAnswerInput(BaseModel):
    answer: str = Field(..., description="Your final numerical answer. Can be a number, percentage, or mathematical expression.")


# ====================
# Helper Functions
# ====================

def format_table_markdown(table: List[List[str]]) -> str:
    """
    Convert nested list table to markdown format with header row.

    Args:
        table: List of rows, where each row is a list of cell values

    Returns:
        Markdown-formatted table string
    """
    if not table:
        return ""

    lines = []

    # First row as header
    header = table[0]
    lines.append("| " + " | ".join(str(cell) for cell in header) + " |")

    # Separator row
    lines.append("|" + "|".join("---" for _ in header) + "|")

    # Data rows
    for row in table[1:]:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(lines)


def stated_units(task_data: dict) -> tuple[float, ...]:
    """Scales ("in thousands/millions/billions") the task's text, table or question states amounts in."""
    parts = [*task_data.get("pre_text", []), *task_data.get("post_text", []), task_data["qa"]["question"]]
    parts += [" ".join(str(cell) for cell in row) for row in task_data.get("table", [])]
    words = {m.group(1).lower() for m in _STATED_UNIT.finditer(" ".join(parts))}
    return tuple(sorted(SCALE_WORDS[w] for w in words))


def validate_numerical_answer(submitted: str, expected: str, units: tuple[float, ...] = ()) -> bool:
    """
    Validate numerical answers with flexible comparison.

    The reference is FinQA's executed program result (exe_ans), which gives
    percentages as fractions (0.11852) or occasionally already scaled (11.852),
    while answers are written as "11.85%", "11.85" or "0.1185". So the answer
    value (with any "%" sign dropped) is compared against both the reference
    and 100x the reference.

    An answer matches a target when it is within MAX_RELATIVE_ERROR of it
    ("2.60", "2.6" and "2.61" match 2.60192; "3.5%" for 3.537% and "7%" for
    6.58% do not). A zero reference needs an exact zero.

    Thousand separators and "$" are ignored. A non-numeric reference
    ("yes"/"no") is compared as a case-insensitive string.

    An answer with a scale word ("95 million", "$1.2bn") is read as the full
    amount and matches when it equals the reference taken as a full amount or
    in one of `units`, the scales the task's document states its amounts in.
    So "95 million" and "0.095 billion" match a reference of 95 in a document
    "in millions", and "95 billion" or "95 thousand" do not.

    Args:
        submitted: Agent's submitted answer
        expected: Ground truth answer
        units: Scales the task's document states amounts in (see stated_units)

    Returns:
        True if the answer matches the reference
    """

    def parse_number(s: str) -> float:
        s = s.strip().replace(',', '').replace('$', '').strip()
        if s.endswith('%'):
            s = s[:-1].strip()
        return float(s)

    def within_tolerance(value: float, targets) -> bool:
        for target in targets:
            if target == 0:
                if value == 0:
                    return True
                continue
            if abs(value - target) / abs(target) <= MAX_RELATIVE_ERROR:
                return True
        return False

    try:
        expected_num = parse_number(expected)
    except ValueError:
        # Fallback to string comparison for non-numeric answers
        return submitted.strip().lower() == expected.strip().lower()

    try:
        submitted_num = parse_number(submitted)
    except ValueError:
        scaled = _SCALED_NUMBER.fullmatch(submitted.strip().replace(',', '').replace('$', '').strip())
        if scaled is None:
            return False
        amount = float(scaled.group("num")) * SCALE_WORDS[scaled.group("word").lower().removesuffix("s")]
        return within_tolerance(amount, [expected_num] + [expected_num * unit for unit in units])

    return within_tolerance(submitted_num, (expected_num, expected_num * 100))


def format_full_prompt(task_data: dict) -> str:
    """
    Generate complete prompt with formatted financial data.

    Args:
        task_data: Task dictionary with pre_text, table, post_text, qa

    Returns:
        Formatted prompt string
    """
    sections = []

    # Pre-table text
    if task_data.get("pre_text"):
        sections.append("## Pre-Table Context")
        sections.append("\n\n".join(task_data["pre_text"]))
        sections.append("")

    # Table
    if task_data.get("table"):
        sections.append("## Financial Data Table")
        sections.append(format_table_markdown(task_data["table"]))
        sections.append("")

    # Post-table text
    if task_data.get("post_text"):
        sections.append("## Post-Table Context")
        sections.append("\n\n".join(task_data["post_text"]))
        sections.append("")

    # Question
    sections.append("## Question")
    sections.append(task_data["qa"]["question"])
    sections.append("")

    # Instructions
    sections.append("## Instructions")
    sections.append("You have access to CLI tools (bash, read, write, grep, etc.) to perform calculations and analyze the data.")
    sections.append("Use these tools to compute your answer step by step.")
    sections.append("When ready, submit your final numerical answer using the submit_answer tool.")
    sections.append("")
    sections.append("**Important**: Your answer should be a single numerical value (e.g., '1.61' or '15.3%').")
    sections.append("Give at least 3 significant figures: an answer is accepted only if it is within 1% of the correct value, so a coarsely rounded answer (e.g. '7%' for 6.58%) is marked incorrect.")

    return "\n".join(sections)


# ====================
# Main Environment Class
# ====================

class FinQA(CLIEnvironment):
    """
    FinQA: Financial Question Answering with reasoning over tables and text.

    Agents must read financial documents, perform numerical calculations,
    and submit accurate numerical answers.

    Reference: Chen et al., "FinQA: A Dataset of Numerical Reasoning over Financial Data", EMNLP 2021
    """

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available dataset splits."""
        return ["train", "dev", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """
        Return list of tasks for a given split.

        Args:
            split: One of 'train', 'dev', 'test'

        Returns:
            List of task specifications (dicts with id and split)
        """
        if split not in FINQA_DATA:
            return []
        return [
            {"id": task["id"], "split": split}
            for task in FINQA_DATA[split]
        ]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize FinQA environment for a specific task.

        Args:
            task_spec: Task specification with id and split
            secrets: API keys and secrets (requires 'api_key' for sandbox)
        """
        super().__init__(task_spec, secrets=secrets)

        # Validate and load task data
        spec = FinQATaskSpec.model_validate(task_spec)
        self.task_id = spec.id
        self.split = spec.split

        # Graded submissions this session. Only the first is scored, so the task
        # cannot be re-graded for a second payout.
        self.submitted = 0

        # Find task in data
        self.task_data = next(
            (task for task in FINQA_DATA[self.split] if task["id"] == self.task_id),
            None
        )

        if self.task_data is None:
            raise ValueError(f"Task with id '{self.task_id}' not found in split '{self.split}'")

        # Configure sandbox (workspace only, no data mount)
        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/FinQA",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="0.5:0.5",
            block_network=False,
        )

        # Initialize sandbox
        or_client = AsyncOpenReward(api_key=secrets.get("api_key", ""))
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def get_prompt(self) -> list[TextBlock]:
        """
        Generate prompt with formatted financial data.

        Returns:
            List containing single TextBlock with formatted prompt
        """
        prompt_text = format_full_prompt(self.task_data)
        return [TextBlock(type="text", text=prompt_text)]

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your final numerical answer for evaluation.

        This tool validates your answer against the ground truth and ends the episode.

        Args:
            params: SubmitAnswerInput with 'answer' field

        Returns:
            ToolOutput with correctness feedback, reward, and finished=True
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        # Extract ground truth
        ground_truth = str(self.task_data["qa"]["exe_ans"])
        submitted = params.answer.strip()

        # Validate answer
        is_correct = validate_numerical_answer(submitted, ground_truth, stated_units(self.task_data))
        reward = 1.0 if is_correct else 0.0

        # Generate feedback message
        if is_correct:
            message = f"✅ Correct! Your answer '{submitted}' matches the expected answer."
        else:
            message = f"❌ Incorrect. Your answer: '{submitted}'."

        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(type="text", text=message)],
            metadata={
                "task_id": self.task_id,
                "submitted_answer": submitted,
                "correct": is_correct,
            },
            reward=reward,
            finished=True,
        )
