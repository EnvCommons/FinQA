import json
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

def load_finqa_data():
    """Load all FinQA splits at module import time."""
    data_dir = ENV_PATH / "data"

    with open(data_dir / "train.json", "r") as f:
        train_tasks = json.load(f)

    with open(data_dir / "dev.json", "r") as f:
        dev_tasks = json.load(f)

    with open(data_dir / "test.json", "r") as f:
        test_tasks = json.load(f)

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


def validate_numerical_answer(submitted: str, expected: str, tolerance: float = 1e-4) -> bool:
    """
    Validate numerical answers with flexible comparison.

    Handles:
    - Percentages (15.3% vs 15.3)
    - Decimal variations (1000 vs 1000.0)
    - Thousand separators (1,000 vs 1000)
    - Scientific notation
    - Negative numbers

    Args:
        submitted: Agent's submitted answer
        expected: Ground truth answer
        tolerance: Relative tolerance for floating point comparison (default: 0.01%)

    Returns:
        True if answers match within tolerance
    """

    def parse_number(s: str) -> float:
        """Parse string to number, handling percentages and edge cases."""
        s = s.strip()

        # Handle percentage
        is_percentage = s.endswith('%')
        if is_percentage:
            s = s[:-1].strip()

        # Remove common formatting
        s = s.replace(',', '')  # Remove thousand separators

        # Parse to float
        try:
            num = float(s)
            # If original had %, keep as-is (don't divide by 100)
            # This allows comparing "15.3%" with "15.3"
            return num
        except ValueError:
            raise ValueError(f"Cannot parse '{s}' as a number")

    try:
        submitted_num = parse_number(submitted)
        expected_num = parse_number(expected)

        # Absolute difference comparison
        diff = abs(submitted_num - expected_num)

        # For very small numbers, use absolute tolerance
        # For larger numbers, use relative tolerance
        if abs(expected_num) < 1e-6:
            return diff < tolerance
        else:
            relative_diff = diff / abs(expected_num)
            return relative_diff < tolerance

    except (ValueError, ZeroDivisionError):
        # Fallback to string comparison for non-numeric answers
        return submitted.strip().lower() == expected.strip().lower()


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
        is_correct = validate_numerical_answer(submitted, ground_truth)
        reward = 1.0 if is_correct else 0.0

        # Generate feedback message
        if is_correct:
            message = f"✅ Correct! Your answer '{submitted}' matches the expected answer '{ground_truth}'."
        else:
            message = f"❌ Incorrect. Your answer: '{submitted}'. Expected: '{ground_truth}'."

        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(type="text", text=message)],
            metadata={
                "task_id": self.task_id,
                "submitted_answer": submitted,
                "expected_answer": ground_truth,
                "correct": is_correct,
            },
            reward=reward,
            finished=True,
        )
