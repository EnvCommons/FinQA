# FinQA - Financial Question Answering Environment

OpenReward environment for the FinQA dataset (EMNLP 2021).

## Overview

FinQA is a financial reasoning environment that requires agents to read financial documents (tables + text), perform numerical calculations, and submit accurate answers.

**Paper**: [Chen et al., "FinQA: A Dataset of Numerical Reasoning over Financial Data", EMNLP 2021](https://github.com/czyssrs/FinQA)

## Task Description

Agents must:
1. Read pre-table text, financial tables, and post-table text
2. Understand the question about the financial data
3. Use CLI tools (bash, read, write, etc.) to perform calculations
4. Submit a final numerical answer

## Environment Details

- **Type**: Single-turn evaluation with CLIEnvironment
- **Splits**:
  - `train`: 6,250+ tasks
  - `dev`: 883 tasks
  - `test`: 1,147 tasks
- **Evaluation**: Execution accuracy (numerical answer matching)
- **Tools**:
  - 12 CLI tools (bash, glob, grep, ls, read, write, edit, multi_edit, todo_write)
  - 1 custom tool: `submit_answer`

## Example Task

**Prompt:**
```markdown
## Financial Data Table
| Year | Revenue | Expenses |
|------|---------|----------|
| 2020 | 5829    | 4500     |
| 2021 | 5735    | 4200     |

## Question
What is the percentage decrease in revenue from 2020 to 2021?

## Instructions
You have access to CLI tools (bash, read, write, grep, etc.) to perform calculations.
Use these tools to analyze the data and compute your answer.
When ready, submit your final numerical answer using the submit_answer tool.
```

**Expected Agent Behavior:**
1. Write Python script: `(5829 - 5735) / 5829 * 100`
2. Execute: `bash: python calculate.py`
3. Submit: `submit_answer: {"answer": "1.61"}`
4. Result: ✅ Correct! (reward=1.0)

## Usage

### Prerequisites

- Python 3.11+
- OpenReward SDK
- OpenAI API key (for testing)

### Local Development

1. **Start the server:**
   ```bash
   python server.py
   ```

   Server will start on `http://0.0.0.0:8080`

2. **Test with an agent:**
   ```bash
   export OPENAI_API_KEY="your-key"
   python test_agent.py
   ```

### Docker

1. **Build the image:**
   ```bash
   docker build -t finqa:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 finqa:latest
   ```

## Architecture

### Files

- `finqa.py` - Main environment class extending CLIEnvironment
- `cli_environment.py` - Base CLIEnvironment with 12 built-in tools
- `utils.py` - Sandbox utilities for file upload/download
- `server.py` - Minimal server wrapper
- `test_agent.py` - OpenAI Responses API test client
- `data/` - Dataset files (train.json, dev.json, test.json)

### Key Features

1. **Table Formatting**: Financial tables converted to markdown for readability
2. **Numerical Validation**: Flexible answer comparison handling percentages, decimals, separators
3. **CLI Tools**: Agents can write scripts, execute calculations, iterate on solutions
4. **Tolerance**: 1e-4 (0.01%) relative tolerance for floating-point accuracy

## Answer Validation

The environment validates numerical answers with flexible comparison:

- ✅ Handles percentages: `"15.3%"` == `"15.3"`
- ✅ Handles decimals: `"1000"` == `"1000.0"`
- ✅ Handles separators: `"1,000"` == `"1000"`
- ✅ Uses relative tolerance (0.01%) for large numbers
- ✅ Fallback to string comparison for non-numeric values

## Dataset

- **Source**: [FinQA GitHub Repository](https://github.com/czyssrs/FinQA)
- **Size**: ~8.5K tasks across train/dev/test splits
- **Format**: JSON files with financial documents and questions

Each task contains:
```json
{
  "id": "unique_identifier",
  "pre_text": ["text before table"],
  "table": [["Header1", "Header2"], ["Value1", "Value2"]],
  "post_text": ["text after table"],
  "qa": {
    "question": "What is the percentage change?",
    "exe_ans": "25.5"
  }
}
```

## License

MIT License (consistent with original FinQA dataset)

## Citation

If you use this environment, please cite the original FinQA paper:

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021}
}
```

## Contributing

This environment is part of the EnvCommons collection. For issues or improvements, please open a GitHub issue.

## Contact

For questions about this environment, please open an issue on the [EnvCommons/finqa](https://github.com/EnvCommons/finqa) repository.
