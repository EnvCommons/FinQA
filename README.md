# FinQA

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/FinQA)

## Description

FinQA is an environment for evaluating numerical reasoning over financial data. Agents must analyze financial documents containing tables and contextual text, perform multi-step calculations, and produce accurate numerical answers. Tasks are derived from real earnings reports and SEC filings requiring domain-specific financial reasoning.

## Capabilities

- Numerical reasoning over financial tables
- Multi-step calculation and computation
- Understanding financial document context
- Extracting and combining values from tables and text

## Compute Requirements

Agents are given a sandboxed environment with 0.5 CPU and 0.5 GB RAM, with access to CLI tools for computation.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are three splits in this environment:

- **train**: 6,251 tasks
- **dev**: 883 tasks
- **test**: 1,147 tasks

Each task presents a financial document with pre-table context, a data table, post-table context, and a numerical question requiring calculation.

## Reward Structure

This is a multi-turn environment. Agents can use CLI tools (bash, read, write, grep, etc.) to analyze data and perform calculations. The agent submits a final numerical answer via the `submit_answer` tool. Validation is numerical: the answer must agree with the reference (FinQA's executed program result) at the answer's own number of decimal places and be within 1% of it. Percentages may be given as "11.85%", "11.85" or "0.1185". Reward is binary: 1.0 if correct, 0.0 if incorrect.

## Data

Data consists of JSON files (`train.json`, `dev.json`, `test.json`) containing financial documents with tables, context text, and QA pairs. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands for computation |
| `read` | Read file contents |
| `write` | Write content to files |
| `grep` | Search for patterns in files |
| `submit_answer` | Submit your final numerical answer. Ends the episode. |

## Time Horizon

Multi-turn. Agents can perform multiple computation steps before submitting a final answer.

## Environment Difficulty

FinQA evaluates financial numerical reasoning capabilities requiring table understanding and multi-step calculation.

## Other Environment Requirements

There are no further environment requirements; FinQA works out of the box with the OpenReward endpoint without any external API keys.

## Safety

Agents in FinQA perform financial calculations in a sandboxed environment. The environment does not present direct safety risks.

## Citation

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={7672--7685},
  year={2021}
}
```
