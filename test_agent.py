import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")


async def main() -> None:
    """
    Test the FinQA environment with an OpenAI agent using the Responses API.
    """

    # Initialize clients
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Get environment and tasks
    environment = or_client.environments.get(name="local/finqa", base_url=BASE_URL)
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in dev split")
    print(f"Available tools: {len(tools)}")
    print()

    # Test first task
    task = tasks[50]
    print(task)
    print("=" * 80)

    async with environment.session(task=task) as session:
        # Get initial prompt
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text if isinstance(prompt, list) else prompt

        print("\n=== PROMPT ===")
        print(prompt_text)
        print("\n=== AGENT INTERACTION ===\n")

        # Initialize conversation
        input_list = [{"role": "user", "content": prompt_text}]
        finished = False
        turn_count = 0
        max_turns = 20  # Safety limit

        while not finished and turn_count < max_turns:
            turn_count += 1
            print(f"--- Turn {turn_count} ---")

            # Get model response
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            # Add response to conversation
            input_list += response.output

            # Process model output
            for item in response.output:
                if item.type == "text":
                    print(f"Agent: {item.text}")

                elif item.type == "function_call":
                    print(f"\nTool call: {item.name}")
                    print(f"Arguments: {item.arguments}")

                    # Call tool via environment
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments))
                    )

                    # Display result
                    result_text = tool_result.blocks[0].text if tool_result.blocks else ""
                    print(f"Result: {result_text}")
                    print(f"Reward: {tool_result.reward:.3f}")

                    # Add tool output to conversation
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_text
                    })

                    # Check if episode finished
                    if tool_result.finished:
                        finished = True
                        print("\n=== EVALUATION COMPLETE ===")
                        print(f"Final reward: {tool_result.reward:.3f}")
                        print(f"Correct: {tool_result.metadata.get('correct', False)}")
                        break

            print()

        if turn_count >= max_turns and not finished:
            print(f"WARNING: Reached maximum turns ({max_turns}) without finishing")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Usage: export OPENAI_API_KEY='your-key' && python test_agent.py")
        exit(1)

    asyncio.run(main())
