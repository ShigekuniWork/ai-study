import argparse
import asyncio
from typing import List, cast  # ←追加

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools.types import BaseTool
from llama_index.llms.ollama import Ollama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def run_mcp_task(prompt: str):
    try:
        # MCPクライアント → ツールリスト取得
        client = BasicMCPClient(command_or_url="http://localhost:8000/mcp", timeout=300)
        tools_spec = McpToolSpec(client=client, allowed_tools=["echo"])

        tools_raw = await tools_spec.to_tool_list_async()

        tools = cast(List[BaseTool], tools_raw)

        llm = Ollama(model="llama3:instruct", request_timeout=60)

        # ReActAgentの初期化（新しいworkflow版）
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
        )

        workflow_result = await agent.run(prompt)
        response = workflow_result
        print(">>", response)
    except ValueError as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(run_mcp_task(args.prompt))


if __name__ == "__main__":
    main()
