from fastapi import FastAPI
from fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("テスト")


@mcp.tool(
    name="echo",
    description="Echo back the input string.",
)
def echo_tool(input: str, **kwargs) -> str:
    return input


if __name__ == "__main__":
    mcp.run(
        transport="http",        
    )
