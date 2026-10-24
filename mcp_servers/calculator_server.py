"""Safe arithmetic calculator as a standalone MCP server (stdio)."""
from mcp.server.fastmcp import FastMCP

from orchestrator.tools import calculator as _calculator

mcp = FastMCP("calculator", log_level="WARNING")


@mcp.tool(name=_calculator.name, description=_calculator.description)
def calculator(expression: str) -> str:
    return _calculator.invoke(expression)


if __name__ == "__main__":
    mcp.run(transport="stdio")
