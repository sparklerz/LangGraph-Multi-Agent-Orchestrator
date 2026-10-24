"""DuckDuckGo web search as a standalone MCP server (stdio)."""
from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

_web = DuckDuckGoSearchRun()

mcp = FastMCP("web_search", log_level="WARNING")


@mcp.tool(name=_web.name, description=_web.description)
def duckduckgo_search(query: str) -> str:
    try:
        return _web.invoke(query)
    except Exception as e:  # noqa: BLE001 - surface a readable message to the agent
        return f"[web search error] {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
