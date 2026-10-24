"""arXiv paper search as a standalone MCP server (stdio)."""
from mcp.server.fastmcp import FastMCP

from orchestrator.tools import _make_arxiv_tool

_arxiv = _make_arxiv_tool(arxiv_k=3)

mcp = FastMCP("arxiv", log_level="WARNING")


@mcp.tool(name=_arxiv.name, description=_arxiv.description)
def arxiv(query: str) -> str:
    return _arxiv.invoke(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
