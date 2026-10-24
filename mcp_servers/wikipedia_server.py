"""Wikipedia lookup as a standalone MCP server (stdio)."""
import os

from mcp.server.fastmcp import FastMCP

from orchestrator.tools import _make_wikipedia_tool

# Reuses the in-process tool, so the User-Agent fix and error handling apply here too.
_wiki = _make_wikipedia_tool(wiki_k=3, wiki_chars=int(os.getenv("WIKI_DOC_CHARS", "2000")))

mcp = FastMCP("wikipedia", log_level="WARNING")


@mcp.tool(name=_wiki.name, description=_wiki.description)
def wikipedia(query: str) -> str:
    return _wiki.invoke(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
