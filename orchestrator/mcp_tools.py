"""Load the Tools Agent's tools from MCP servers instead of building them in-process.

Each tool runs as its own MCP server (see mcp_servers/), spawned as a subprocess
over stdio. The adapter discovers them with tools/list and hands them back as
ordinary LangChain tools, so bind_tools and ToolNode use them unchanged.

The adapter's tools are async-only, so the tools graph has to be driven with
ainvoke (see graphs.py and app.py).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from orchestrator.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# server name -> module under mcp_servers/
MCP_SERVERS = {
    "web_search": "mcp_servers.web_search_server",
    "wikipedia": "mcp_servers.wikipedia_server",
    "arxiv": "mcp_servers.arxiv_server",
    "calculator": "mcp_servers.calculator_server",
}


def mcp_connections(settings: Settings) -> dict:
    # stdio_client forwards only a short allowlist of env vars by default; the
    # servers also need WIKI_*, SSL_CERT_FILE and any proxy settings.
    env = {**os.environ, "WIKI_DOC_CHARS": str(settings.wiki_doc_content_chars_max)}
    return {
        name: {
            "transport": "stdio",
            # Same interpreter as the app, so the servers see the same packages.
            "command": sys.executable,
            "args": ["-m", module],
            "cwd": str(PROJECT_ROOT),
            "env": env,
        }
        for name, module in MCP_SERVERS.items()
    }


def _as_text(tool):
    """Collapse the tool's MCP content blocks into one plain string.

    The adapter returns a list of content blocks that each carry an `id`, and
    langchain-groq forwards list content to the API verbatim. A plain string is
    what the in-process tools return, so the LLM sees the same thing either way.
    """
    call = tool.coroutine

    async def call_as_text(**kwargs):
        content, artifact = await call(**kwargs)
        if isinstance(content, list):
            content = "\n".join(
                b["text"] if isinstance(b, dict) else str(b)
                for b in content
                if not isinstance(b, dict) or b.get("type") == "text"
            )
        return content, artifact

    tool.coroutine = call_as_text
    return tool


def load_mcp_tools(settings: Settings) -> list:
    """Start every server once to list its tools; returns LangChain tools."""
    client = MultiServerMCPClient(mcp_connections(settings))
    return [_as_text(t) for t in asyncio.run(client.get_tools())]
