"""Each Tools Agent tool, packaged as its own standalone MCP server.

Every module here is runnable on its own over stdio, e.g.

    python -m mcp_servers.wikipedia_server

The orchestrator spawns them through orchestrator/mcp_tools.py, but any MCP
client (Claude Desktop, Cursor, MCP Inspector) can launch them the same way.
"""
