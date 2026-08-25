from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# --- Calculator tool (safe arithmetic) ---
import ast
import operator as op

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}

def _eval_expr(expr: str) -> float:
    """Safely evaluate a basic arithmetic expression."""
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Num):  # py<3.8
            return n.n
        if isinstance(n, ast.Constant):  # py>=3.8
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numbers are allowed.")
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        raise ValueError("Only basic arithmetic is allowed.")

    return float(_eval(node))

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input must be a plain arithmetic expression (e.g., '12*(3+4)')."""
    try:
        return str(_eval_expr(expression))
    except Exception as e:
        return f"[calculator error] {e}"

# --- Web/Wiki/Arxiv tools ---

# Wikimedia blocks the shared default User-Agent the `wikipedia` package ships with
# and answers 403/429 with a text/plain body. That package calls r.json() without
# checking the status, so the failure surfaces as a bare
# "Expecting value: line 1 column 1 (char 0)". Send our own descriptive agent per
# https://meta.wikimedia.org/wiki/User-Agent_policy instead.
DEFAULT_WIKI_USER_AGENT = (
    "LangGraph-Multi-Agent-Orchestrator/0.1 "
    "(https://github.com/sparklerz/LangGraph-Multi-Agent-Orchestrator)"
)

def _configure_wikipedia_client() -> None:
    """Give the `wikipedia` package a compliant User-Agent and polite pacing."""
    try:
        import wikipedia
    except ImportError:
        return
    wikipedia.set_user_agent(os.getenv("WIKI_USER_AGENT", DEFAULT_WIKI_USER_AGENT))
    wikipedia.set_rate_limiting(True)

def _make_wikipedia_tool(*, wiki_k: int, wiki_chars: int):
    """Wikipedia tool that reports API rejections instead of leaking a JSON error."""
    _configure_wikipedia_client()

    # IMPORTANT: WikipediaQueryRun requires api_wrapper in your installed versions.
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=wiki_k, doc_content_chars_max=wiki_chars)
    runner = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    @tool("wikipedia")
    def wikipedia_search(query: str) -> str:
        """Look up an article on Wikipedia and return its summary.

        Pass a short topic or entity name -- the article title you expect, e.g.
        "Attention Is All You Need" or "Alan Turing". Do NOT pass the user's full
        question: Wikipedia matches loose keywords, so a long sentence returns
        unrelated articles.
        """
        try:
            return runner.run(query)
        except json.JSONDecodeError:
            return (
                "[wikipedia unavailable] Wikipedia's API returned a non-JSON response "
                "(usually a 403/429 for the User-Agent or rate limit). Set WIKI_USER_AGENT "
                "to a descriptive string with contact info and retry."
            )
        except Exception as e:  # noqa: BLE001 - surface a readable message to the agent
            return f"[wikipedia error] {type(e).__name__}: {e}"

    return wikipedia_search

def _make_arxiv_tool(*, arxiv_k: int = 3, doc_chars: int = 4000):
    """arXiv tool driven through arxiv.Client rather than langchain's wrapper.

    langchain_community still calls Search(...).results(), which arxiv 4.0
    removed -- it surfaces as "'Search' object has no attribute 'results'".
    Client.results() has existed since arxiv 2.0, so this works on every version
    requirements.txt allows. Output format matches ArxivQueryRun, so nothing
    downstream has to change.
    """
    import arxiv

    client = arxiv.Client()

    @tool("arxiv")
    def arxiv_search(query: str) -> str:
        """Search arXiv for research papers; returns title, authors, and abstract.

        Pass a paper title or a few topic keywords, not the user's full question.
        arXiv supports field prefixes -- when you know the exact title use
        ti:"Attention Is All You Need", because bare keywords rank the real paper
        far down. au:"Author Name" filters by author.
        Use this when the question is about a paper, its authors, or affiliations.
        """
        try:
            search = arxiv.Search(query=query[:300], max_results=arxiv_k)
            docs = [
                f"Published: {r.updated.date()}\n"
                f"Title: {r.title}\n"
                f"Authors: {', '.join(a.name for a in r.authors)}\n"
                f"Summary: {r.summary}"
                for r in client.results(search)
            ]
        except Exception as e:  # noqa: BLE001 - surface a readable message to the agent
            return f"[arxiv error] {type(e).__name__}: {e}"

        if not docs:
            return "No good Arxiv Result was found"
        return "\n\n".join(docs)[:doc_chars]

    return arxiv_search

def make_web_wiki_arxiv_tools(*, wiki_k: int = 3, wiki_chars: int = 2000, arxiv_k: int = 3):
    """Return tool objects compatible with LangGraph ToolNode."""

    web = DuckDuckGoSearchRun()

    wiki = _make_wikipedia_tool(wiki_k=wiki_k, wiki_chars=wiki_chars)

    # Its underlying API doesn't require keys.
    arxiv = _make_arxiv_tool(arxiv_k=arxiv_k)

    return [web, wiki, arxiv, calculator]

# @dataclass
# class ToolResult:
#     tool: str
#     output: str

@dataclass
class ToolResult:
    tool: str
    output: str
    ok: bool = True
    error: Optional[str] = None

def run_tools_once(query: str, *, wiki_k: int = 3, wiki_chars: int = 2000) -> List[ToolResult]:
    """Non-agent helper: run each tool once and return outputs (good for debugging)."""
    tools = make_web_wiki_arxiv_tools(wiki_k=wiki_k, wiki_chars=wiki_chars)
    out: List[ToolResult] = []
    for t in tools:
        try:
            out.append(ToolResult(tool=t.name, output=str(t.run(query))))
        except Exception as e:
            out.append(ToolResult(tool=t.name, output=f"[tool error] {e}", ok=False, error=str(e)))

    return out
