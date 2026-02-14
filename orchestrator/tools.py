from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

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
def make_web_wiki_arxiv_tools(*, wiki_k: int = 3, wiki_chars: int = 2000):
    """Return tool objects compatible with LangGraph ToolNode."""

    web = DuckDuckGoSearchRun()

    # IMPORTANT: WikipediaQueryRun requires api_wrapper in your installed versions.
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=wiki_k, doc_content_chars_max=wiki_chars)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # ArxivQueryRun works similarly; its underlying API doesn't require keys.
    arxiv = ArxivQueryRun()

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
