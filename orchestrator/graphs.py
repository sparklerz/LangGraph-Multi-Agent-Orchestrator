from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from orchestrator.factories import get_llm
from orchestrator.graph_agent import graph_answer
from orchestrator.settings import Settings
from orchestrator.sql_agent import sql_answer
from orchestrator.tools import make_web_wiki_arxiv_tools

Route = Literal["sql", "graph", "tools", "general"]


class RouterState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Route
    debug: Dict[str, Any]


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def _last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _safe_text(m.content).strip()
    return ""


def _messages_to_transcript(messages: list[BaseMessage], max_turns: int = 8) -> str:
    """
    Build a lightweight transcript from the last N Human/AI messages.
    We intentionally skip tool messages to keep prompts stable.
    """
    kept: List[BaseMessage] = []
    for m in reversed(messages):
        if isinstance(m, (HumanMessage, AIMessage)):
            kept.append(m)
        if len(kept) >= max_turns * 2:  # ~turns * 2 messages
            break
    kept.reverse()

    lines: List[str] = []
    for m in kept:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {_safe_text(m.content)}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {_safe_text(m.content)}")
    return "\n".join(lines).strip()


def _merge_debug(state: RouterState, **kv: Any) -> Dict[str, Any]:
    dbg = dict(state.get("debug") or {})
    for k, v in kv.items():
        if v is not None:
            dbg[k] = v
    return dbg


def _extract_tool_names(messages: list[BaseMessage]) -> List[str]:
    """
    Extract tool names from AIMessage.tool_calls across LangChain variants.
    """
    names: List[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            for tc in tool_calls:
                # tc may be dict-like or object-like
                if isinstance(tc, dict):
                    n = tc.get("name")
                else:
                    n = getattr(tc, "name", None)
                if n:
                    names.append(str(n))
    # de-dupe, preserve order
    out: List[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _rewrite_to_standalone(llm, messages: list[BaseMessage]) -> str:
    """
    If the user asks a follow-up like "show them", rewrite into a standalone question.
    """
    question = _last_user_text(messages)
    if not question:
        return ""

    # If there's only one user message total, no rewrite needed.
    num_user_msgs = sum(1 for m in messages if isinstance(m, HumanMessage))
    if num_user_msgs <= 1:
        return question

    transcript = _messages_to_transcript(messages, max_turns=8)
    prompt = (
        "Rewrite the user's latest question into a standalone question.\n"
        "Do NOT answer the question.\n\n"
        "Conversation:\n"
        f"{transcript}\n\n"
        "Latest user question:\n"
        f"{question}\n\n"
        "Standalone question:"
    )
    msg = llm.invoke(
        [
            SystemMessage(content="You rewrite follow-up questions into standalone questions."),
            HumanMessage(content=prompt),
        ]
    )
    rewritten = _safe_text(getattr(msg, "content", "")).strip()
    return rewritten or question


def build_tools_agent_graph(settings: Settings):
    tools = make_web_wiki_arxiv_tools(
        wiki_chars=settings.wiki_doc_content_chars_max,
    )
    llm = get_llm(settings, temperature=0).bind_tools(tools)

    def assistant(state: RouterState):
        msg = llm.invoke(state["messages"])
        return {"messages": [msg]}

    g = StateGraph(RouterState)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", tools_condition)
    g.add_edge("tools", "assistant")
    return g.compile()


def build_router_graph(settings: Settings):
    tools_graph = build_tools_agent_graph(settings)
    llm_router = get_llm(settings, temperature=0)

    route_prompt = (
        "You are a router for a multi-agent system.\n"
        "Choose exactly ONE route label from: sql, graph, tools, general.\n\n"
        "Routing rules:\n"
        "- sql: querying a relational database (tables/rows, SQL, students DB, counts, filters).\n"
        "- graph: querying a Neo4j graph database (nodes/relationships, Cypher).\n"
        "- tools: needs external knowledge / searching (Wikipedia/arXiv/web) or tool use.\n"
        "- general: conceptual explanation or chat that doesn't need tools/DB queries.\n\n"
        "Return ONLY the label.\n"
    )

    def router(state: RouterState):
        msgs = state.get("messages", [])
        q = _last_user_text(msgs)
        transcript = _messages_to_transcript(msgs, max_turns=8)

        payload = (
            "Conversation transcript:\n"
            f"{transcript}\n\n"
            "Latest user question:\n"
            f"{q}"
        )

        msg = llm_router.invoke(
            [SystemMessage(content=route_prompt), HumanMessage(content=payload)]
        )
        label = _safe_text(msg.content).strip().lower()
        if label not in ("sql", "graph", "tools", "general"):
            label = "general"

        dbg = _merge_debug(state, router_label=label, router_raw=msg.content, routed_to=label)
        return {"route": label, "debug": dbg}

    def sql_node(state: RouterState):
        standalone = _rewrite_to_standalone(llm_router, state["messages"])
        out = sql_answer(settings, standalone)
        dbg = _merge_debug(state, routed_to="sql", sql=out, standalone_question=standalone)
        return {"route": "sql", "messages": [AIMessage(content=str(out["answer"]))], "debug": dbg}

    def graph_node(state: RouterState):
        standalone = _rewrite_to_standalone(llm_router, state["messages"])
        out = graph_answer(settings, standalone)
        dbg = _merge_debug(state, routed_to="graph", graph=out.get("debug", {}), standalone_question=standalone)
        return {"route": "graph", "messages": [AIMessage(content=str(out["answer"]))], "debug": dbg}

    def tools_node(state: RouterState):
        out_state = tools_graph.invoke({"messages": state["messages"]})
        out_msgs = out_state.get("messages", [])
        tools_used = _extract_tool_names(out_msgs)

        dbg = _merge_debug(
            state,
            routed_to="tools",
            tools_used=tools_used,
            tools_graph={"messages_len": len(out_msgs)},
        )
        return {"route": "tools", "messages": out_msgs, "debug": dbg}

    def general_node(state: RouterState):
        # Use the conversation itself (not just last message)
        convo = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))]
        msg = llm_router.invoke([SystemMessage(content="You are a helpful assistant.")] + convo)
        dbg = _merge_debug(state, routed_to="general")
        return {"route": "general", "messages": [AIMessage(content=_safe_text(msg.content))], "debug": dbg}

    g = StateGraph(RouterState)
    g.add_node("router", router)
    g.add_node("sql", sql_node)
    g.add_node("graph", graph_node)
    g.add_node("tools", tools_node)
    g.add_node("general", general_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"sql": "sql", "graph": "graph", "tools": "tools", "general": "general"},
    )
    g.add_edge("sql", END)
    g.add_edge("graph", END)
    g.add_edge("tools", END)
    g.add_edge("general", END)

    return g.compile()
