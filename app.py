from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from orchestrator.settings import Settings
from orchestrator.factories import get_llm
from orchestrator.sql_agent import sql_answer
from orchestrator.graph_agent import graph_answer
from orchestrator.tools import run_tools_once
from orchestrator.graphs import build_router_graph, build_tools_agent_graph

load_dotenv()

st.set_page_config(page_title="Multi-Agent Orchestration (LangGraph)", page_icon="🧭", layout="wide")


def _dict_messages_to_lc(messages: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


def _extract_tool_names_from_messages(messages: list[BaseMessage]) -> list[str]:
    names: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    n = tc.get("name")
                else:
                    n = getattr(tc, "name", None)
                if n:
                    names.append(str(n))
    deduped: list[str] = []
    for n in names:
        if n not in deduped:
            deduped.append(n)
    return deduped


def _rewrite_followup_to_standalone(settings: Settings, chat_messages: list[dict], question: str) -> str:
    """
    Used in the *direct* SQL/Graph pages to make follow-ups work better.
    Router graph already does this internally.
    """
    user_count = sum(1 for m in chat_messages if m.get("role") == "user")
    if user_count <= 1:
        return question

    llm = get_llm(settings, temperature=0)

    # Build a short transcript
    recent = chat_messages[-12:]
    lines = []
    for m in recent:
        if m.get("role") == "user":
            lines.append(f"User: {m.get('content','')}")
        else:
            lines.append(f"Assistant: {m.get('content','')}")
    transcript = "\n".join(lines)

    prompt = (
        "Rewrite the user's latest question into a standalone question.\n"
        "Do NOT answer the question.\n\n"
        f"Conversation:\n{transcript}\n\n"
        f"Latest user question:\n{question}\n\n"
        "Standalone question:"
    )

    msg = llm.invoke(
        [
            SystemMessage(content="You rewrite follow-up questions into standalone questions."),
            HumanMessage(content=prompt),
        ]
    )
    rewritten = (msg.content or "").strip()
    return rewritten or question


# --- Sidebar ---
st.sidebar.title("🧭 Multi-Agent Orchestration")

page = st.sidebar.radio(
    "Navigation",
    ["Router Chat", "SQL Agent", "Graph Agent", "Tools Agent", "Settings"],
    index=0,
)

# Runtime settings overrides (UI -> env-like)
st.sidebar.subheader("Model")
# llm_model = st.sidebar.text_input("LLM_MODEL (Groq)", value=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"))
MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

default_model = os.getenv("LLM_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
if default_model not in MODEL_OPTIONS:
    MODEL_OPTIONS.insert(0, default_model)

llm_model = st.sidebar.selectbox("LLM_MODEL", MODEL_OPTIONS, index=MODEL_OPTIONS.index(default_model))

st.sidebar.subheader("SQL (SQLite)")
sqlite_path = st.sidebar.text_input("SQLITE_PATH", value=os.getenv("SQLITE_PATH", "student.db"))

st.sidebar.subheader("Neo4j (Graph DB)")
neo4j_uri = st.sidebar.text_input("NEO4J_URI", value=os.getenv("NEO4J_URI", ""))
neo4j_username = st.sidebar.text_input("NEO4J_USERNAME", value=os.getenv("NEO4J_USERNAME", ""))
neo4j_password = st.sidebar.text_input("NEO4J_PASSWORD", value=os.getenv("NEO4J_PASSWORD", ""), type="password")

st.sidebar.subheader("UI")
show_routing = st.sidebar.checkbox("Show routed agent", value=True)
show_tools_used = st.sidebar.checkbox("Show tools used", value=True)

settings = Settings(
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    llm_model=llm_model,
    sqlite_path=sqlite_path,
    neo4j_uri=neo4j_uri,
    neo4j_username=neo4j_username,
    neo4j_password=neo4j_password,
    wiki_doc_content_chars_max=int(os.getenv("WIKI_DOC_CHARS", "2000")),
    debug=os.getenv("DEBUG", "0") in ("1", "true", "True"),
)


@st.cache_resource
def _router_graph_cached(model: str):
    s = Settings(
        groq_api_key=settings.groq_api_key,
        llm_model=model,
        sqlite_path=settings.sqlite_path,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_username,
        neo4j_password=settings.neo4j_password,
        wiki_doc_content_chars_max=settings.wiki_doc_content_chars_max,
        debug=settings.debug,
    )
    return build_router_graph(s)


@st.cache_resource
def _tools_graph_cached(model: str):
    s = Settings(
        groq_api_key=settings.groq_api_key,
        llm_model=model,
        sqlite_path=settings.sqlite_path,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_username,
        neo4j_password=settings.neo4j_password,
        wiki_doc_content_chars_max=settings.wiki_doc_content_chars_max,
        debug=settings.debug,
    )
    return build_tools_agent_graph(s)


# --- Pages ---
if page == "Router Chat":
    st.title("🧭 Router Chat (LangGraph)")
    st.write("Multi-turn chat. The router chooses SQL / Graph / Tools / General automatically.")

    if "router_messages" not in st.session_state:
        st.session_state.router_messages = [
            {"role": "assistant", "content": "Hi! Ask a question — I will route it to the right agent."}
        ]

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Reset chat", key="reset_router"):
            st.session_state.router_messages = [
                {"role": "assistant", "content": "Chat reset. Ask a question!"}
            ]
            st.rerun()

    for m in st.session_state.router_messages:
        with st.chat_message(m["role"]):
            meta = m.get("meta") or {}
            if m["role"] == "assistant" and show_routing and meta.get("route"):
                st.caption(f"🧭 Routed to: `{meta['route']} agent`")
            if m["role"] == "assistant" and show_tools_used and meta.get("tools_used"):
                tools_line = ", ".join([f"`{t}`" for t in meta["tools_used"]])
                st.caption(f"🧰 Tools used: {tools_line}")
            st.write(m["content"])

    prompt = st.chat_input("Ask a question...", key="router_chat_input")
    if prompt:
        st.session_state.router_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        try:
            with st.chat_message("assistant"):
                route_slot = st.empty()
                tools_slot = st.empty()
                answer_slot = st.empty()

                with st.spinner("Thinking..."):
                    graph = _router_graph_cached(settings.llm_model)
                    msgs = _dict_messages_to_lc(st.session_state.router_messages)

                    out = graph.invoke({"messages": msgs})
                    out_msgs = out.get("messages", []) or []

                    last_ai = next((mm for mm in reversed(out_msgs) if isinstance(mm, AIMessage)), None)
                    answer = last_ai.content if last_ai else "(no answer)"

                    dbg = out.get("debug", {}) or {}
                    route = out.get("route") or dbg.get("router_label") or dbg.get("routed_to") or "general"
                    tools_used = dbg.get("tools_used") or []

                # Update same bubble (no jump)
                if show_routing:
                    route_slot.caption(f"🧭 Routed to: `{route}` agent")
                if show_tools_used and tools_used:
                    tools_slot.caption("🧰 Tools used: " + ", ".join([f"`{t}`" for t in tools_used]))
                answer_slot.write(answer)

            # Append to chat history AFTER we have final answer
            st.session_state.router_messages.append(
                {"role": "assistant", "content": answer, "meta": {"route": route, "tools_used": tools_used}}
            )

            with st.expander("Debug (route + steps)"):
                st.write(out.get("debug", {}))
                st.write("Messages produced:", len(out_msgs))

        except Exception as e:
            st.error(str(e))

elif page == "SQL Agent":
    st.title("🧮 SQL Agent (Chat)")
    st.write("Multi-turn SQL chat. Good for follow-ups like “now filter by …”")

    # --- Intro: what the DB contains ---
    with st.expander("📌 What's in the SQL database?", expanded=False):
        st.markdown(
            """
            The database contains information about **students, courses, enrollments, and attendance**.

            - **students**: student_id, name, program, section, year
            - **courses**: course_id, course_code, course_name, department, credits
            - **enrollments**: student-course enrollment per semester with score and grade
            - **attendance**: per-class attendance for each student in each course and semester (present = 1/0)
            - **view**: student_performance (avg_score, num_A grades, num_courses per student per semester)

            Use this chat for analytics questions like rankings, averages, cohorts, and time/semester filtering.
            """
        )

    # --- Session init ---
    if "sql_messages" not in st.session_state:
        st.session_state.sql_messages = [
            {"role": "assistant", "content": "Ask a question about the student analytics database, or try an example below."}
        ]

    # --- Reset ---
    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Reset chat", key="reset_sql"):
            st.session_state.sql_messages = [{"role": "assistant", "content": "Chat reset. Ask a SQL question!"}]
            st.rerun()

    # --- Example queries (auto-run) ---
    st.subheader("⚡ Try an example")
    e1, e2, e3 = st.columns(3)

    if e1.button("🏆 Top students (2025-Fall)", use_container_width=True):
        st.session_state.sql_demo_query = (
            "Show the top 10 students by average score in semester 2025-Fall. "
            "Use the student_performance view. Return name, program, avg_score, num_courses, and num_A."
        )

    if e2.button("📉 Lowest scoring course (2025-Fall)", use_container_width=True):
        st.session_state.sql_demo_query = (
            "In 2025-Fall, which course has the lowest average score? "
            "Return course_code, course_name, department, and avg_score."
        )

    if e3.button("🧾 Attendance < 70% (2025-Fall)", use_container_width=True):
        st.session_state.sql_demo_query = (
            "For semester 2025-Fall, show students whose overall attendance is below 70%. "
            "Compute attendance_percent as 100 * AVG(present). "
            "Return student name, program, attendance_percent, and total_classes."
        )

    demo_query = st.session_state.pop("sql_demo_query", None)

    # --- Render chat history ---
    for m in st.session_state.sql_messages:
        st.chat_message(m["role"]).write(m["content"])

    # --- Input (manual OR demo) ---
    prompt = st.chat_input("Ask a SQL question...", key="sql_chat_input")
    user_query = prompt or demo_query

    if user_query:
        st.session_state.sql_messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        try:
            # Create assistant bubble immediately (prevents flicker)
            with st.chat_message("assistant"):
                answer_slot = st.empty()

                with st.spinner("Thinking..."):
                    standalone = _rewrite_followup_to_standalone(
                        settings,
                        st.session_state.sql_messages,
                        user_query,
                    )
                    out = sql_answer(settings, standalone)
                    answer = str(out.get("answer", ""))

                answer_slot.write(answer)

            # Append to history AFTER we have the final answer
            st.session_state.sql_messages.append({"role": "assistant", "content": answer})

            with st.expander("Debug"):
                st.write("Standalone question:", standalone)
                st.json(out)

        except Exception as e:
            st.error(str(e))

elif page == "Graph Agent":
    st.title("🕸️ Graph Agent (Chat)")
    st.write("Multi-turn Cypher/Q&A chat over Neo4j.")

    # --- Explain what graph contains ---
    with st.expander("📌 What's in the Neo4j database?", expanded=False):
        st.markdown(
            """
            **Theme:** Hollywood movies.

            **Nodes**
            - `Movie`: title, tagline, released (year)
            - `Person`: name, born (year)

            **Relationships**
            - `(:Person)-[:ACTED_IN]->(:Movie)`
            - `(:Person)-[:DIRECTED]->(:Movie)`
            - `(:Person)-[:PRODUCED]->(:Movie)`

            **Examples you can ask about**
            - Movies: “The Matrix”, “Top Gun”, “Jerry Maguire”
            - People: “Tom Cruise”, “Keanu Reeves”, “Tom Hanks”
            """
        )

    with st.expander("🧠 Why Neo4j (graph DB) vs Web Search?", expanded=False):
        st.markdown(
            """
            **Neo4j is best for relationship-heavy questions** where you want exact, structured answers:
            - “Who co-starred with Tom Cruise the most?”
            - “Find actors who worked with both Tom Cruise and Tom Hanks.”
            - “Show movies connected to *The Matrix* via shared actors.”

            **Web search is best for open-world facts** (news, definitions, anything outside your dataset).  
            So: Web search = broad; Neo4j = deep structured relationships inside your graph.
            """
        )

    # --- Session init ---
    if "graph_messages" not in st.session_state:
        st.session_state.graph_messages = [
            {"role": "assistant", "content": "Ask a question about the Neo4j movies graph, or try an example below."}
        ]

    # --- Reset button ---
    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Reset chat", key="reset_graph"):
            st.session_state.graph_messages = [
                {"role": "assistant", "content": "Chat reset. Ask a graph question!"}
            ]
            st.rerun()

    # --- Example queries (auto-run) ---
    st.subheader("⚡ Try an example")
    e1, e2, e3 = st.columns(3)

    if e1.button("🎭 Similar to The Matrix (shared actors)", use_container_width=True):
        st.session_state.graph_demo_query = (
            "Find movies that share at least 2 actors with The Matrix. "
            "Return the movie titles and how many actors are shared."
        )

    if e2.button("🧭 Shortest path: Tom Hanks ↔ Tom Cruise", use_container_width=True):
        st.session_state.graph_demo_query = (
            "Show the shortest connection between Tom Hanks and Tom Cruise."
        )

    if e3.button("🎬 Recommend like Cast Away", use_container_width=True):
        st.session_state.graph_demo_query = (
            "Recommend movies like Cast Away based on shared actor and director, and also name them."
        )

    demo_query = st.session_state.pop("graph_demo_query", None)

    # --- Render chat history ---
    for m in st.session_state.graph_messages:
        st.chat_message(m["role"]).write(m["content"])

    # --- Input (manual OR demo) ---
    prompt = st.chat_input("Ask a graph question...", key="graph_chat_input")
    user_query = prompt or demo_query

    if user_query:
        st.session_state.graph_messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        try:
            # Create assistant bubble immediately (prevents flicker)
            with st.chat_message("assistant"):
                answer_slot = st.empty()

                with st.spinner("Thinking..."):
                    standalone = _rewrite_followup_to_standalone(
                        settings,
                        st.session_state.graph_messages,
                        user_query,
                    )
                    out = graph_answer(settings, standalone)
                    answer = str(out.get("answer", ""))

                answer_slot.write(answer)

            # Append to history AFTER we have the final answer
            st.session_state.graph_messages.append({"role": "assistant", "content": answer})

            with st.expander("Debug (Cypher + results)"):
                st.write("Standalone question:", standalone)
                st.json(out.get("debug", {}))

        except Exception as e:
            st.error(str(e))

elif page == "Tools Agent":
    st.title("🧰 Tools Agent (Chat)")
    st.write("Tool-Assisted Research Chat (Web + Wikipedia + arXiv + Calculator).")

    if "tools_messages" not in st.session_state:
        st.session_state.tools_messages = [{"role": "assistant", "content": "Ask a question — I'll search web/Wikipedia/arXiv and use tools when needed."}]

    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("Reset chat", key="reset_tools"):
            st.session_state.tools_messages = [{"role": "assistant", "content": "Chat reset. Ask a tools question!"}]
            st.rerun()

    for m in st.session_state.tools_messages:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask a tools question...", key="tools_chat_input")
    if prompt:
        st.session_state.tools_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            with st.chat_message("assistant"):
                tools_slot = st.empty()
                answer_slot = st.empty()

                with st.spinner("Thinking..."):
                    tools_graph = _tools_graph_cached(settings.llm_model)
                    msgs = _dict_messages_to_lc(st.session_state.tools_messages)

                    out = tools_graph.invoke({"messages": msgs})
                    out_msgs = out.get("messages", []) or []

                    last_ai = next((mm for mm in reversed(out_msgs) if isinstance(mm, AIMessage)), None)
                    answer = last_ai.content if last_ai else "(no answer)"
                    tools_used = _extract_tool_names_from_messages(out_msgs)

                if show_tools_used and tools_used:
                    tools_slot.caption("🧰 Tools used: " + ", ".join([f"`{t}`" for t in tools_used]))
                answer_slot.write(answer)

            st.session_state.tools_messages.append({"role": "assistant", "content": answer})

            with st.expander("Debug (tool messages)"):
                st.write("Tools used:", tools_used)
                st.write("Messages produced:", len(out_msgs))

        except Exception as e:
            st.error(str(e))

    # Optional: keep your old "run once each" tester as a quick health check
    with st.expander("Quick tool health-check (run each tool once)"):
        q = st.text_input("Query for one-shot tools test", key="tools_q_once")
        if st.button("Run one-shot tools", type="secondary"):
            try:
                results = run_tools_once(
                    q,
                    wiki_chars=settings.wiki_doc_content_chars_max,
                )
                for r in results:
                    with st.expander(r.tool):
                        st.write(r.output)
            except Exception as e:
                st.error(str(e))

else:
    st.title("⚙️ Settings / Health Check")
    st.write("Use this page to confirm your keys and connections.")

    if not settings.groq_api_key:
        st.warning("GROQ_API_KEY is not set. Add it in your environment or .env.")
    else:
        st.success("GROQ_API_KEY is set.")

    st.write("**Current model:**", settings.llm_model)
    st.write("**SQLite path:**", settings.sqlite_path)

    if settings.neo4j_uri:
        st.write("**Neo4j URI:**", settings.neo4j_uri)
    else:
        st.info("Neo4j not configured yet (NEO4J_URI empty). Graph Agent will fail until set.")
