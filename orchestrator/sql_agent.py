from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import sqlite3

from sqlalchemy import create_engine

from orchestrator.settings import Settings

from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent


def _resolve_sqlite_path(settings: Settings, db_path: Optional[str] = None) -> Path:
    p = Path(db_path or settings.sqlite_path)
    if not p.is_absolute():
        # project root = parent of orchestrator/
        p = (Path(__file__).resolve().parents[1] / p).resolve()
    return p


def _make_sql_db_readonly(sqlite_path: Path) -> SQLDatabase:
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at: {sqlite_path}\n"
            f"Fix: put student.db at project root OR set SQLITE_PATH to an absolute path."
        )

    def _connect():
        return sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)

    engine = create_engine("sqlite:///", creator=_connect)
    return SQLDatabase(engine)


def _make_llm(settings: Settings):
    # ChatGroq param names differ across versions; support both.
    try:
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0,
        )
    except TypeError:
        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0,
        )


def make_sql_agent(settings: Settings, *, db_path: Optional[str] = None):
    llm = _make_llm(settings)

    sqlite_path = _resolve_sqlite_path(settings, db_path=db_path)
    db = _make_sql_db_readonly(sqlite_path)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # This is the key difference vs your b version:
    # Force the tool-calling SQL agent (most reliable on LC 1.2.x).
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        handle_parsing_errors=True,
        max_iterations=30,
        max_execution_time=60,
        verbose=bool(settings.debug),
        return_intermediate_steps=bool(settings.debug),
    )

    return agent, db, str(sqlite_path)


def sql_answer(settings: Settings, question: str, *, db_path: Optional[str] = None) -> Dict[str, Any]:
    agent, db, sqlite_path = make_sql_agent(settings, db_path=db_path)

    q = (question or "").strip().lower()

    # Keep your deterministic shortcut (nice UX)
    if any(s in q for s in ["list the tables", "list tables", "show tables", "what tables"]):
        tables = db.get_usable_table_names()
        return {"answer": "Tables: " + ", ".join(tables), "db_path": sqlite_path}

    # Run agent
    out = agent.invoke({"input": question})

    # Normalize output
    answer = out.get("output") if isinstance(out, dict) else str(out)

    result = {"answer": str(answer), "db_path": sqlite_path, "agent": "sql"}

    # If debug enabled, surface intermediate steps in Streamlit expander
    if isinstance(out, dict) and "intermediate_steps" in out:
        result["intermediate_steps"] = out["intermediate_steps"]

    return result







# from __future__ import annotations

# from pathlib import Path
# from typing import Optional, Dict, Any
# import sqlite3

# from sqlalchemy import create_engine

# from orchestrator.settings import Settings
# from orchestrator.factories import get_llm

# # --- Imports that vary across LangChain versions ---
# try:
#     # langchain >= 1.x
#     from langchain.sql_database import SQLDatabase
# except Exception:
#     # older / community
#     from langchain_community.utilities import SQLDatabase

# try:
#     from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# except Exception:
#     # older path (rare)
#     from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# try:
#     from langchain.agents import create_sql_agent
# except Exception:
#     from langchain_community.agent_toolkits.sql.base import create_sql_agent


# def _resolve_sqlite_path(settings: Settings) -> Path:
#     """
#     Resolve SQLITE_PATH relative to project root (parent of orchestrator/),
#     so Streamlit's current working directory does not break DB loading.
#     """
#     p = Path(settings.sqlite_path)
#     if not p.is_absolute():
#         p = (Path(__file__).resolve().parents[1] / p).resolve()
#     return p


# def _make_sql_db_readonly(sqlite_path: Path) -> SQLDatabase:
#     """
#     Open SQLite in READ-ONLY mode so a wrong path does NOT create an empty DB file.
#     """
#     if not sqlite_path.exists():
#         raise FileNotFoundError(
#             f"SQLite DB not found at: {sqlite_path}\n"
#             f"Fix: put student.db at the project root OR set SQLITE_PATH to an absolute path."
#         )

#     def _connect():
#         return sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)

#     engine = create_engine("sqlite:///", creator=_connect)
#     return SQLDatabase(engine)


# def _create_agent(llm, toolkit, verbose: bool):
#     """
#     Create SQL agent WITHOUT passing kwargs that frequently clash with defaults
#     in langchain-classic AgentExecutor.
#     """
#     # Keep only the safest option; many builds already set other defaults internally.
#     agent_exec_kwargs = {"handle_parsing_errors": True}

#     # Some versions accept max_iterations/max_execution_time top-level.
#     # Some accept neither.
#     # We try progressively.
#     try:
#         return create_sql_agent(
#             llm=llm,
#             toolkit=toolkit,
#             verbose=verbose,
#             max_iterations=25,
#             max_execution_time=60,
#             agent_executor_kwargs=agent_exec_kwargs,
#         )
#     except TypeError:
#         # Try without time/iteration controls to avoid duplicate kwargs.
#         return create_sql_agent(
#             llm=llm,
#             toolkit=toolkit,
#             verbose=verbose,
#             agent_executor_kwargs=agent_exec_kwargs,
#         )


# def make_sql_agent(settings: Settings, *, db_path: Optional[str] = None):
#     llm = get_llm(settings, temperature=0)

#     sqlite_path = Path(db_path).expanduser().resolve() if db_path else _resolve_sqlite_path(settings)
#     db = _make_sql_db_readonly(sqlite_path)
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#     agent = _create_agent(llm, toolkit, verbose=getattr(settings, "debug", False))
#     return agent, db, str(sqlite_path)


# def sql_answer(settings: Settings, question: str, *, db_path: Optional[str] = None) -> Dict[str, Any]:
#     agent, db, sqlite_path = make_sql_agent(settings, db_path=db_path)

#     # Deterministic shortcut so this never loops.
#     q = (question or "").strip().lower()
#     if any(s in q for s in ["list the tables", "list tables", "show tables", "what tables"]):
#         try:
#             tables = db.get_usable_table_names()
#         except Exception:
#             # fallback for older SQLDatabase implementations
#             tables = []
#         return {
#             "answer": "Tables: " + (", ".join(tables) if tables else "(none found)"),
#             "db_path": sqlite_path,
#         }

#     # Run agent
#     out = agent.invoke({"input": question})

#     # Normalize output
#     if isinstance(out, dict):
#         answer = out.get("output") or out.get("answer") or str(out)
#     else:
#         answer = str(out)

#     return {"answer": answer, "db_path": sqlite_path}
