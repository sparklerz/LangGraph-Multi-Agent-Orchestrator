from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

    # SQL (SQLite by default)
    sqlite_path: str = os.getenv("SQLITE_PATH", "student.db")

    # Neo4j Graph DB
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Tool settings
    # wiki_top_k_results: int = int(os.getenv("WIKI_TOP_K", "3"))
    wiki_doc_content_chars_max: int = int(os.getenv("WIKI_DOC_CHARS", "2000"))

    # Debug
    debug: bool = os.getenv("DEBUG", "0") in ("1","true","True","yes","YES")
