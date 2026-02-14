from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

from orchestrator.settings import Settings
from orchestrator.factories import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_community.graphs import Neo4jGraph
except Exception as e:  # pragma: no cover
    Neo4jGraph = None

@dataclass
class GraphAgentDebug:
    cypher: str = ""
    raw_results: Any = None
    error: str = ""

def _get_graph(settings: Settings):
    if Neo4jGraph is None:
        raise ImportError("Neo4jGraph not available. Install langchain-community[neo4j] or neo4j driver.")
    if not (settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password):
        raise ValueError("Missing NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD.")
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

def graph_answer(settings: Settings, question: str) -> Dict[str, Any]:
    """
    A simple Graph DB agent:
      1) Get graph schema
      2) Ask LLM to write Cypher (ONLY the query)
      3) Execute Cypher
      4) Ask LLM to produce a final answer grounded in results
    """
    llm = get_llm(settings, temperature=0)
    graph = _get_graph(settings)

    # schema string
    schema = getattr(graph, "schema", None)
    if callable(schema):  # older versions: graph.schema is a function
        schema = schema()
    schema = schema or "Schema not available."

    cypher_prompt = ChatPromptTemplate.from_template(
        """You are a Neo4j Cypher expert.
Given the graph schema below, write a Cypher query to answer the user question.
Return ONLY the Cypher query (no backticks, no explanation).

Schema:
{schema}

User question:
{question}
"""
    )

    to_cypher = cypher_prompt | llm | StrOutputParser()

    dbg = GraphAgentDebug()

    try:
        cypher = (to_cypher.invoke({"schema": schema, "question": question}) or "").strip()
        # Basic cleanup
        cypher = cypher.strip("` ")
        dbg.cypher = cypher
        if not cypher or len(cypher) < 6:
            raise ValueError("LLM did not produce a valid Cypher query.")

        results = graph.query(cypher)
        dbg.raw_results = results

        answer_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions using ONLY the database results.
If results are empty, say you couldn't find relevant rows.

User question:
{question}

Cypher results (JSON-like):
{results}

Answer concisely and clearly.
"""
        )
        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"question": question, "results": results})

        return {"answer": answer, "debug": dbg.__dict__, "agent": "graph"}

    except Exception as e:
        dbg.error = str(e)
        return {
            "answer": "I couldn't query the graph database for that question. Check Neo4j connection/schema and try again.",
            "debug": dbg.__dict__,
        }
