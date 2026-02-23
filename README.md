---
title: LangGraph Multi-Agent Orchestrator
emoji: 🧭
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
  - streamlit
  - langgraph
  - langchain
  - groq
  - sqlite
  - neo4j
short_description: A multi-agent chat app powered by LangGraph.
---

# 🧭 LangGraph Multi-Agent Orchestrator (Streamlit)

A **multi-agent chat orchestrator** built with **LangGraph + LangChain**, wrapped in a **Streamlit** UI.

It routes each user message to the best specialist:

- **SQL Agent** → relational questions over **SQLite**
- **Graph Agent** → relationship questions over **Neo4j** (Cypher)
- **Tools Agent** → **Web / Wikipedia / arXiv / Calculator** assisted answers
- **General** → normal LLM chat/explanations

---

## ✨ What you can do

### 1) Router Chat (auto orchestration)
A single chat experience where a router decides which agent to run each turn:
- SQL / Graph / Tools / General
- Handles follow-up questions by rewriting them into standalone queries

### 2) SQL Agent (SQLite analytics)
Ask questions like:
- “Top 10 students by average score in 2025-Fall”
- “Which course has the lowest average score?”
- “Who has attendance below 70%?”

### 3) Graph Agent (Neo4j Cypher Q&A)
Ask relationship-heavy questions like:
- “Movies that share actors with The Matrix”
- “Shortest path between Tom Hanks and Tom Cruise”
- “Actors who worked with both X and Y”

> Graph Agent is optional — it only works if Neo4j credentials are configured.

### 4) Tools Agent (Web + Wikipedia + arXiv + Calculator)
Useful for open-world questions outside your databases:
- quick research + citations from tools (inside the agent)
- safe arithmetic calculator tool for quick computations

---

## 🧱 Architecture (high-level)

```

User → Router (LangGraph) → one route:
├─ SQL Agent   (SQLite, read-only)
├─ Graph Agent (Neo4j → Cypher → answer grounded in results)
├─ Tools Agent (DuckDuckGo + Wikipedia + arXiv + Calculator)
└─ General     (LLM-only)

```

## 🚀 Run locally (Python)

### 1) Install
```bash
git clone https://github.com/sparklerz/LangGraph-Multi-Agent-Orchestrator
cd LangGraph-Multi-Agent-Orchestrator

pip install -r requirements.txt
````

### 2) Configure environment variables

Create a `.env` file:

```bash
# Required
GROQ_API_KEY="YOUR_GROQ_KEY"

# Recommended (the repo includes school.db)
SQLITE_PATH="school.db"

# Optional model override (Groq)
LLM_MODEL="meta-llama/llama-4-maverick-17b-128e-instruct"

# Optional: enable Neo4j Graph Agent
NEO4J_URI="neo4j+s://97329836.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"

# Optional: tool config
WIKI_DOC_CHARS="2000"

# Optional: debugging (shows more internal steps)
DEBUG="1"
```

### 3) Run

```bash
streamlit run app.py
```

---

## 🗄️ SQLite database (SQL Agent)

This repo ships with a demo SQLite DB: **`school.db`**.

If you see “DB not found” errors:

* ensure `school.db` exists in the repo root
* set `SQLITE_PATH=school.db`

### Optional: regenerate / reseed the DB

```bash
python sqlite.py
```

> `sqlite.py` supports env overrides like `SQLITE_DB`, `NUM_STUDENTS`, etc.

---

## 🕸️ Neo4j setup (Graph Agent)

Graph Agent is optional. To enable it, set:

* `NEO4J_URI`
* `NEO4J_USERNAME`
* `NEO4J_PASSWORD`

**Tip:** Neo4j’s “Movies” dataset is perfect for testing locally.

If Neo4j is not configured, Graph Agent calls will fail gracefully (and the rest of the app still works).

---
## Hugging Face Spaces (Docker) deployment

This repo is configured for **Docker Spaces** using the .github/workflows/main.yml.

### 1) Create the Space
1. Create a new Space on Hugging Face
2. Choose **SDK → Docker**
3. Push / sync this repository to the Space

### 2) Add required Secret
In your Space: **Settings → Variables and secrets**
- Add **Secret**: `GROQ_API_KEY` (required)

### 3) Add Variables (recommended)

Add **Variables**:

* `SQLITE_PATH=school.db`
* Optional: `LLM_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct`
* Optional: `WIKI_DOC_CHARS=2000`
* Optional: `DEBUG=1`

### 4) Neo4j on Spaces (optional)

If you want Graph Agent on Spaces, your Neo4j must be reachable from the public internet.
Add these as **Secrets**:

* `NEO4J_URI`
* `NEO4J_USERNAME`
* `NEO4J_PASSWORD`

> If you don’t set Neo4j creds, just avoid Graph-only questions (Router will still work for SQL/Tools/General).

---

## ⚙️ Configuration reference

| Variable         | Default                                         | Purpose                                              |
| ---------------- | ----------------------------------------------- | ---------------------------------------------------- |
| `GROQ_API_KEY`   | (none)                                          | Enables Groq LLM calls                               |
| `LLM_MODEL`      | `meta-llama/llama-4-maverick-17b-128e-instruct` | Default Groq model                                   |
| `SQLITE_PATH`    | `student.db`                                    | Path to SQLite DB (set to `school.db` for this repo) |
| `NEO4J_URI`      | (empty)                                         | Neo4j connection URI                                 |
| `NEO4J_USERNAME` | (empty)                                         | Neo4j username                                       |
| `NEO4J_PASSWORD` | (empty)                                         | Neo4j password                                       |
| `WIKI_DOC_CHARS` | `2000`                                          | Wikipedia doc truncation size for tools              |
| `DEBUG`          | `0`                                             | Show more intermediate/debug info                    |

---

## 🧪 Example prompts to try

### Router Chat

* “Show the top 10 students by average score in 2025-Fall”
* “Now only Computer Science students”
* “Explain what LangGraph is, in simple terms”
* “Summarize the latest research on graph RAG” (tools)

### SQL Agent

* “List the tables”
* “Attendance below 70% in 2025-Fall”
* “Which department has the highest average course score?”

### Graph Agent

* “Find movies that share at least 2 actors with The Matrix”
* “Shortest connection between Tom Hanks and Tom Cruise”

### Tools Agent

* “What’s the difference between BM25 and dense retrieval?”
* “Find 3 papers on multi-agent orchestration and summarize”
* “(12*(3+4))/2”

---

## 🧯 Troubleshooting

* **Missing `GROQ_API_KEY`**
  Set it in `.env` (local) or Space **Secrets** (HF).

* **SQLite DB not found**
  Ensure the file exists in the repo root and `SQLITE_PATH` matches (recommended: `school.db`).

* **Neo4j errors**
  Confirm Neo4j is running and reachable, credentials are correct, and the dataset exists.

* **Tools Agent can’t access web**
  Some hosted environments restrict outbound requests. If web tools fail, try Router/SQL/General flows.

---

## 📄 License

Apache-2.0