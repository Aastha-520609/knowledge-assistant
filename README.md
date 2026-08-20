---
title: AI Docs Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.0
app_file: app/ui.py
pinned: false
---

# AI Engineering Documentation Assistant

A production-style RAG (Retrieval-Augmented Generation) system that answers questions over pre-indexed AI engineering documentation — LangGraph, LangChain, FastAPI, and Qdrant.

---

## Architecture (Current — Phase 8, Complete)

```
User Query
    │
    ▼
[Streamlit UI]  ──  source filter (optional)
    │
    ▼
[QueryRewriter]  ──  rewrites question into retrieval-optimized query (preserves 'what is/are' form)
    │
    ▼
[MultiQueryGenerator]  ──  generates 3 variations → 4 total queries
    │
    ▼
[Retriever]  ──  hits Qdrant Cloud once per query, deduplicates, returns up to ~32 unique chunks
    │
    ▼
[RelevanceFilter]  ──  procedural queries: top-12 by score, no threshold
                    ──  concept queries: cosine similarity ≥ 0.25, capped at top-12
    │
    ▼
[Reranker]  ──  embedding reranker — declarative anchor sentence, cosine similarity, keeps top-6
    │
    ▼
[ContextBuilder]  ──  groups chunks by source, assembles structured context string
    │
    ▼
[AnswerGenerator]  ──  question-type-aware prompt + conversation history
    │
    ▼
[CitationFormatter]  ──  URL keyword match — only cites pages whose URL contains the concept keyword
    │
    ▼
[Answer + Citations]  ──  returned to UI, citations shown in expander
```

---

## Project Structure

```
Assistant/
├── app/
│   ├── sources.py      # defines all documentation sources (sitemap URLs, metadata tags)
│   ├── ingest.py       # loads, chunks, embeds, and stores docs into Qdrant Cloud
│   ├── retriever.py    # loads Qdrant Cloud and returns retriever docs with optional source filter
│   ├── chain.py        # thin bridge — invokes LangGraph workflow, returns answer + citations
│   ├── graph.py        # full LangGraph StateGraph — all RAG nodes defined here
│   └── ui.py           # Streamlit chat interface
├── .env                # GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, USER_AGENT
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component       | Tool                                            |
|----------------|-------------------------------------------------|
| LLM            | Groq — `llama-3.1-8b-instant`                  |
| Embeddings     | HuggingFace `all-MiniLM-L6-v2` (local, no API) |
| Vector Store   | Qdrant Cloud (persisted remotely)               |
| Doc Loading    | LangChain SitemapLoader                         |
| Chunking       | RecursiveCharacterTextSplitter                  |
| Orchestration  | LangGraph StateGraph                            |
| UI             | Streamlit                                       |
| Hosting        | Hugging Face Spaces                             |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file:
```
GROQ_API_KEY=<your_groq_api_key>
QDRANT_URL=<your_qdrant_cluster_url>
QDRANT_API_KEY=<your_qdrant_api_key>
USER_AGENT=rag-assistant/1.0
```

### 3. Ingest documentation
```bash
python -m app.ingest
```
Crawls all 4 documentation sources, strips HTML noise, chunks, embeds, and saves to Qdrant Cloud. Run once — vectors persist in the cloud.

### 4. Run the app
```bash
streamlit run app/ui.py
```
