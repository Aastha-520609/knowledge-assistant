# Phase Evolution — AI Engineering Documentation Assistant

A log of every phase in this project: what was built, what worked, what broke, and why we moved forward.

---

## Phase 1 — Basic RAG MVP

### What was built
The simplest possible RAG pipeline — one source, one chain, one answer.

- Loaded a single documentation source via LangChain `SitemapLoader`
- Cleaned HTML with BeautifulSoup (removed nav, footer, scripts, style tags)
- Split pages into chunks: 1000 chars, 150 overlap via `RecursiveCharacterTextSplitter`
- Embedded chunks locally using `all-MiniLM-L6-v2` (no API key needed)
- Stored vectors in ChromaDB persisted to `data/chroma/`
- At query time: retrieved top-6 chunks by cosine similarity, joined them into a context string, called Groq LLM to generate an answer
- Streamlit UI with chat history and a citations expander

### What worked
End-to-end pipeline worked. Ask a question, get an answer with source links. Fast to build, easy to understand.

### Why we moved to Phase 2
The system only knew about one documentation source. If a user asked about FastAPI or Qdrant, it had no knowledge of those tools. The knowledge base was too narrow to be useful as a general AI engineering assistant. We needed multi-source support.

---

## Phase 2 — Multi-Source Knowledge Base

### What was built
Expanded the knowledge base from 1 source to 4, with metadata-based separation.

- Added 4 documentation sources: LangGraph, LangChain, FastAPI, Qdrant — all defined in `sources.py`
- Each chunk tagged at ingest time with `source_type` (e.g. `"qdrant"`) and `source_display` (e.g. `"Qdrant"`) metadata
- Retriever applies a metadata filter at query time — user can search one source or all sources
- Dynamic source dropdown in the UI — driven directly by `sources.py`, no hardcoding
- Citation format improved to `SourceName — URL`
- All 4 sources stored in one ChromaDB collection, separated only by metadata tags

### What worked
Multi-source retrieval worked well. Source filtering was clean. Adding a new source only required one entry in `sources.py`. Citations correctly attributed answers to the right documentation.

### Why we moved to Phase 3
The pipeline was a single LCEL chain — one prompt, one LLM call, no intermediate steps. There was no query understanding, no filtering of irrelevant chunks, no way to improve retrieval quality without rewriting the whole chain. As questions got more complex, the flat pipeline had no room to grow. We needed a proper orchestration layer with distinct, composable steps.

---

## Phase 3 — LangGraph Workflow Orchestration

### What was built
Replaced the flat LCEL chain with a full LangGraph `StateGraph` — each step became a named node with a clear responsibility.

- `query_rewriter`: rewrites the user's raw question into a retrieval-optimized query — preserves exact terms, adds tool name, avoids synonym drift
- `retriever`: hits ChromaDB with the rewritten query, returns top-k chunks
- `relevance_filter`: LLM judges each chunk yes/no — keeps only chunks that directly define or explain the concept asked
- `context_builder`: groups chunks by source, assembles a structured context string with labeled sections
- `answer_generator`: question-type-aware prompt — answer shape adapts to what/how/why/comparison questions
- `citation_formatter`: only cites sources whose keywords appear in the generated answer

### What worked
The modular graph made each step independently improvable. The query rewriter meaningfully improved retrieval precision. The context builder produced cleaner, more structured input to the LLM. Answer quality improved noticeably — especially for "what is" vs "how to" questions.

### Why we moved to Phase 4
Two problems emerged:

1. **Single query retrieval missed pages**: With only one query hitting ChromaDB, pages that were slightly off-angle from the rewritten query never surfaced. For example, the `manage-data/collections/` page — the exact right answer for "what is a collection in Qdrant" — was not consistently appearing in top-k results.

2. **No reranking**: All chunks that passed the relevance filter were treated equally. A chunk that vaguely mentioned the concept ranked the same as a chunk that directly defined it. The LLM received noisy context and sometimes produced weaker answers.

We needed broader retrieval coverage and a quality signal to prioritize the best chunks.

---

## Phase 4 — Retrieval Improvements

### What was built
Two new nodes added to the pipeline, plus a critical ingest fix.

**`multi_query_generator` node:**
- Takes the rewritten query and generates 3 structural variations (same keywords, different phrasing)
- All 4 queries (original + 3 variations) hit ChromaDB independently
- Results are deduplicated by `page_content` — up to ~32 unique chunks per question
- k increased from 15→20 in `retriever.py` to cast a wider net

**`reranker` node:**
- Scores each chunk 1-10 on how directly it answers the question
- Sorts descending, keeps top-6 — only the highest quality chunks reach the LLM
- Concrete scoring examples in the prompt prevent score inflation

**`ingest.py` nav-strip fix:**
- Root cause discovered: chunks 1-4 of the `collections/` page were pure sidebar nav text (600+ chars of menu links) — the actual definition was buried in chunk 5
- Fix: `extract_text` now detects whether `content` is already a `BeautifulSoup` object via `hasattr(content, 'find')` — uses it directly instead of calling `.text` first (which destroyed the HTML structure)
- Uses `soup.find("article")` as the container — Qdrant and LangGraph docs put only page body content inside `<article>`, no sidebar
- Re-ingested all 4 sources: total chunks dropped from 10047→8157 (Qdrant alone: 4599→3460), confirming nav content removed
- Key discovery: LangChain `SitemapLoader` passes a pre-parsed `BeautifulSoup` object to `parsing_function`, not raw HTML — this was undocumented behavior that caused three failed fix attempts before being confirmed via debug logging

**Prompt fixes:**
- `query_rewriter` and `multi_query_generator` prompts updated with explicit bans on synonym drift (e.g. "relationships/entities/associations/vertices" replacing "connections/edges")
- Few-shot examples added to both prompts showing correct behavior

### What worked
Multi-query retrieval consistently surfaced the `collections/` page. The reranker correctly prioritized definition chunks over tangential ones. Nav-strip fix confirmed clean — chunk 1 of `collections/` now starts directly with `"A collection is a named set of points..."`. Answer quality for both test queries (LangGraph connections, Qdrant collections) was correct.

### Why we are moving to Phase 6
Two problems remained after Phase 5:

1. **LLM reranker drift**: The reranker called the LLM once per chunk (up to 30 calls). By chunk 15+, the model ignored its own scoring rules — integration pages (`voltagent/`, `graphrag/`) scored 9 for basic concept questions, while the definition page (`collections/`) scored 3. The prompt rules were not strong enough to override the model's pattern of scoring any chunk that mentions the keyword highly.

2. **Procedural queries scored near-zero in filter**: For follow-up queries like "how to create a Qdrant collection", the embedding filter scored all chunks near-zero (highest was 0.25). The model sees a question-shaped sentence vs reference-doc-shaped text and finds low similarity. The `collections/` page scored 0.04 — it was dropped before the reranker even ran.

---

## Phase 6 — Embedding Reranker + Retrieval Robustness

### What was built
Two targeted fixes to the reranker and relevance filter.

**Embedding reranker (replaces LLM reranker):**
- Converts the rewritten query into a declarative anchor sentence before scoring
- `"what is a Qdrant collection"` → `"Qdrant collection is. A Qdrant collection defines"` — matches definition page vocabulary
- `"how to create a Qdrant collection"` → `"Create a Qdrant collection. Create a Qdrant collection API parameters"` — matches reference page vocabulary
- Each chunk scored by cosine similarity against the anchor — sorted descending, top-6 kept
- Zero LLM calls, fully deterministic, ~10x faster than LLM reranker

**Relevance filter procedural bypass:**
- Procedural queries (`how to`, `how do`, `steps to`, `how can`) skip the 0.25 threshold entirely
- Instead: all chunks scored by embedding, top-12 by score passed directly to reranker
- Concept queries still use threshold but are also capped at top-12 — prevents reranker from scoring 30 chunks

**Query rewriter fix:**
- Explicit rule added: if question starts with `what is` or `what are`, preserve that form — only add tool name
- `what are` example added to few-shot prompt
- Prevents rewriter from converting definition questions into procedural ones (observed bug: `"what is a collection"` → `"how to create a collection"`)

### What worked
- `collections/` now consistently appears in reranker top-6 for definition queries
- Procedural queries no longer drop all chunks at the filter stage
- Answer quality verified correct across all 3 test queries: definition, creation, comparison
- Reranker scores now in log as floats (0.0–1.0) — easier to reason about than 1-10 integers

### Why we are moving to Phase 7
- Embedding scores are still too flat for procedural queries — `collections/` scores 0.635 while `voltagent/` scores 0.699, a spread of only 0.06
- The anchor sentence `"Create a Qdrant collection API parameters"` is generic enough that integration pages mentioning collection creation score equally high
- A URL-based score boost is planned: after embedding scoring, chunks whose URL contains the concept keyword get +0.15 bonus — directly encodes structural knowledge that the canonical page URL matches the concept name

### What was built
Two independent improvements implemented together.

**Embedding cosine similarity filter (replaces LLM-based filter):**
- `SentenceTransformer("all-MiniLM-L6-v2")` loaded once at module level in `graph.py` — same model already used in `ingest.py`
- `relevance_filter` now encodes the question into a vector once, encodes each chunk into a vector, computes cosine similarity between them
- Chunks with similarity score ≥ 0.25 are kept — threshold is intentionally permissive since reranker handles fine-grained scoring after
- Fallback unchanged: if all chunks drop, keep top-2 by score
- Zero LLM calls, deterministic, fast — same question always gives same scores

**Conversation memory:**
- `chat_history` field added to `GraphState` — list of `{"role": ..., "content": ...}` dicts
- `ui.py` extracts the last 6 messages (3 turns) from `st.session_state.messages` before each query
- `chain.py` `ask()` accepts `chat_history` parameter and passes it into `rag_graph.invoke`
- `answer_generator` prepends conversation history to the prompt as a labeled `Conversation so far:` section
- LLM can now resolve follow-up questions like "how do I create one?" by reading prior turns

### What changed
- `graph.py`: imports added (`SentenceTransformer`, `numpy`), `embedding_model` loaded at module level, `relevance_filter` fully replaced, `chat_history` added to `GraphState`, `answer_generator` prompt updated
- `chain.py`: `ask()` signature updated to accept `chat_history`, passed into `rag_graph.invoke`
- `ui.py`: last 6 messages extracted and passed to `ask()` on every query

### Why this fully solves the Phase 4 problems
- No more 429 rate limit crashes — relevance filter makes zero LLM calls
- No more non-deterministic wrong verdicts — cosine similarity is pure math
- `collections/` page now scores high (semantically close to the question) and is never dropped
- Conversation context means follow-up questions work correctly

### Why we moved to Phase 6
Two problems remained:

1. **LLM reranker drift**: Scoring 30 chunks one-by-one caused the model to ignore its own rules by chunk 15. Integration pages (`voltagent/`, `graphrag/`) scored 9 for basic concept questions. The definition page (`collections/`) scored 3. Prompt rules were not strong enough to override the model's keyword-matching pattern.

2. **Procedural queries scored near-zero in filter**: For follow-up queries like `"how to create a Qdrant collection"`, the embedding filter scored all chunks near-zero — the model sees a question-shaped sentence vs reference-doc prose and finds low similarity. `collections/` scored 0.04 and was dropped before the reranker ran.

---

## Phase 6 — Embedding Reranker + Retrieval Robustness

### What was built
Two targeted fixes to the reranker and relevance filter.

**Embedding reranker (replaces LLM reranker):**
- Converts the rewritten query into a declarative anchor sentence before scoring
- `"what is a Qdrant collection"` → `"Qdrant collection is. A Qdrant collection defines"` — matches definition page vocabulary
- `"how to create a Qdrant collection"` → `"Create a Qdrant collection. Create a Qdrant collection API parameters"` — matches reference page vocabulary
- Each chunk scored by cosine similarity against the anchor — sorted descending, top-6 kept
- Zero LLM calls, fully deterministic, ~10x faster than LLM reranker

**Relevance filter procedural bypass:**
- Procedural queries (`how to`, `how do`, `steps to`, `how can`) skip the 0.25 threshold entirely
- Instead: all chunks scored by embedding, top-12 by score passed directly to reranker
- Concept queries still use threshold but capped at top-12 — prevents reranker from scoring 30 chunks

**Query rewriter fix:**
- Explicit rule added: if question starts with `what is` or `what are`, preserve that form — only add tool name
- `what are` example added to few-shot prompt
- Prevents rewriter from converting definition questions into procedural ones

### What worked
- `collections/` consistently appears in reranker top-6 for definition queries
- Procedural queries no longer drop all chunks at the filter stage
- Answer quality verified correct across all 3 test queries: definition, creation, comparison
- Reranker scores now floats (0.0–1.0) in log — easier to reason about than 1-10 integers

### Why we are moving to Phase 7
- Embedding scores are still too flat for procedural queries — `collections/` scores 0.635 while `voltagent/` scores 0.699, a spread of only 0.06
- The anchor sentence is generic enough that integration pages mentioning collection creation score equally high
- A URL-based score boost is planned: chunks whose URL contains the concept keyword get +0.15 bonus — encodes structural knowledge that the canonical page URL matches the concept name
