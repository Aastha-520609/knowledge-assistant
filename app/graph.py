from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.retriever import get_retriever_docs
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging
import warnings

# suppress noisy [transformers] __path__ warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── LOGGER SETUP ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("rag")
logger.setLevel(logging.DEBUG)
# only add handlers once — prevents duplicate log lines on Streamlit reruns
if not logger.handlers:
    fh = logging.FileHandler("logs/rag.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
# log() → terminal + file | logd() → file only (verbose per-chunk lines)
log = logger.info
logd = logger.debug

# actually loads the .env file — must be called before os.getenv
load_dotenv()

# create one shared LLM instance — all nodes reuse this, not created fresh each time
llm = ChatGroq(
    # which model to use on Groq
    model="llama-3.1-8b-instant",
    # reads GROQ_API_KEY from .env
    api_key=os.getenv("GROQ_API_KEY")
)

# Phase 5: load embedding model once at module level — reused across all relevance_filter calls
# same model used in ingest.py so vectors are in the same space
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── STATE ─────────────────────────────────────────────────────────────────────
# shared dict passed between all nodes — each node reads and writes to this
class GraphState(TypedDict):
    # the original question the user typed
    question: str
    # Phase 3: single rewritten query string
    # Phase 4: still kept — used as fallback label in logs
    rewritten_query: str
    # Phase 4: list of query variations generated from the rewritten query
    query_variations: List[str]
    # "langgraph", "langchain", "fastapi", "qdrant", or "all"
    source_filter: str
    # list of Document chunks returned by ChromaDB
    docs: List[Document]
    # the final assembled string of all relevant chunks — passed to the LLM
    context: str
    # the LLM's generated answer
    answer: str
    # list of "SourceName — URL" strings shown to the user
    citations: List[str]
    # Phase 5: last N turns of conversation — list of {"role": ..., "content": ...} dicts
    chat_history: List[dict]


# ── NODE 1: QUERY REWRITER ────────────────────────────────────────────────────
# takes the raw user question and rewrites it into a precise, retrieval-optimized query
# goal: surface the most relevant chunks from ChromaDB by using exact technical terminology
# scope is strictly limited to LangGraph, LangChain, FastAPI, Qdrant — no drift allowed
def query_rewriter(state: GraphState) -> GraphState:
    # Phase 5: build history string so rewriter can resolve pronouns like "one", "it", "that"
    # e.g. "how do I create one?" + history "what is a collection" → "how to create a Qdrant collection"
    history = state.get("chat_history", [])
    history_str = ""
    if history:
        history_str = "Conversation so far:\n" + "\n".join(
            f"{t['role'].capitalize()}: {t['content']}" for t in history
        ) + "\n\n"

    prompt = ChatPromptTemplate.from_template(
        "You are a search query optimizer for a technical documentation assistant.\n"
        "The documentation covers ONLY these 4 topics: LangGraph, LangChain, FastAPI, and Qdrant.\n\n"
        "{history_str}"
        "Your job:\n"
        "- If the question contains pronouns like 'it', 'one', 'this', 'that', resolve them using the conversation history above\n"
        "- Keep the rewritten query as close to the original as possible\n"
        "- Only add the tool name and 1-2 specific technical terms that directly match the question\n"
        "- PRESERVE every exact term from the original question — never replace or paraphrase them\n"
        "  (e.g. 'collection' stays 'collection', 'connected' stays 'connected', 'endpoint' stays 'endpoint')\n"
        "- Do NOT use synonyms, do NOT rephrase into academic or technical-sounding language\n"
        "- Do NOT add concepts, frameworks, or ideas that are not explicitly in the original question\n"
        "- If the question starts with 'what is' or 'what are', KEEP that exact form — only add the tool name\n"
        "- If the question is already clear, just add the tool name and return it\n"
        "- Return ONLY the rewritten question, no explanation\n\n"
        "Examples:\n"
        "  Original: 'what is a collection in Qdrant' → 'what is a Qdrant collection'\n"
        "  Original: 'what are nodes in LangGraph' → 'what are LangGraph nodes'\n"
        "  Original: 'how are things connected in LangGraph' → 'how are nodes and edges connected in LangGraph'\n"
        "  Original: 'how do I create an endpoint in FastAPI' → 'how to create a FastAPI route endpoint'\n"
        "  History: 'what is a collection in Qdrant', Original: 'how do I create one?' → 'how to create a Qdrant collection'\n\n"
        "Question: {question}"
    )
    response = (prompt | llm).invoke({"question": state["question"], "history_str": history_str})
    rewritten = response.content.strip()
    log(f"")
    log(f"{'='*60}")
    log(f"[QueryRewriter] Original : {state['question']}")
    log(f"[QueryRewriter] Rewritten: {rewritten}")
    return {**state, "rewritten_query": rewritten}


# ── NODE 1B: MULTI-QUERY GENERATOR (Phase 4) ──────────────────────────────────
# Phase 3: single rewritten query was passed directly to retriever_node
# Phase 4: takes the rewritten query and generates 3 more variations with different phrasings
# goal: cover more angles so ChromaDB finds chunks it would have missed with just one query
def multi_query_generator(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        "You are a search query generator for a technical documentation assistant.\n"
        "The documentation covers ONLY these 4 topics: LangGraph, LangChain, FastAPI, and Qdrant.\n\n"
        "Given a search query, generate 3 alternative versions of it.\n"
        "Rules:\n"
        "- KEEP the exact same keywords and technical terms from the original — do NOT replace them with synonyms\n"
        "- Only vary the sentence structure or question format (e.g. 'what is X', 'how does X work', 'explain X')\n"
        "- Do NOT use words like 'relationships', 'entities', 'associations', 'vertices', 'links' unless they are in the original\n"
        "- Do NOT introduce unrelated tools, concepts, or academic-sounding vocabulary\n"
        "- Return exactly 3 queries, one per line, no numbering, no explanation\n\n"
        "Examples:\n"
        "  Original: 'how are nodes and edges connected in LangGraph'\n"
        "  → 'what connects nodes and edges in LangGraph'\n"
        "  → 'explain how nodes and edges work in LangGraph'\n"
        "  → 'LangGraph nodes edges connection explained'\n\n"
        "Original query: {query}"
    )
    # send the rewritten query to LLM and ask for 3 variations
    response = (prompt | llm).invoke({"query": state["rewritten_query"]})
    # split response by newline — each line is one query variation
    variations = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    # include the original rewritten query as the first variation so it's always in the pool
    all_queries = [state["rewritten_query"]] + variations[:3]
    log(f"[MultiQueryGenerator] Generated {len(all_queries)} variations:")
    for i, q in enumerate(all_queries):
        log(f"  [{i+1}] {q}")
    return {**state, "query_variations": all_queries}


# ── NODE 2: RETRIEVER ─────────────────────────────────────────────────────────
def retriever_node(state: GraphState) -> GraphState:
    # Phase 3: single query retrieval
    # docs = get_retriever_docs(state["rewritten_query"], state.get("source_filter"))

    # Phase 4: run each query variation against ChromaDB and merge all results
    seen_contents = set()
    # set to track chunk content already added — used for deduplication
    all_docs = []
    for query in state["query_variations"]:
        # hit ChromaDB once per query variation — returns up to k=15 chunks each time
        docs = get_retriever_docs(query, state.get("source_filter"))
        for doc in docs:
            # use page_content as the deduplication key — same text = same chunk
            if doc.page_content not in seen_contents:
                # only add chunk if its content hasn't been seen from a previous query variation
                seen_contents.add(doc.page_content)
                all_docs.append(doc)

    log(f"[Retriever] Retrieved {len(all_docs)} unique chunks across {len(state['query_variations'])} queries")
    for i, doc in enumerate(all_docs):
        logd(f"  [{i+1}] {doc.metadata.get('source_display','?')} — {doc.metadata.get('source','')[:80]}")
    return {**state, "docs": all_docs}


# ── NODE 3: RELEVANCE FILTER ─────────────────────────────────────────────────
# Phase 4 (replaced): LLM-based filter — called LLM once per chunk, caused 429 rate limit
# errors with 32 chunks and dropped correct chunks non-deterministically
# def relevance_filter_llm(state): ...  # commented out — see git history

# Phase 5: embedding cosine similarity filter
# encodes the question once, scores each chunk against it, keeps chunks above threshold
# zero LLM calls, deterministic, fast — same model already loaded for ingest
def relevance_filter(state: GraphState) -> GraphState:
    query_text = state.get("rewritten_query") or state["question"]

    # Phase 6 fix: procedural queries ("how to", "how do") score near-zero against reference-doc chunks
    # because the embedding model sees a question-shaped sentence vs API/config-shaped text
    # skip the threshold filter entirely for these — pass all docs to reranker which handles it better
    is_procedural = query_text.lower().startswith(("how to", "how do", "steps to", "how can"))
    if is_procedural:
        scored = []
        for doc in state["docs"]:
            chunk_vec = embedding_model.encode(doc.page_content[:1000])
            question_vec = embedding_model.encode(query_text)
            score = float(np.dot(question_vec, chunk_vec) /
                          (np.linalg.norm(question_vec) * np.linalg.norm(chunk_vec) + 1e-10))
            scored.append((score, doc))
        # sort by score descending and pass top 12 to reranker — no threshold cut
        top12 = [doc for _, doc in sorted(scored, key=lambda x: x[0], reverse=True)[:12]]
        for score, doc in sorted(scored, key=lambda x: x[0], reverse=True):
            logd(f"  [RelevanceFilter] procedural score={score:.2f} — {doc.metadata.get('source','')[:70]}")
        log(f"[RelevanceFilter] procedural query — skipped threshold, top {len(top12)} by score passed to reranker")
        return {**state, "docs": top12}

    # Phase 5: embedding cosine similarity filter for definition/concept queries
    # scores against rewritten_query so exact technical terms match chunk vocabulary
    question_vec = embedding_model.encode(query_text)
    scored = []
    for doc in state["docs"]:
        chunk_vec = embedding_model.encode(doc.page_content[:1000])
        score = float(np.dot(question_vec, chunk_vec) /
                      (np.linalg.norm(question_vec) * np.linalg.norm(chunk_vec) + 1e-10))
        scored.append((score, doc))

    threshold = 0.25
    filtered = [doc for score, doc in scored if score >= threshold]
    # also cap at top 12 so reranker never scores more than 12 chunks
    filtered = sorted(filtered, key=lambda d: next(s for s, x in scored if x is d), reverse=True)[:12]

    for score, doc in scored:
        status = "✓ kept" if score >= threshold else "✗ dropped"
        logd(f"  [RelevanceFilter] {status} score={score:.2f} — {doc.metadata.get('source','')[:70]}")

    final_docs = filtered if filtered else [doc for _, doc in sorted(scored, key=lambda x: x[0], reverse=True)[:2]]
    log(f"[RelevanceFilter] {len(final_docs)}/{len(state['docs'])} chunks kept (threshold={threshold})")
    return {**state, "docs": final_docs}


# ── NODE 3B: RE-RANKER (Phase 6) ──────────────────────────────────────────────
# Phase 4 (replaced): LLM scored each chunk 1-10 — drifted badly on 12+ chunks,
# ignored prompt rules, scored integration/ops pages 9 for basic concept questions
#
# Phase 6: embedding reranker — zero LLM calls, fully deterministic
# converts the rewritten query into a declarative "answer-shaped" sentence,
# then scores each chunk by cosine similarity against that sentence
#
# why declarative? chunks are written in prose/reference style, not question style
# "A Qdrant collection is a named group of points" matches the collections/ chunk vocabulary
# much better than "what is a Qdrant collection" does — same model, different sentence shape
def reranker(state: GraphState) -> GraphState:
    rewritten = state.get("rewritten_query") or state["question"]

    # convert question into a declarative sentence the embedding model can match against chunk prose
    # e.g. "what is a Qdrant collection" → "A Qdrant collection is"
    #      "how to create a Qdrant collection" → "Creating a Qdrant collection"
    q = rewritten.lower().strip()
    if q.startswith("what is "):
        # e.g. "what is a Qdrant collection" → "Qdrant collection is. A Qdrant collection defines"
        noun = rewritten[len("what is "):].strip()
        anchor = f"{noun.capitalize()} is. A {noun} defines"
    elif q.startswith("what are "):
        noun = rewritten[len("what are "):].strip()
        anchor = f"{noun.capitalize()} are. {noun} define"
    elif q.startswith(("how to ", "how do ", "how can ")):
        # e.g. "how to create a Qdrant collection" → "Create a Qdrant collection. Qdrant collection creation API"
        for prefix in ("how to ", "how do i ", "how do you ", "how can i ", "how can you ", "how do ", "how can "):
            if q.startswith(prefix):
                action = rewritten[len(prefix):].strip()
                anchor = f"{action.capitalize()}. {action} API parameters"
                break
        else:
            anchor = rewritten
    else:
        anchor = rewritten

    anchor_vec = embedding_model.encode(anchor)
    scored = []
    for doc in state["docs"]:
        chunk_vec = embedding_model.encode(doc.page_content[:1000])
        score = float(np.dot(anchor_vec, chunk_vec) /
                      (np.linalg.norm(anchor_vec) * np.linalg.norm(chunk_vec) + 1e-10))
        scored.append((score, doc))
        logd(f"  [Reranker] score={score:.3f} — {doc.metadata.get('source','')[:70]}")

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:6]]
    log(f"[Reranker] anchor: '{anchor}'")
    log(f"[Reranker] Top {len(top_docs)} chunks kept:")
    for score, doc in scored[:6]:
        log(f"  score={score:.3f} — {doc.metadata.get('source','')[:70]}")
    return {**state, "docs": top_docs}


# ── NODE 4: CONTEXT BUILDER ───────────────────────────────────────────────────
# assembles the filtered and re-ranked chunks into a structured context string
# groups chunks by source so the LLM sees all LangGraph content together, then LangChain, etc.
# only receives top-scored chunks that passed both relevance filter and re-ranker
def context_builder(state: GraphState) -> GraphState:
    # defaultdict(list) means accessing a missing key auto-creates an empty list for it
    from collections import defaultdict
    grouped = defaultdict(list)
    for doc in state["docs"]:
        # read source_display from metadata — e.g. "LangGraph", "FastAPI"
        source = doc.metadata.get("source_display", "Unknown")
        # group all chunks from the same source together
        grouped[source].append(doc)

    sections = []
    for source, docs in grouped.items():
        # add a labeled header so the LLM knows which tool this section is about
        section = f"=== {source} Documentation ===\n"
        # join all chunks from this source with a blank line between them
        section += "\n\n".join(doc.page_content for doc in docs)
        sections.append(section)

    # join all source sections into one final context string
    context = "\n\n".join(sections)
    log(f"[ContextBuilder] Built context from {len(state['docs'])} chunks across {len(grouped)} sources ({len(context)} chars)")
    return {**state, "context": context}


# ── NODE 5: ANSWER GENERATOR ──────────────────────────────────────────────────
# receives only the filtered, re-ranked, relevant context — generates a grounded answer
# Phase 5: also receives chat_history — last N turns prepended so LLM can resolve follow-up questions
def answer_generator(state: GraphState) -> GraphState:
    # build conversation history string — each turn labeled as User/Assistant
    # empty string if no history yet (first question in session)
    history = state.get("chat_history", [])
    history_str = ""
    if history:
        history_str = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
        )

    prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable technical documentation assistant for LangGraph, LangChain, FastAPI, and Qdrant.
Your goal is to explain concepts clearly to someone learning these tools for the first time.

Rules:
- If there is conversation history, use it to understand follow-up questions and resolve pronouns like "it", "this", "that", "they"
- Read the question carefully and answer in the most natural way for that question type:
  - If it's a "what is" question: define it clearly, explain why it exists, and what problem it solves
  - If it's a "how to" question: walk through the steps practically, show the flow
  - If it's a "how does X work" question: explain the mechanism and the reasoning behind it
  - If it's a comparison or difference question: contrast clearly side by side
  - Never force a fixed template — let the question shape the answer
- Write conversationally — like a senior engineer explaining to a junior, not like a textbook
- Be thorough and detailed — do not summarize in 2-3 lines
- Use prose, bullet points, or numbered steps only where they naturally fit — do not overuse bullets
- NEVER include code unless the exact code appears word-for-word in the context below
- If no code is in the context, skip code entirely — do not write pseudo-code or invented examples
- Use the context as your primary source, expand naturally where needed
- Do NOT start by restating the question

{history_section}Context:
{context}

Question: {question}

Answer:
""")
    history_section = f"Conversation so far:\n{history_str}\n\n" if history_str else ""
    response = (prompt | llm).invoke({
        "context": state["context"],
        "question": state["question"],
        "history_section": history_section
    })
    log(f"[AnswerGenerator] Answer preview: {response.content[:150]}...")
    return {**state, "answer": response.content}


# ── NODE 6: CITATION FORMATTER ────────────────────────────────────────────────
# builds citations only from docs whose source_display matches the tools mentioned in the answer
# e.g. if answer only talks about LangGraph — only LangGraph URLs are cited
# deduplicates by URL, format: "SourceName — URL"
def citation_formatter(state: GraphState) -> GraphState:
    # lowercase the answer once so all keyword checks are case-insensitive
    answer_lower = state["answer"].lower()
    # map each source_display to keywords that must appear in the answer to justify citing it
    source_keywords = {
        "langgraph": ["langgraph", "stategraph", "stategraph"],
        "langchain": ["langchain", "lcel", "chain", "runnable"],
        "fastapi": ["fastapi", "endpoint", "router", "uvicorn"],
        "qdrant": ["qdrant", "vector store", "collection", "payload"],
    }
    # list to collect only the docs whose source tool is actually mentioned in the answer
    cited = []
    for doc in state["docs"]:
        # get the source name in lowercase — e.g. "langgraph"
        display = doc.metadata.get("source_display", "").lower()
        # get the keyword list for this source — empty list if source not in map
        keywords = source_keywords.get(display, [])
        # check if any of this source's keywords appear in the answer text
        if any(kw in answer_lower for kw in keywords):
            cited.append(doc)
        else:
            # log which sources were excluded and why
            logd(f"  [CitationFormatter] excluded — '{display}' keywords not found in answer")
    final_docs = cited if cited else state["docs"]
    citations = list({
        f"{doc.metadata.get('source_display', 'Unknown')} — {doc.metadata.get('source', '')}"
        for doc in final_docs
    })
    log(f"[CitationFormatter] {len(citations)} citations")
    for c in citations:
        log(f"  {c}")
    return {**state, "citations": citations}


# ── GRAPH ASSEMBLY ────────────────────────────────────────────────────────────
def build_graph():
    # create a new graph that uses GraphState as the shape of data flowing between nodes
    graph = StateGraph(GraphState)

    # register each function as a named node in the graph
    graph.add_node("query_rewriter", query_rewriter)
    # Phase 4: added multi_query_generator between query_rewriter and retriever
    graph.add_node("multi_query_generator", multi_query_generator)
    graph.add_node("retriever", retriever_node)
    graph.add_node("relevance_filter", relevance_filter)
    # Phase 4: added reranker between relevance_filter and context_builder
    graph.add_node("reranker", reranker)
    graph.add_node("context_builder", context_builder)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("citation_formatter", citation_formatter)

    # set query_rewriter as the first node to run when graph.invoke() is called
    graph.set_entry_point("query_rewriter")

    # Phase 3 edges:
    # graph.add_edge("query_rewriter", "retriever")
    # graph.add_edge("retriever", "relevance_filter")
    # graph.add_edge("relevance_filter", "context_builder")

    # Phase 4 edges: two new nodes inserted into the pipeline
    graph.add_edge("query_rewriter", "multi_query_generator")
    # multi_query_generator produces query_variations — retriever loops through all of them
    graph.add_edge("multi_query_generator", "retriever")
    graph.add_edge("retriever", "relevance_filter")
    # reranker scores and filters down to top 6 before context is built
    graph.add_edge("relevance_filter", "reranker")
    graph.add_edge("reranker", "context_builder")

    graph.add_edge("context_builder", "answer_generator")
    graph.add_edge("answer_generator", "citation_formatter")
    # END tells LangGraph to stop after citation_formatter — no more nodes to run
    graph.add_edge("citation_formatter", END)

    # validate the graph structure and return a runnable object
    return graph.compile()


# build and compile the graph once at module load time — ready to use when chain.py imports it
rag_graph = build_graph()
