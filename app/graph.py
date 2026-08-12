from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.retriever import get_retriever_docs
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── STATE ─────────────────────────────────────────────────────────────────────
# shared dict passed between all nodes — each node reads and writes to this
class GraphState(TypedDict):
    question: str
    rewritten_query: str
    source_filter: str
    docs: List[Document]
    context: str
    answer: str
    citations: List[str]


# ── NODE 1: QUERY REWRITER ────────────────────────────────────────────────────
# takes the raw user question and rewrites it into a precise, retrieval-optimized query
# goal: surface the most relevant chunks from ChromaDB by using exact technical terminology
# scope is strictly limited to LangGraph, LangChain, FastAPI, Qdrant — no drift allowed
def query_rewriter(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        "You are a search query optimizer for a technical documentation assistant.\n"
        "The documentation covers ONLY these 4 topics: LangGraph, LangChain, FastAPI, and Qdrant.\n\n"
        "Your job:\n"
        "- Identify which topic(s) the question is about\n"
        "- Rewrite it using precise technical terms from that topic's documentation\n"
        "- Make it specific enough to retrieve the most relevant documentation chunks\n"
        "- Do NOT introduce concepts, tools, or frameworks outside the 4 topics above\n"
        "- Return ONLY the rewritten question, no explanation\n\n"
        "Question: {question}"
    )
    response = (prompt | llm).invoke({"question": state["question"]})
    rewritten = response.content.strip()
    print(f"\n[QueryRewriter] Original : {state['question']}")
    print(f"[QueryRewriter] Rewritten: {rewritten}")
    return {**state, "rewritten_query": rewritten}


# ── NODE 2: RETRIEVER ─────────────────────────────────────────────────────────
def retriever_node(state: GraphState) -> GraphState:
    docs = get_retriever_docs(state["rewritten_query"], state.get("source_filter"))
    print(f"\n[Retriever] Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.metadata.get('source_display','?')} — {doc.metadata.get('source','')[:80]}")
    return {**state, "docs": docs}


# ── NODE 3: RELEVANCE FILTER ─────────────────────────────────────────────────
# acts as a quality gate between retrieval and context building
# ChromaDB returns top-6 by cosine similarity — but similarity != relevance
# this node asks the LLM to judge each chunk: does it actually help answer the question?
# drops chunks that are topically adjacent but not directly useful (e.g. graphrag-neo4j when asking about LangGraph edges)
def relevance_filter(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        "You are a relevance judge for a documentation QA system.\n"
        "Decide if a document chunk is useful for answering the user's question.\n\n"
        "Rules:\n"
        "- Answer 'yes' if the chunk is primarily about the same tool/framework the question asks about\n"
        "- Answer 'yes' if the chunk contains concepts, definitions, or examples directly related to the question\n"
        "- Answer 'no' only if the chunk is clearly about a completely different tool and has no relevant content\n"
        "- When in doubt, answer 'yes' — it is better to keep a chunk than to lose useful information\n\n"
        "Question: {question}\n\nChunk:\n{chunk}\n\n"
        "Reply with only 'yes' or 'no'."
    )
    filtered = []
    for doc in state["docs"]:
        response = (prompt | llm).invoke({"question": state["question"], "chunk": doc.page_content[:500]})
        verdict = response.content.strip().lower()
        status = "✓ kept" if verdict == "yes" else "✗ dropped"
        print(f"  [RelevanceFilter] {status} — {doc.metadata.get('source','')[:70]}")
        if verdict == "yes":
            filtered.append(doc)
    # fallback: if all chunks dropped, keep top-2 to avoid empty context
    final_docs = filtered if filtered else state["docs"][:2]
    print(f"[RelevanceFilter] {len(final_docs)}/{len(state['docs'])} chunks kept")
    return {**state, "docs": final_docs}


# ── NODE 4: CONTEXT BUILDER ───────────────────────────────────────────────────
# assembles the filtered chunks into a structured context string
# groups chunks by source so the LLM sees all LangGraph content together, then LangChain, etc.
# labels each chunk with its source URL so the LLM knows where each piece of info comes from
# only receives chunks that passed the relevance filter — no noise reaches the LLM
def context_builder(state: GraphState) -> GraphState:
    # group docs by source_display so related content is clustered together
    from collections import defaultdict
    grouped = defaultdict(list)
    for doc in state["docs"]:
        source = doc.metadata.get("source_display", "Unknown")
        grouped[source].append(doc)

    sections = []
    for source, docs in grouped.items():
        section = f"=== {source} Documentation ===\n"
        section += "\n\n".join(doc.page_content for doc in docs)
        sections.append(section)

    context = "\n\n".join(sections)
    print(f"\n[ContextBuilder] Built context from {len(state['docs'])} chunks across {len(grouped)} sources ({len(context)} chars)")
    return {**state, "context": context}


# ── NODE 5: ANSWER GENERATOR ──────────────────────────────────────────────────
# receives only the filtered, relevant context — generates a grounded answer
# context is the primary source, but the LLM can explain and elaborate naturally
# uses the original question (not rewritten) so the answer matches what the user actually asked
def answer_generator(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable technical documentation assistant for LangGraph, LangChain, FastAPI, and Qdrant.
Your goal is to explain concepts clearly to someone learning these tools for the first time.

Rules:
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

Context:
{context}

Question: {question}

Answer:
""")
    response = (prompt | llm).invoke({"context": state["context"], "question": state["question"]})
    print(f"\n[AnswerGenerator] Answer preview: {response.content[:100]}...")
    return {**state, "answer": response.content}


# ── NODE 6: CITATION FORMATTER ────────────────────────────────────────────────
# builds citations only from docs whose source_display matches the tools mentioned in the answer
# e.g. if answer only talks about LangGraph — only LangGraph URLs are cited
# deduplicates by URL, format: "SourceName — URL"
def citation_formatter(state: GraphState) -> GraphState:
    answer_lower = state["answer"].lower()
    # map each source_display to keywords that must appear in the answer to justify citing it
    source_keywords = {
        "langgraph": ["langgraph", "stategraph", "stategraph"],
        "langchain": ["langchain", "lcel", "chain", "runnable"],
        "fastapi": ["fastapi", "endpoint", "router", "uvicorn"],
        "qdrant": ["qdrant", "vector store", "collection", "payload"],
    }
    cited = []
    for doc in state["docs"]:
        display = doc.metadata.get("source_display", "").lower()
        keywords = source_keywords.get(display, [])
        if any(kw in answer_lower for kw in keywords):
            cited.append(doc)
        else:
            print(f"  [CitationFormatter] excluded — '{display}' keywords not found in answer")
    # fallback: if nothing matched, use all filtered docs
    final_docs = cited if cited else state["docs"]
    citations = list({
        f"{doc.metadata.get('source_display', 'Unknown')} — {doc.metadata.get('source', '')}"
        for doc in final_docs
    })
    print(f"[CitationFormatter] {len(citations)} citations from {len(final_docs)}/{len(state['docs'])} docs")
    return {**state, "citations": citations}


# ── GRAPH ASSEMBLY ────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("retriever", retriever_node)
    graph.add_node("relevance_filter", relevance_filter)  # Phase 3 enhancement
    graph.add_node("context_builder", context_builder)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("citation_formatter", citation_formatter)

    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "relevance_filter")       # retriever → filter → context
    graph.add_edge("relevance_filter", "context_builder")
    graph.add_edge("context_builder", "answer_generator")
    graph.add_edge("answer_generator", "citation_formatter")
    graph.add_edge("citation_formatter", END)

    return graph.compile()


rag_graph = build_graph()
