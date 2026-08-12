from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_retriever(source_filter: str = None):
    # must use same model as ingest.py — different model = different vector space = wrong results
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # loads existing ChromaDB from disk — does NOT recreate, just reads
    db = Chroma(
        persist_directory="data/chroma",
        embedding_function=embeddings
    )

    # ── METADATA FILTER ────────────────────────────────────────────────────────
    # k=6: return top 6 most relevant chunks per query — Phase 2
    # k=10: increased for Phase 3 so relevance filter has more chunks to work with
    # more chunks = richer context = better answers after filtering
    search_kwargs = {"k": 15}
    if source_filter and source_filter != "all":
        search_kwargs["filter"] = {"source_type": source_filter}  # ChromaDB metadata filter

    # return db.as_retriever(search_kwargs=search_kwargs) # Phase 2: returns a retriever object
    # Phase 3: returns raw docs directly so graph nodes can pass them between steps
    return db.as_retriever(search_kwargs=search_kwargs)


def get_retriever_docs(query: str, source_filter: str = None):
    # Phase 3: used by graph.py retriever node — returns raw Document list instead of a retriever
    retriever = get_retriever(source_filter)
    return retriever.invoke(query)
