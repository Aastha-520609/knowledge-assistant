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
    # k=6: return top 6 most relevant chunks per query
    # filter: if source_filter provided, only search chunks tagged with that source_type
    # if None: search across all sources
    search_kwargs = {"k": 6}
    if source_filter and source_filter != "all":
        search_kwargs["filter"] = {"source_type": source_filter}  # ChromaDB metadata filter

    return db.as_retriever(search_kwargs=search_kwargs)
