# HuggingFaceEmbeddings runs the embedding model locally — no API call needed
from langchain_huggingface import HuggingFaceEmbeddings
# Phase 1-7: from langchain_chroma import Chroma
# Phase 8: swap to Qdrant Cloud
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()


def get_retriever(source_filter: str = None):
    # load the same embedding model used during ingest — must match exactly
    # different model = different vector space = cosine similarity comparisons break
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Phase 1-7: connect to ChromaDB on disk
    # Phase 8: connect to Qdrant Cloud — reads from remote cluster, no local data/ needed
    db = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="rag_docs"
    )

    # k=20 — return top 20 most similar chunks per query
    search_kwargs = {"k": 20}
    # only add a metadata filter if a specific source was selected (not None and not "all")
    if source_filter and source_filter != "all":
        # tells Qdrant to only search chunks where source_type matches — e.g. "langgraph"
        search_kwargs["filter"] = {"source_type": source_filter}

    # wrap the Qdrant instance as a LangChain retriever with the search settings applied
    return db.as_retriever(search_kwargs=search_kwargs)


def get_retriever_docs(query: str, source_filter: str = None):
    # build the retriever with the filter applied
    retriever = get_retriever(source_filter)
    # run the query against ChromaDB — returns a List[Document] of the top k matching chunks
    return retriever.invoke(query)
