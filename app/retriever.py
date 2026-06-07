from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_retriever():
    # Load the same embedding model used during ingestion
    # must be identical — different model = different vector space = wrong results
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the existing ChromaDB from disk — does NOT recreate, just reads
    db = Chroma(
        persist_directory="data/chroma",    # same path used in ingest.py
        embedding_function=embeddings
    )

    # as_retriever() wraps the db into a LangChain-compatible search interface
    # k=4 means return top 4 most relevant chunks for any query
    return db.as_retriever(search_kwargs={"k": 6})  # increased from 4 to cast wider net for conceptual questions
