# HuggingFaceEmbeddings runs the embedding model locally — no API call needed
from langchain_huggingface import HuggingFaceEmbeddings
# Chroma is the LangChain wrapper around ChromaDB — used to load and query the vector store
from langchain_chroma import Chroma


def get_retriever(source_filter: str = None):
    # load the same embedding model used during ingest — must match exactly
    # different model = different vector space = cosine similarity comparisons break
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # connect to the existing ChromaDB on disk — does NOT recreate or overwrite, just reads
    db = Chroma(
        # path where ingest.py saved the vectors
        persist_directory="data/chroma",
        # needed so ChromaDB can embed the query in the same vector space as the stored chunks
        embedding_function=embeddings
    )

    # start with k=15 — return top 15 most similar chunks per query
    # higher k gives relevance_filter more chunks to work with before dropping irrelevant ones
    search_kwargs = {"k": 20}
    # only add a metadata filter if a specific source was selected (not None and not "all")
    if source_filter and source_filter != "all":
        # tells ChromaDB to only search chunks where source_type matches — e.g. "langgraph"
        search_kwargs["filter"] = {"source_type": source_filter}

    # wrap the ChromaDB instance as a LangChain retriever with the search settings applied
    return db.as_retriever(search_kwargs=search_kwargs)


def get_retriever_docs(query: str, source_filter: str = None):
    # build the retriever with the filter applied
    retriever = get_retriever(source_filter)
    # run the query against ChromaDB — returns a List[Document] of the top k matching chunks
    return retriever.invoke(query)
