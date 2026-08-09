from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.retriever import get_retriever
from dotenv import load_dotenv
import os

load_dotenv()

# prompt template — {context} and {question} filled at runtime
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about AI engineering documentation.
Use the context below to answer the question as thoroughly as possible.
If the context contains partial information, use it and expand naturally.
Only say "I don't have enough information" if the context has absolutely nothing relevant.

Context:
{context}

Question: {question}

Answer:
""")

# llama-3.1-8b-instant: fast, free on Groq, replaces decommissioned llama3-8b-8192
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


def ask(question: str, source_filter: str = None):
    # source_filter: "langgraph", "langchain", "fastapi", "qdrant", or None for all sources
    retriever = get_retriever(source_filter)
    docs = retriever.invoke(question)

    # join all chunks into one context string for the prompt
    context = "\n\n".join(doc.page_content for doc in docs)

    # build citations: "SourceName — URL" format, set() removes duplicates
    sources = list({
        f"{doc.metadata.get('source_display', 'Unknown')} — {doc.metadata.get('source', '')}"
        for doc in docs
    })

    # prompt | llm: pipe syntax — filled prompt feeds into LLM, response comes back
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, sources
