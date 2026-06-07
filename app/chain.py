from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.retriever import get_retriever
from dotenv import load_dotenv
import os

load_dotenv()

# prompt template — instructs LLM to answer only from retrieved context
# {context} and {question} are placeholders filled in at runtime
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about LangGraph.
Use the context below to answer the question as thoroughly as possible.
If the context contains partial information, use it and expand naturally.
Only say "I don't have enough information" if the context has absolutely nothing relevant.

Context:
{context}

Question: {question}

Answer:
""")

# llama-3.1-8b-instant: fast, free on Groq, replacement for decommissioned llama3-8b-8192
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


def ask(question: str):
    # Step 1: retrieve top 4 relevant chunks from ChromaDB
    retriever = get_retriever()
    docs = retriever.invoke(question)

    # Step 2: join chunks into single context string for the prompt
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 3: collect unique source URLs — set() removes duplicate URLs
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    # Step 4: fill prompt with context + question, send to Groq, get answer
    # prompt | llm is LangChain's pipe syntax — output of prompt feeds into llm
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, sources

