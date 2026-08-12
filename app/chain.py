# from langchain_groq import ChatGroq                      # Phase 2: used directly in chain
# from langchain_core.prompts import ChatPromptTemplate    # Phase 2: used directly in chain
# from app.retriever import get_retriever                  # Phase 2: used directly in chain
from app.graph import rag_graph                            # Phase 3: LangGraph workflow
from dotenv import load_dotenv
import os

load_dotenv()

# Phase 2: prompt template (moved to graph.py answer_generator node)
# prompt = ChatPromptTemplate.from_template("""..."""

# Phase 2: direct LLM + prompt setup (replaced by graph nodes in Phase 3)
# llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))


def ask(question: str, source_filter: str = None):
    # source_filter: "langgraph", "langchain", "fastapi", "qdrant", or None for all sources

    # Phase 2: linear LCEL chain
    # retriever = get_retriever(source_filter)
    # docs = retriever.invoke(question)
    # context = "\n\n".join(doc.page_content for doc in docs)
    # sources = list({f"{doc.metadata.get('source_display', 'Unknown')} — {doc.metadata.get('source', '')}" for doc in docs})
    # chain = prompt | llm
    # response = chain.invoke({"context": context, "question": question})
    # return response.content, sources

    # Phase 3: run LangGraph workflow
    result = rag_graph.invoke({"question": question, "source_filter": source_filter or "all"})
    return result["answer"], result["citations"]
