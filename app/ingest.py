from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from app.sources import SOURCES
import os
import re

load_dotenv()
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "rag-assistant/1.0")

def extract_text(content):
    # parsing_function receives a requests.Response object
    if content is None:
        return ""
    raw_html = content.text if hasattr(content, "text") else str(content)
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove noise elements — nav, footer, scripts pollute embeddings
    for tag in soup(["nav", "footer", "header", "script", "style", "aside", "button"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # collapse multiple whitespaces into single space for cleaner chunks
    return re.sub(r'\s+', ' ', text).strip()


def ingest():
    # chunk_size=1000: max characters per chunk
    # chunk_overlap=150: overlap prevents losing context at chunk boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    # all-MiniLM-L6-v2: local model, cached after first run, no API needed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_chunks = []

    # ── LOOP THROUGH ALL SOURCES ───────────────────────────────────────────────
    # each source is loaded, chunked, and tagged with metadata before storing
    for source in SOURCES:
        print(f"\nLoading {source['display_name']} docs...")

        loader = SitemapLoader(
            web_path=source["sitemap"],
            filter_urls=[source["filter_url"]],  # only load pages under this URL prefix
            parsing_function=extract_text
        )
        loader.requests_per_second = 1           # polite crawling, avoid getting blocked
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from {source['display_name']}")

        if not docs:
            print(f"Skipping {source['display_name']} — no pages loaded")
            continue

        chunks = splitter.split_documents(docs)

        # ── TAG EACH CHUNK WITH SOURCE METADATA ───────────────────────────────
        # source_type is used for filtering in retriever.py
        # source_display is shown in citations in the UI
        for chunk in chunks:
            chunk.metadata["source_type"] = source["name"]
            chunk.metadata["source_display"] = source["display_name"]

        print(f"Split into {len(chunks)} chunks from {source['display_name']}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all sources: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks created. Check source URLs.")
        return

    # ── EMBED + STORE ALL CHUNKS ───────────────────────────────────────────────
    # all sources stored in one ChromaDB — metadata tags separate them at query time
    print("\nGenerating embeddings and saving to ChromaDB...")
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="data/chroma"  # vectors stored here, loaded by retriever.py
    )
    print("Done! All vectors saved to data/chroma")


if __name__ == "__main__":
    ingest()
