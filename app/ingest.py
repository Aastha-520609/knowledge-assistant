from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

# Set user agent so websites don't block our requests
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "rag-assistant/1.0")


def extract_text(content):
    # parsing_function receives a requests.Response object, not raw HTML string
    # so we read .text from it — then strip all tags to get clean plain text
    if content is None:
        return ""
    raw_html = content.text if hasattr(content, "text") else str(content)
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove all noise elements — navigation, footer, header, scripts, styles
    # these pollute embeddings and make similarity search less accurate
    for tag in soup(["nav", "footer", "header", "script", "style", "aside", "button"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # collapse multiple whitespace/newlines into single space for cleaner chunks
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def ingest():

    # ── STEP 1: LOAD ──────────────────────────────────────────────────────────
    # SitemapLoader: reads sitemap.xml which lists ALL pages — more reliable than crawling
    # filter_urls: only loads LangGraph pages, ignores rest of docs.langchain.com
    print("Loading docs...")
    loader = SitemapLoader(
        web_path="https://docs.langchain.com/sitemap.xml",
        filter_urls=["https://docs.langchain.com/oss/python/langgraph"],  # only LangGraph pages
        parsing_function=extract_text   # cleans HTML before storing
    )
    loader.requests_per_second = 1      # be polite to server, avoid getting blocked
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    # Safety check — stop early if no pages loaded instead of crashing later
    if not docs:
        print("No pages loaded. Check if the sitemap URL is accessible.")
        return

    # ── STEP 2: CHUNK ─────────────────────────────────────────────────────────
    # chunk_size=1000: max characters per chunk
    # chunk_overlap=150: last 150 chars of chunk1 repeat at start of chunk2 (prevents losing context at boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)  # preserves source URL metadata in each chunk
    print(f"Split into {len(chunks)} chunks")

    # Safety check — stop if chunking produced nothing
    if not chunks:
        print("No chunks created. Documents may be empty.")
        return

    # ── STEP 3: EMBED + STORE ─────────────────────────────────────────────────
    # all-MiniLM-L6-v2: small ~90MB model, downloads once and caches locally, converts text → vectors
    print("Generating embeddings (this takes a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Converts each chunk to a vector and saves both text + vector to disk — run this only once
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/chroma"  # all vectors stored here, loaded by retriever.py later
    )
    print("Done! Vectors saved to data/chroma")


if __name__ == "__main__":
    ingest()
