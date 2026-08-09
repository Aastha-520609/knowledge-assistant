# This is the configuration file.
# ── ALL DOCUMENTATION SOURCES ─────────────────────────────────────────────────
# Each source has:
#   name         → used as metadata tag on every chunk (for filtering)
#   sitemap      → URL of the sitemap.xml to load pages from
#   filter_url   → only load pages under this URL prefix
#   display_name → shown in the UI dropdown



SOURCES = [
    {
        "name": "langgraph",
        "sitemap": "https://docs.langchain.com/sitemap.xml",
        "filter_url": "https://docs.langchain.com/oss/python/langgraph",
        "display_name": "LangGraph"
    },
    {
        "name": "langchain",
        "sitemap": "https://docs.langchain.com/sitemap.xml",
        "filter_url": "https://docs.langchain.com/oss/python/langchain",
        "display_name": "LangChain"
    },
    {
        "name": "fastapi",
        "sitemap": "https://fastapi.tiangolo.com/sitemap.xml",
        "filter_url": "https://fastapi.tiangolo.com",
        "display_name": "FastAPI"
    },
    {
        "name": "qdrant",
        "sitemap": "https://qdrant.tech/sitemap.xml",
        "filter_url": "https://qdrant.tech/documentation",
        "display_name": "Qdrant"
    },
]
