import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # adds project root to path

import streamlit as st
from app.chain import ask
from app.sources import SOURCES  # imports source list to build dropdown dynamically

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Docs Assistant", page_icon="🤖")
st.title("AI Engineering Documentation Assistant")
st.caption("Ask anything about LangGraph, LangChain, FastAPI, or Qdrant")

# ── SOURCE SELECTOR ───────────────────────────────────────────────────────────
# dropdown built dynamically from SOURCES list — add new source to sources.py and it appears here automatically
source_options = {"All Sources": "all"}
for s in SOURCES:
    source_options[s["display_name"]] = s["name"]  # "LangGraph" → "langgraph"

selected_label = st.selectbox(
    "Search in:",
    options=list(source_options.keys())
)
selected_source = source_options[selected_label]  # maps display name to source_type value

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
# st.session_state persists across reruns — without this history resets on every question
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── RENDER PREVIOUS MESSAGES ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):      # "user" or "assistant" styles the bubble
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):        # collapsible citation section
                for source in message["sources"]:
                    st.write(source)

# ── HANDLE NEW INPUT ──────────────────────────────────────────────────────────
# st.chat_input stays fixed at bottom of page
if query := st.chat_input("Ask about LangGraph, LangChain, FastAPI, or Qdrant..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # pass source filter to chain — None searches all sources
            filter_value = None if selected_source == "all" else selected_source
            answer, sources = ask(query, source_filter=filter_value)
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(source)

    # save to session so messages persist across reruns
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
