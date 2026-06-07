import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # adds project root to path so 'app' module is found

import streamlit as st
from app.chain import ask  # only external dependency — ask() does all the RAG work

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
# sets browser tab title and icon
st.set_page_config(page_title="LangGraph Assistant", page_icon="🤖")
st.title("LangGraph Documentation Assistant")
st.caption("Ask anything about LangGraph — answers grounded in official docs")

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
# streamlit reruns the whole script on every user action
# st.session_state persists data across those reruns — like a memory for the session
# without this, chat history would reset every time user submits a question
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"/"assistant", "content": "...", "sources": [...]}

# ── RENDER PREVIOUS MESSAGES ──────────────────────────────────────────────────
# on every rerun, redraw all past messages so chat history stays visible
for message in st.session_state.messages:
    with st.chat_message(message["role"]):      # "user" = right bubble, "assistant" = left bubble
        st.write(message["content"])
        # only show sources for assistant messages, and only if sources exist
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):        # collapsible — user clicks to see citation URLs
                for source in message["sources"]:
                    st.write(source)

# ── HANDLE NEW INPUT ──────────────────────────────────────────────────────────
# st.chat_input renders a fixed input box at the bottom of the page
# := walrus operator — assigns query AND checks if it's non-empty in one line
if query := st.chat_input("Ask about LangGraph..."):

    # immediately show user's message in the chat before waiting for answer
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # call the RAG chain — this does retrieval + prompt building + Groq API call
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):     # loading animation while waiting for Groq response
            answer, sources = ask(query)
        st.write(answer)
        with st.expander("Sources"):        # show which doc pages the answer came from
            for source in sources:
                st.write(source)

    # save assistant response to session so it persists on next rerun
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
