"""Local RAG Assistant - Streamlit web UI.

Run with:
    streamlit run app.py

The models load once (cached across reruns); every answer is generated
fully offline on this machine.
"""

import time

import streamlit as st

from rag import config
from rag.pipeline import answer_query
from rag.store import VectorStore

st.set_page_config(page_title="Local RAG Assistant", page_icon="📚")


@st.cache_resource(show_spinner="Loading local models via Foundry Local ...")
def get_engine():
    from rag.foundry import FoundryEngine

    return FoundryEngine()


def get_store() -> VectorStore:
    return VectorStore()


st.title("📚 Local RAG Assistant")
st.caption(
    f"Offline document Q&A · Foundry Local (`{config.CHAT_MODEL_ALIAS}` + "
    f"`{config.EMBED_MODEL_ALIAS}`) · SQLite vector store · zero cloud calls"
)

with st.sidebar:
    st.header("Knowledge base")
    store = get_store()
    chunk_count = store.count()
    if chunk_count == 0:
        st.error("Knowledge base is empty. Run `python main.py ingest` first.")
    else:
        chunks = store.all_chunks()
        sources = sorted({c.source for c in chunks})
        st.metric("Chunks", chunk_count)
        st.metric("Documents", len(sources))
        with st.expander("Documents"):
            for source in sources:
                st.write(f"- {source}")
    store.close()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about the documents ..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating (offline) ..."):
            engine = get_engine()
            store = get_store()
            started = time.perf_counter()
            result = answer_query(question, engine, store)
            elapsed = time.perf_counter() - started
            store.close()

        st.markdown(result.answer)
        if result.sources:
            with st.expander(f"Retrieved context · {elapsed:.1f}s"):
                for sc in result.sources:
                    st.markdown(
                        f"**{sc.chunk.source}** · chunk {sc.chunk.chunk_index} · "
                        f"similarity `{sc.score:.2f}`"
                    )
                    st.text(sc.chunk.content[:400] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer}
    )
