"""RegRAG web UI.   Run from the project folder:   streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from config import settings  # noqa: E402
from pipeline import VERSION_NAMES, VERSIONS, RegRAG  # noqa: E402
from retrieval.embeddings import get_embedder  # noqa: E402
from retrieval.store import Store  # noqa: E402

st.set_page_config(page_title="RegRAG – RBI & SEBI Q&A", page_icon="📜", layout="wide")


@st.cache_resource(show_spinner="Loading index and models…")
def load_store():
    return Store(settings.processed_dir, settings.index_dir, get_embedder(settings.embed_model))


@st.cache_resource
def load_pipeline(version: str):
    return RegRAG(version=version, store=load_store())


try:
    store = load_store()
except Exception as e:  # index not built yet
    st.title("📜 RegRAG")
    st.error(f"Index not ready: {e}")
    st.markdown("""
**Setup steps**
1. Put PDFs in `data/raw/rbi` and `data/raw/sebi` (or run `python -m ingest.scraper --source rbi --limit 30`)
2. Build the index: `python -m ingest.build_index`
3. Refresh this page
""")
    st.stop()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    versions = list(VERSIONS)
    version = st.selectbox("Pipeline version", versions, index=versions.index(settings.default_version),
                           format_func=lambda v: f"{v} – {VERSION_NAMES[v]}")
    regs_available = sorted({c["regulator"] for c in store.chunks.values()})
    regulators = st.multiselect("Regulator", regs_available, default=regs_available,
                                help="Leave both selected to let the router decide (v5+).")
    years_known = sorted({int(c["year"]) for c in store.chunks.values() if c["year"]})
    years = None
    if len(years_known) > 1:
        y = st.slider("Document year", years_known[0], years_known[-1], (years_known[0], years_known[-1]))
        if y != (years_known[0], years_known[-1]):
            years = y
    show_debug = st.toggle("Show debug info", value=False)
    st.divider()
    st.caption(f"📚 {len(store.docs)} documents · {len(store.chunks)} chunks")
    st.caption(f"Embeddings: {settings.embed_model}")
    st.caption("Not legal advice. Always verify with the official circular.")
    if st.button("Clear chat"):
        st.session_state.history = []

st.title("📜 RegRAG")
st.caption("Ask questions about RBI & SEBI regulations. Every answer is cited to the exact circular and paragraph.")

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.markdown("**Try asking:**")
    examples = ["What is the periodicity of KYC updation for high risk customers?",
                "What are the conditions for video based customer identification (V-CIP)?",
                "उच्च जोखिम वाले ग्राहकों के लिए KYC कितनी बार अपडेट करना होता है?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()


def render(res: dict) -> None:
    st.markdown(res["answer"])
    for w in res.get("warnings", []):
        st.warning(w)
    if res.get("citations"):
        st.markdown("**Sources**")
        for c in res["citations"]:
            with st.expander(f"[{c['ref']}] {c['label']}"):
                st.write(c["text"])
                if c.get("also_in"):
                    st.caption("Same rule also appears in: " + "; ".join(t[:90] for t in c["also_in"]))
                if c.get("url"):
                    st.markdown(f"[Open official document ↗]({c['url']})")
    if show_debug:
        with st.expander("🔧 Debug"):
            st.json({k: v for k, v in res["debug"].items()})


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render(turn)

question = st.chat_input("Ask about KYC, NBFCs, mutual funds, listing rules…") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching circulars…"):
            try:
                rag = load_pipeline(version)
                regs = regulators if regulators and len(regulators) < len(regs_available) else None
                res = rag.ask(question, regulators=regs, years=years)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        render(res)
        st.caption(f"⏱ {res['debug']['total_seconds']}s · {res['debug'].get('llm_calls', 0)} LLM calls · {version}")
    st.session_state.history.append(res)
