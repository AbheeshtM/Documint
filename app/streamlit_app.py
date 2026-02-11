import os
import sys
import uuid
import shutil
from typing import List, Dict, Any

import streamlit as st

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import RAGConfig, ensure_session_dir
from rag.logging_utils import get_logger
from rag.file_detection import detect_file
from rag.parsing import parse_file
from rag.cleaning import clean_text
from rag.chunking import TokenChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import FAISSStore
from rag.retrieval import Retriever
from rag.generation import GroqGenerator
from rag.export import export_rag_session_zip

# -------------------------------------------------
# Setup
# -------------------------------------------------
logger = get_logger("ui")

st.set_page_config(
    page_title="Production RAG",
    layout="wide",
)

st.title("DocuMint")

# -------------------------------------------------
# Helpers
# -------------------------------------------------


def init_session() -> None:
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "cfg" not in st.session_state:
        cfg = RAGConfig()
        cfg.validate()  # 🔥 fail fast
        st.session_state.cfg = cfg

    if "session_path" not in st.session_state:
        sp = os.path.join(st.session_state.cfg.session_dir, st.session_state.session_id)
        ensure_session_dir(sp)
        st.session_state.session_path = sp

    st.session_state.setdefault("parsed", None)
    st.session_state.setdefault("cleaned_pages", None)
    st.session_state.setdefault("docs", None)
    st.session_state.setdefault("faiss", None)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("source_path", None)


def reset_session() -> None:
    """Clear session state and delete artifacts."""
    logger.info("Resetting RAG session")
    sp = st.session_state.get("session_path")
    st.session_state.clear()
    if sp and os.path.exists(sp):
        try:
            shutil.rmtree(sp)
        except Exception:
            pass
    init_session()
    st.toast("Session reset", icon="🔄")


# -------------------------------------------------
# Init
# -------------------------------------------------
init_session()

# -------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    cfg = st.session_state.cfg
    cfg.groq_model = st.selectbox(
        "Groq model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0 if cfg.groq_model == "llama-3.3-70b-versatile" else 1,
    )

    cfg.chunk_size_tokens = st.slider(
        "Chunk size (tokens)",
        400,
        600,
        cfg.chunk_size_tokens,
        10,
    )

    cfg.chunk_overlap_tokens = st.slider(
        "Chunk overlap (tokens)",
        80,
        100,
        cfg.chunk_overlap_tokens,
        5,
    )

    cfg.top_k = st.slider(
        "Top-k retrieval",
        1,
        8,
        cfg.top_k,
    )

    cfg.max_distance = st.slider(
        "Max distance threshold (lower = stricter)",
        0.2,
        2.0,
        cfg.max_distance,
        0.05,
    )

    st.button("Reset Session", on_click=reset_session)

# -------------------------------------------------
# Upload & Ingestion
# -------------------------------------------------
st.subheader("Upload File")
uploaded = st.file_uploader("Any file type supported", type=None)

if uploaded:
    logger.info("File uploaded: %s", uploaded.name)

    ext = os.path.splitext(uploaded.name.lower())[1]
    tmp_path = os.path.join(st.session_state.session_path, f"source{ext}")

    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.session_state.source_path = tmp_path
    st.success(f"Uploaded: {uploaded.name}")

    cfg = st.session_state.cfg
    info = detect_file(tmp_path, cfg)

    size_mb = info.size_bytes / (1024 * 1024)
    if size_mb > cfg.max_file_size_mb:
        st.error(
            f"File too large: {size_mb:.2f} MB "
            f"(limit {cfg.max_file_size_mb} MB)"
        )
    else:
        with st.spinner("Parsing and cleaning document..."):
            try:
                parsed = parse_file(info.path, info.ext, cfg)
                cleaned_pages = clean_text(parsed)

                st.session_state.parsed = parsed
                st.session_state.cleaned_pages = cleaned_pages

                st.write(
                    f"Sections: {parsed.page_count} | "
                    f"Parsing: {parsed.parsing_method}"
                )

                logger.info(
                    "Parsed document: pages=%s method=%s",
                    parsed.page_count,
                    parsed.parsing_method,
                )
            except Exception as e:
                logger.exception("Parsing failed")
                st.error(str(e))

    if st.session_state.get("cleaned_pages"):
        with st.spinner("Chunking and embedding..."):
            chunker = TokenChunker(cfg)
            docs = chunker.chunk_pages(
                st.session_state.cleaned_pages,
                source_file=uploaded.name,
                parsing_method=st.session_state.parsed.parsing_method,
            )

            emb = EmbeddingService(cfg)
            store = FAISSStore(cfg, emb)
            store.build_index(docs)
            store.save_local(st.session_state.session_path)

            st.session_state.docs = docs
            st.session_state.faiss = store

            logger.info("FAISS index built with %d chunks", len(docs))
            st.success("Vector index built")

# -------------------------------------------------
# Question Answering
# -------------------------------------------------
if st.session_state.faiss:
    st.subheader("Ask Questions")

    q = st.text_input("Enter your question")

    if q:
        cfg = st.session_state.cfg

        retriever = Retriever(cfg, st.session_state.faiss)

        with st.spinner("Retrieving relevant chunks..."):
            try:
                results = retriever.retrieve(q, top_k=cfg.top_k, threshold=cfg.max_distance)
            except ValueError as e:
                st.warning(str(e))
                results = []

        with st.expander("Sources"):
            for doc, score in results:
                st.write(f"{doc.metadata.get('chunk_id')} | {doc.metadata.get('section_or_page')} | distance {score:.4f}")
                st.text(doc.page_content[:500])

        with st.spinner("Generating answer..."):
            try:
                generator = GroqGenerator(cfg)
                answer, citations = generator.generate(q, results)
            except EnvironmentError as e:
                st.error(str(e))
                answer, citations = ("Groq API key missing.", [])

        st.markdown(answer)

        st.session_state.chat.append(
            {
                "question": q,
                "answer": answer,
                "citations": citations,
            }
        )

        logger.info("Answered question")

# -------------------------------------------------
# Export Session
# -------------------------------------------------
if (
    st.session_state.faiss
    and st.session_state.cleaned_pages
    and st.session_state.source_path
):
    st.subheader("Download RAG Session")

    if st.button("Create ZIP"):
        logger.info("Exporting RAG session")

        pages_text_only = [t for _, t in st.session_state.cleaned_pages]
        metadata = [d.metadata for d in st.session_state.docs]

        zip_path = export_rag_session_zip(
            st.session_state.session_id,
            st.session_state.session_path,
            st.session_state.cfg,
            st.session_state.source_path,
            pages_text_only,
            metadata,
            st.session_state.chat,
        )

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download ZIP",
                data=f,
                file_name=os.path.basename(zip_path),
            )
