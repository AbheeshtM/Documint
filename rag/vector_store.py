"""FAISS vector store management with per-session persistence."""
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embeddings import EmbeddingService
from .config import RAGConfig, ensure_session_dir
from .logging_utils import get_logger, Timer

logger = get_logger("vector_store")


class FAISSStore:
    """Convenience wrapper for building and persisting FAISS indices."""
    def __init__(self, cfg: RAGConfig, embedding_service: EmbeddingService):
        self.cfg = cfg
        self.embedding_service = embedding_service
        self.store: Optional[FAISS] = None

    def build_index(self, docs: List[Document]) -> FAISS:
        """Create a FAISS index from documents using the configured embeddings."""
        timer = Timer()
        self.store = FAISS.from_documents(docs, self.embedding_service.model)
        logger.info(
            "FAISS index created",
            extra={
                "event": "faiss_index_create",
                "data": {"docs": len(docs), "ms": timer.elapsed_ms()},
            },
        )
        return self.store

    def save_local(self, session_path: str) -> str:
        """Persist index files into the session directory."""
        ensure_session_dir(session_path)
        assert self.store is not None, "Index not built"
        self.store.save_local(session_path)
        logger.info(
            "FAISS index saved",
            extra={
                "event": "faiss_index_save",
                "data": {"path": session_path},
            },
        )
        return session_path

    def load_local(self, session_path: str) -> FAISS:
        """Load an existing FAISS index from the session directory."""
        self.store = FAISS.load_local(session_path, self.embedding_service.model, allow_dangerous_deserialization=True)
        return self.store
