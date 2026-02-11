"""Local embedding generation using sentence-transformers."""
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .config import RAGConfig
from .logging_utils import get_logger, Timer

logger = get_logger("embeddings")


class EmbeddingService:
    """Wrapper around HuggingFaceEmbeddings with normalization enabled."""
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.model = HuggingFaceEmbeddings(
            model_name=cfg.model_name,
            model_kwargs={"device": cfg.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, docs: List[Document]) -> None:
        """Compute embeddings for timing/logging; vector store consumes the model itself."""
        timer = Timer()
        _ = self.model.embed_documents([d.page_content for d in docs])
        logger.info(
            "Embedding complete",
            extra={"event": "embedding", "data": {"docs": len(docs), "ms": timer.elapsed_ms()}},
        )
