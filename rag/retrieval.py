"""Top-k retrieval with distance threshold and token-limit enforcement."""
from typing import List, Tuple
from transformers import AutoTokenizer
from langchain_core.documents import Document
from .vector_store import FAISSStore
from .config import RAGConfig
from .logging_utils import get_logger, Timer

logger = get_logger("retrieval")


class Retriever:
    """Perform similarity search and filter by a maximum distance threshold."""
    def __init__(self, cfg: RAGConfig, store: FAISSStore):
        self.cfg = cfg
        self.store = store
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def retrieve(self, query: str, top_k: int = None, threshold: float = None) -> List[Tuple[Document, float]]:
        if top_k is None:
            top_k = self.cfg.top_k
        if threshold is None:
            threshold = self.cfg.max_distance

        q_tokens = self._count_tokens(query)
        if q_tokens > self.cfg.max_query_tokens:
            logger.info(
                "Query exceeds token limit",
                extra={"event": "refusal", "data": {"query_tokens": q_tokens, "limit": self.cfg.max_query_tokens}},
            )
            raise ValueError("Query too long. Please shorten your question.")

        assert self.store.store is not None, "Vector store not initialized"
        timer = Timer()
        docs_with_scores = self.store.store.similarity_search_with_score(query, k=top_k)
        if not docs_with_scores:
            raise ValueError("No relevant context found")
        filtered: List[Tuple[Document, float]] = []
        for doc, score in docs_with_scores:
            d = float(score)
            logger.info(
                "Retrieval score",
                extra={
                    "event": "retrieval_score",
                    "data": {"chunk_id": doc.metadata.get("chunk_id"), "distance": d},
                },
            )
            if d <= threshold:
                filtered.append((doc, d))
        if not filtered and docs_with_scores:
            best_doc, best_score = min(docs_with_scores, key=lambda x: float(x[1]))
            best_distance = float(best_score)
            filtered = [(best_doc, best_distance)]
            logger.warning(
                "Retrieval fallback applied",
                extra={"event": "retrieval_fallback", "data": {"kept_chunk_id": best_doc.metadata.get("chunk_id"), "distance": best_distance}},
            )

        logger.info(
            "Retrieval complete",
            extra={
                "event": "retrieval",
                "data": {"requested_k": top_k, "returned": len(filtered), "ms": timer.elapsed_ms()},
            },
        )
        return filtered
