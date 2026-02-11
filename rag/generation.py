"""Grounded answer generation via Groq LLM with strict refusal logic."""
import os
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from .prompts import grounded_prompt
from .config import RAGConfig
from .logging_utils import get_logger, Timer

logger = get_logger("generation")


class GroqGenerator:
    """LLM wrapper enforcing grounded answers and token budgets."""
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is required")
        model_override = os.environ.get("GROQ_MODEL") or cfg.groq_model
        self.model_name = model_override
        self.api_key = api_key
        self.model = ChatGroq(temperature=0.1, groq_api_key=self.api_key, model_name=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.prompt = grounded_prompt()

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _build_context(self, docs_with_scores: List[Tuple[Document, float]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Concatenate retrieved chunks until hitting the context token budget."""
        parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        total_tokens = 0
        for doc, score in docs_with_scores:
            chunk_id = doc.metadata.get("chunk_id")
            section = doc.metadata.get("section_or_page")
            header = f"[chunk_id={chunk_id} | section={section} | score={score:.4f}]"
            body = doc.page_content.strip()
            segment = f"{header}\n{body}"
            seg_tokens = self._count_tokens(segment)
            if total_tokens + seg_tokens > self.cfg.max_context_tokens:
                break
            parts.append(segment)
            total_tokens += seg_tokens
            citations.append({"chunk_id": chunk_id, "section": section, "score": float(score)})
        return "\n\n".join(parts), citations

    def generate(self, question: str, docs_with_scores: List[Tuple[Document, float]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate an answer grounded in the context; refuse if context is weak."""
        if not docs_with_scores:
            logger.info("No context retrieved", extra={"event": "refusal"})
            return ("I cannot answer based on the provided document.", [])
        q_tokens = self._count_tokens(question)
        if q_tokens > self.cfg.max_query_tokens:
            logger.info(
                "Query exceeds token limit",
                extra={"event": "refusal", "data": {"query_tokens": q_tokens}},
            )
            return ("Your question is too long. Please shorten it.", [])
        context, citations = self._build_context(docs_with_scores)
        if not context:
            logger.info("Context overflow or empty", extra={"event": "refusal"})
            return ("I cannot answer based on the provided document.", [])
        timer = Timer()
        messages = self.prompt.format_messages(context=context, question=question)
        try:
            resp = self.model.invoke(messages, config={"max_tokens": self.cfg.max_output_tokens})
        except Exception as e:
            msg = str(e)
            if "model_decommissioned" in msg or "has been decommissioned" in msg:
                for candidate in [*self.cfg.groq_fallbacks]:
                    try:
                        self.model_name = candidate
                        self.model = ChatGroq(temperature=0.1, groq_api_key=self.api_key, model_name=self.model_name)
                        resp = self.model.invoke(messages, config={"max_tokens": self.cfg.max_output_tokens})
                        logger.info("Groq model fallback applied", extra={"event": "model_fallback", "data": {"model": self.model_name}})
                        break
                    except Exception:
                        continue
                else:
                    raise
            else:
                raise
        logger.info(
            "LLM call complete",
            extra={"event": "llm_call", "data": {"ms": timer.elapsed_ms()}},
        )
        answer = resp.content.strip()
        citation_items = [f"{c['chunk_id']} ({c['section']})" for c in citations]
        suffix = f" [{' ; '.join(citation_items)}]" if citation_items else " [none]"
        return (answer + suffix, citations)
