"""Text-based chunking with overlap using approximate token limits."""
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import re
from langchain_core.documents import Document
from .config import RAGConfig
from .logging_utils import get_logger, Timer

logger = get_logger("chunking")


@dataclass
class ChunkMetadata:
    """Metadata captured for each chunk."""
    chunk_id: str
    section_or_page: str
    source_file: str
    parsing_method: str


class TokenChunker:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into word windows with configured overlap."""
        words = re.findall(r"\S+", text)
        if not words:
            return []
        size = max(self.cfg.chunk_size_tokens, 1)
        overlap = max(min(self.cfg.chunk_overlap_tokens, size - 1), 0)
        step = max(size - overlap, 1)
        chunks: List[str] = []
        for start in range(0, len(words), step):
            end = min(start + size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if not chunk_text:
                continue
            if chunk_text in {"[CLS]", "[SEP]", "[CLS] [SEP]"}:
                continue
            chunks.append(chunk_text)
            if end == len(words):
                break
        return chunks

    def chunk_pages(self, cleaned_pages: List[Tuple[str, str]], source_file: str, parsing_method: str) -> List[Document]:
        """Produce LangChain Document objects with metadata per chunk."""
        timer = Timer()
        docs: List[Document] = []
        chunk_counter = 0
        for sec_id, text in cleaned_pages:
            page_chunks = self._chunk_text(text)
            for pc in page_chunks:
                chunk_counter += 1
                chunk_id = f"chunk-{chunk_counter:05d}"
                meta = {
                    "chunk_id": chunk_id,
                    "section_or_page": f"{sec_id}",
                    "source_file": source_file,
                    "parsing_method": parsing_method,
                }
                docs.append(Document(page_content=pc, metadata=meta))
        logger.info(
            "Chunking complete",
            extra={
                "event": "chunking",
                "data": {"chunks": len(docs), "ms": timer.elapsed_ms()},
            },
        )
        return docs
