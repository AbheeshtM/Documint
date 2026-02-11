"""Configuration definitions for the RAG system."""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class RAGConfig:
    """Central configuration for ingestion, chunking, retrieval, and generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    top_k: int = 4
    max_pages_pdf: int = 100
    max_distance: float = 1.2
    max_query_tokens: int = 512
    max_context_tokens: int = 2000
    max_output_tokens: int = 512
    max_file_size_mb: int = 25
    session_dir: str = field(default_factory=lambda: os.path.join("sessions"))
    device: str = "cpu"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallbacks: list = field(default_factory=lambda: ["llama-3.1-8b-instant"])
    allow_zip_download: bool = True

    def validate(self) -> None:
        """Fail-fast checks for configuration values."""
        if not (400 <= self.chunk_size_tokens <= 600):
            raise ValueError("chunk_size_tokens must be between 400 and 600")
        if not (80 <= self.chunk_overlap_tokens <= 100):
            raise ValueError("chunk_overlap_tokens must be between 80 and 100")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be greater than 0")
        if self.max_pages_pdf < 1:
            raise ValueError("max_pages_pdf must be >= 1")
        if self.max_query_tokens < 1 or self.max_context_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be >= 1")


def ensure_session_dir(session_path: str) -> None:
    """Create the session directory if it does not exist."""
    os.makedirs(session_path, exist_ok=True)
