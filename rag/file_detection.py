"""File type detection and validation."""
import os
from dataclasses import dataclass
from typing import Optional
from .logging_utils import get_logger
from .config import RAGConfig

logger = get_logger("file_detection")


@dataclass
class FileInfo:
    path: str
    name: str
    ext: str
    size_bytes: int


def detect_file(path: str, cfg: RAGConfig) -> FileInfo:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    size = os.path.getsize(path)
    ext = os.path.splitext(path.lower())[1]
    name = os.path.basename(path)
    logger.info(
        "File detected",
        extra={"event": "file_detect", "data": {"file": name, "ext": ext, "size_bytes": size}},
    )
    return FileInfo(path=path, name=name, ext=ext, size_bytes=size)
