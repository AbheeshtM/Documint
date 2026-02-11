"""Structured logging utilities for production observability."""
import logging
import json
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """JSON formatter producing structured logs."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "time": int(record.created * 1000),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = getattr(record, "event")
        if hasattr(record, "data"):
            payload["data"] = getattr(record, "data")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for structured JSON output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class Timer:
    """Simple elapsed time helper."""
    def __init__(self):
        self.start_ts = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.start_ts) * 1000)
