"""Text cleaning heuristics to remove headers, footers, and page numbers."""
import re
from typing import List, Tuple
from .logging_utils import get_logger, Timer
from .parsing import ParsedDocument

logger = get_logger("cleaning")


def _is_page_number(line: str) -> bool:
    s = line.strip()
    if re.fullmatch(r"[Pp]age\s*\d+(\s*/\s*\d+)?", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(doc: ParsedDocument) -> List[Tuple[str, str]]:
    """Return cleaned text per section; for PDFs apply header/footer heuristics."""
    timer = Timer()
    cleaned_pages: List[Tuple[str, str]] = []
    header_candidates = {}
    footer_candidates = {}
    for sec_id, text in doc.sections_text:
        lines = [l for l in text.splitlines()]
        if lines:
            header = lines[0].strip()
            header_candidates[header] = header_candidates.get(header, 0) + 1
            footer = lines[-1].strip()
            footer_candidates[footer] = footer_candidates.get(footer, 0) + 1
    common_headers = {h for h, c in header_candidates.items() if c >= max(2, len(doc.sections_text) // 5)}
    common_footers = {f for f, c in footer_candidates.items() if c >= max(2, len(doc.sections_text) // 5)}

    for sec_id, text in doc.sections_text:
        lines = []
        for raw in text.splitlines():
            l = raw.strip()
            if not l:
                continue
            if doc.parsing_method == "deterministic" and sec_id.startswith("page-"):
                if l in common_headers or l in common_footers:
                    continue
                if _is_page_number(l):
                    continue
            if re.fullmatch(r"[-–—_=]{3,}", l):
                continue
            lines.append(l)
        cleaned = _normalize_spaces("\n".join(lines))
        cleaned_pages.append((sec_id, cleaned))

    logger.info(
        "Cleaning complete",
        extra={"event": "cleaning", "data": {"sections": len(cleaned_pages), "ms": timer.elapsed_ms()}},
    )
    return cleaned_pages
