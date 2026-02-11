import os
import json
import zipfile
from typing import List, Dict, Any
from .config import RAGConfig, ensure_session_dir
from .logging_utils import get_logger

logger = get_logger("export")


def export_rag_session_zip(
    session_id: str,
    session_path: str,
    cfg: RAGConfig,
    source_file_path: str,
    cleaned_text_pages: List[str],
    chunk_metadata: List[Dict[str, Any]],
    chat_history: List[Dict[str, Any]],
) -> str:
    ensure_session_dir(session_path)
    cleaned_text_path = os.path.join(session_path, "cleaned_text.txt")
    with open(cleaned_text_path, "w", encoding="utf-8") as f:
        for ptxt in cleaned_text_pages:
            f.write(ptxt)
            f.write("\n\n")
    meta_path = os.path.join(session_path, "chunk_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    cfg_path = os.path.join(session_path, "rag_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": cfg.model_name,
                "chunk_size_tokens": cfg.chunk_size_tokens,
                "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
                "top_k": cfg.top_k,
                "max_distance": cfg.max_distance,
                "max_query_tokens": cfg.max_query_tokens,
                "max_context_tokens": cfg.max_context_tokens,
                "max_output_tokens": cfg.max_output_tokens,
                "groq_model": cfg.groq_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    chat_path = os.path.join(session_path, "chat_history.json")
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)
    zip_path = os.path.join(session_path, f"rag_session_{session_id}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(source_file_path, arcname=os.path.basename(source_file_path))
        z.write(cleaned_text_path, arcname="cleaned_text.txt")
        z.write(meta_path, arcname="chunk_metadata.json")
        z.write(cfg_path, arcname="rag_config.json")
        z.write(chat_path, arcname="chat_history.json")
        faiss_index = os.path.join(session_path, "index.faiss")
        faiss_pkl = os.path.join(session_path, "index.pkl")
        if os.path.exists(faiss_index):
            z.write(faiss_index, arcname="index.faiss")
        if os.path.exists(faiss_pkl):
            z.write(faiss_pkl, arcname="index.pkl")
    logger.info("RAG session exported", extra={"event": "rag_export", "data": {"zip": zip_path}})
    return zip_path
