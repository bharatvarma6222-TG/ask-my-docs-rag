from typing import List, Dict, Any
import os
import re
import numpy as np
import faiss

from app.storage import (
    load_chunks,
    save_chunks,
    save_pickle,
    load_pickle,
    BM25_PATH,
    FAISS_PATH,
)
from app.ingestion import build_chunks_from_pdf
from app.retrieval_bm25 import build_bm25, bm25_search
from app.retrieval_vector import load_embedder, build_faiss_index, vector_search
from app.retrieval_hybrid import hybrid_merge
from app.reranker import Reranker


_reranker = None
_embedder = None
_faiss = None
_bm25 = None
_chunks = None

# Tune this based on your reranker model
RERANK_REFUSAL_THRESHOLD = -6.0  # if top score below this -> refuse


def _lazy_load_all(load_reranker: bool = False):
    global _reranker, _embedder, _faiss, _bm25, _chunks

    if _chunks is None:
        _chunks = load_chunks()

    if _embedder is None:
        _embedder = load_embedder()

    if _bm25 is None:
        _bm25 = load_pickle(BM25_PATH)

    if _faiss is None and os.path.exists(FAISS_PATH):
        _faiss = faiss.read_index(FAISS_PATH)

    if load_reranker and _reranker is None:
        _reranker = Reranker()


def ingest_pdf(pdf_bytes: bytes, filename: str) -> str:
    global _bm25, _faiss, _chunks

    _lazy_load_all(load_reranker=False)

    new_chunks = build_chunks_from_pdf(pdf_bytes, filename)
    if not new_chunks:
        raise ValueError(
            "No text extracted from PDF. If it is scanned, we need OCR.")

    all_chunks = (_chunks or []) + new_chunks
    save_chunks(all_chunks)

    bm25 = build_bm25(all_chunks)
    save_pickle(bm25, BM25_PATH)

    texts = [c["text"] for c in all_chunks]
    embs = _embedder.encode(texts, normalize_embeddings=True)
    index = build_faiss_index(np.array(embs))
    faiss.write_index(index, FAISS_PATH)

    _bm25 = bm25
    _faiss = index
    _chunks = all_chunks

    return filename


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def build_extractive_answer(chunks: List[Dict[str, Any]], max_bullets: int = 6) -> str:
    bullets: List[str] = []
    for c in chunks:
        sents = _sentences(c["text"])
        for s in sents[:2]:
            bullets.append(f"- {s} [{c['chunk_id']}]")
            if len(bullets) >= max_bullets:
                break
        if len(bullets) >= max_bullets:
            break
    return "\n".join(bullets)


def answer_query(question: str, top_k: int = 6) -> Dict[str, Any]:
    _lazy_load_all(load_reranker=True)

    if not _chunks or _bm25 is None or _faiss is None:
        return {"answer": "No documents indexed yet. Please ingest a PDF first.", "citations": [], "confidence": 0.0}

    bm25_res = bm25_search(_bm25, _chunks, question, k=30)
    vec_res = vector_search(_faiss, _embedder, question, k=30)

    merged = hybrid_merge(bm25_res, vec_res, w_bm25=0.5, w_vec=0.5, k=40)
    candidates = [_chunks[i] for i, _ in merged]

    reranked = _reranker.rerank(question, candidates, top_k=max(top_k, 12))

    # Dedup
    top: List[Dict[str, Any]] = []
    scores: List[float] = []
    seen = set()
    for c, s in reranked:
        cid = c["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        top.append(c)
        scores.append(float(s))
        if len(top) >= max(top_k, 12):
            break

    # Confidence heuristic
    confidence = 0.0
    top_score = None
    if scores:
        top_score = max(scores)
        confidence = float(1 / (1 + np.exp(-top_score / 5)))

    # Refusal condition
    if not top or (top_score is not None and top_score < RERANK_REFUSAL_THRESHOLD):
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "confidence": 0.0
        }

    answer_text = (
        "LLM generation is disabled (no OpenAI billing). "
        "Below is an extractive answer built from the most relevant evidence:\n\n"
        + build_extractive_answer(top[:top_k], max_bullets=6)
    )

    citations = []
    for c in top[:top_k]:
        snippet = c["text"][:220] + ("..." if len(c["text"]) > 220 else "")
        citations.append({"doc_id": c["doc_id"], "page": c["page"],
                         "chunk_id": c["chunk_id"], "snippet": snippet})

    return {"answer": answer_text, "citations": citations, "confidence": confidence}
