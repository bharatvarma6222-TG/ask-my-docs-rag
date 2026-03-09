from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    corpus = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def bm25_search(bm25: BM25Okapi, chunks: List[Dict[str, Any]], query: str, k: int = 20) -> List[Tuple[int, float]]:
    q = _tokenize(query)
    scores = bm25.get_scores(q)
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    return idx_scores[:k]
