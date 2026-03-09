from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # cosine similarity = inner product if vectors are normalized
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def vector_search(index: faiss.IndexFlatIP, embedder: SentenceTransformer, query: str, k: int = 20) -> List[Tuple[int, float]]:
    q = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(q.astype("float32"), k)
    results = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx == -1:
            continue
        results.append((idx, float(score)))
    return results
