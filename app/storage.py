import json
import os
import pickle
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
BM25_PATH = os.path.join(DATA_DIR, "bm25.pkl")
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")  # optional; future use


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_chunks(chunks: List[Dict[str, Any]]):
    ensure_data_dir()
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks() -> List[Dict[str, Any]]:
    if not os.path.exists(CHUNKS_PATH):
        return []
    out: List[Dict[str, Any]] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_pickle(obj: Any, path: str):
    ensure_data_dir()
    # write atomically to avoid half-written/corrupt pickle
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def load_pickle(path: str):
    """
    Safe load:
    - If file missing -> None
    - If file empty/corrupt -> None (instead of crashing with EOFError)
    """
    if not os.path.exists(path):
        return None

    # If file exists but is 0 bytes, treat as corrupted
    try:
        if os.path.getsize(path) == 0:
            return None
    except OSError:
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        # corrupted/partial pickle -> ignore and rebuild later
        return None
