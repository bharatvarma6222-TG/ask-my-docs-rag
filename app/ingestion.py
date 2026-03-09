import re
from io import BytesIO
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader


def _clean(text: str) -> str:
    # Remove nulls and control characters that often show up in PDF extraction
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
                  " ", text)  # removes  too
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pdf_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = _clean(txt)
        if txt:
            pages.append((i + 1, txt))
    return pages


def chunk_text(
    doc_id: str,
    page: int,
    text: str,
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    text = text.strip()
    if not text:
        return chunks

    if len(text) <= chunk_chars:
        return [{
            "doc_id": doc_id,
            "page": page,
            "chunk_id": f"{doc_id}_p{page}_c0",
            "text": text
        }]

    start = 0
    c = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "doc_id": doc_id,
                "page": page,
                "chunk_id": f"{doc_id}_p{page}_c{c}",
                "text": chunk
            })
            c += 1

        if end >= len(text):
            break

        start = max(0, end - overlap_chars)

    return chunks


def build_chunks_from_pdf(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    pages = extract_pdf_pages(pdf_bytes)
    all_chunks: List[Dict[str, Any]] = []
    for page_num, page_text in pages:
        all_chunks.extend(chunk_text(filename, page_num, page_text))
    return all_chunks
