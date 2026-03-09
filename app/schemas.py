from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str
    top_k: int = 6


class Citation(BaseModel):
    doc_id: str
    page: int
    chunk_id: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
