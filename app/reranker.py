from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL)
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        pairs = [(query, c["text"]) for c in candidates]
        enc = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        scores = self.model(**enc).logits.squeeze(-1)
        scores = scores.detach().cpu().tolist()

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
