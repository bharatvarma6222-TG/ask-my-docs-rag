from typing import List, Tuple, Dict


def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {i: 1.0 for i in scores}
    return {i: (s - mn) / (mx - mn) for i, s in scores.items()}


def hybrid_merge(
    bm25_results: List[Tuple[int, float]],
    vec_results: List[Tuple[int, float]],
    w_bm25: float = 0.5,
    w_vec: float = 0.5,
    k: int = 30
) -> List[Tuple[int, float]]:
    bm = {i: s for i, s in bm25_results}
    vc = {i: s for i, s in vec_results}

    bm_n = _normalize(bm)
    vc_n = _normalize(vc)

    all_ids = set(bm_n.keys()) | set(vc_n.keys())
    merged = []
    for i in all_ids:
        score = w_bm25 * bm_n.get(i, 0.0) + w_vec * vc_n.get(i, 0.0)
        merged.append((i, score))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:k]
