from typing import List, Dict, Any, Optional


def reciprocal_rank_fusion(
    sparse_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 30,
    weight_sparse: float = 1.0,
    weight_dense: float = 1.0
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    doc_meta: Dict[str, Dict[str, Any]] = {}
    sparse_ranks: Dict[str, int] = {}
    dense_ranks: Dict[str, int] = {}

    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc["id"]
        doc_meta[doc_id] = doc
        sparse_ranks[doc_id] = rank
        scores[doc_id] = scores.get(doc_id, 0.0) + weight_sparse * (1.0 / (k + rank))

    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        if doc_id not in doc_meta:
            doc_meta[doc_id] = doc
        dense_ranks[doc_id] = rank
        scores[doc_id] = scores.get(doc_id, 0.0) + weight_dense * (1.0 / (k + rank))

    # Sort documents by descending RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)

    results = []
    for rank, doc_id in enumerate(sorted_doc_ids[:top_k], start=1):
        doc = doc_meta[doc_id]
        results.append({
            "id": doc_id,
            "text": doc["text"],
            "score": float(scores[doc_id]),
            "rank": rank,
            "method": "Hybrid (RRF)",
            "sparse_rank": sparse_ranks.get(doc_id, None),
            "dense_rank": dense_ranks.get(doc_id, None)
        })

    return results


def linear_score_fusion(
    sparse_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    alpha: float = 0.5,
    top_k: int = 30
) -> List[Dict[str, Any]]:
    def min_max_normalize(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return {r["id"]: 1.0 for r in results}
        return {r["id"]: (r["score"] - min_s) / (max_s - min_s) for r in results}

    norm_sparse = min_max_normalize(sparse_results)
    norm_dense = min_max_normalize(dense_results)

    all_ids = set(norm_sparse.keys()) | set(norm_dense.keys())
    doc_meta = {r["id"]: r for r in sparse_results + dense_results}

    blended_scores = {}
    for doc_id in all_ids:
        s_score = norm_sparse.get(doc_id, 0.0)
        d_score = norm_dense.get(doc_id, 0.0)
        blended_scores[doc_id] = alpha * d_score + (1.0 - alpha) * s_score

    sorted_ids = sorted(blended_scores.keys(), key=lambda d: blended_scores[d], reverse=True)

    results = []
    for rank, doc_id in enumerate(sorted_ids[:top_k], start=1):
        doc = doc_meta[doc_id]
        results.append({
            "id": doc_id,
            "text": doc["text"],
            "score": float(blended_scores[doc_id]),
            "rank": rank,
            "method": f"Hybrid (Linear alpha={alpha})"
        })

    return results
