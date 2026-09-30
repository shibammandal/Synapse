import math
from typing import List, Dict, Set, Any


def compute_dcg_at_k(relevance_scores: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_relevant: Dict[str, int], k: int = 10) -> float:
    if not ground_truth_relevant:
        return 0.0

    # Actual DCG
    actual_relevances = [ground_truth_relevant.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    actual_dcg = compute_dcg_at_k(actual_relevances, k)

    # Ideal DCG
    ideal_relevances = sorted(ground_truth_relevant.values(), reverse=True)[:k]
    ideal_dcg = compute_dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_mrr_at_k(retrieved_ids: List[str], ground_truth_relevant: Set[str], k: int = 10) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in ground_truth_relevant:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_relevant: Set[str], k: int = 10) -> float:
    if not ground_truth_relevant:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    hits = len(retrieved_set & ground_truth_relevant)
    return hits / len(ground_truth_relevant)


def compute_precision_at_k(retrieved_ids: List[str], ground_truth_relevant: Set[str], k: int = 10) -> float:
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    hits = len(retrieved_set & ground_truth_relevant)
    return hits / k


def evaluate_retrieval_run(
    run_results: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    metrics = {f"NDCG@{k}": [] for k in k_values}
    metrics.update({f"Recall@{k}": [] for k in k_values})
    metrics.update({f"Precision@{k}": [] for k in k_values})
    metrics["MRR@10"] = []

    for qid, relevant_dict in qrels.items():
        if qid not in run_results:
            continue

        retrieved_ids = run_results[qid]
        rel_set = {doc_id for doc_id, rel in relevant_dict.items() if rel > 0}
        
        metrics["MRR@10"].append(compute_mrr_at_k(retrieved_ids, rel_set, k=10))

        for k in k_values:
            metrics[f"NDCG@{k}"].append(compute_ndcg_at_k(retrieved_ids, relevant_dict, k=k))
            metrics[f"Recall@{k}"].append(compute_recall_at_k(retrieved_ids, rel_set, k=k))
            metrics[f"Precision@{k}"].append(compute_precision_at_k(retrieved_ids, rel_set, k=k))

    # Compute macro-averages
    summary = {}
    for metric_name, values in metrics.items():
        summary[metric_name] = float(sum(values) / len(values)) if values else 0.0

    return summary
