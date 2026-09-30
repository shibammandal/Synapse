from typing import List, Dict, Any
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CrossEncoderReranker] Loading re-ranker '{model_name}' on device '{device}'...")
        self.model = CrossEncoder(model_name, max_length=512, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Build query-document pairs
        pairs = [[query, doc["text"]] for doc in candidates]
        
        # Predict cross-encoder scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Attach new scores and preserve candidate history
        scored_candidates = []
        for doc, score in zip(candidates, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            item["initial_rank"] = doc.get("rank")
            scored_candidates.append(item)

        # Sort descending by re-ranker score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Re-assign 1-based ranks
        results = []
        for new_rank, doc in enumerate(scored_candidates[:top_k], start=1):
            doc["final_rank"] = new_rank
            doc["method"] = "Hybrid + Cross-Encoder"
            results.append(doc)

        return results
