import time
from typing import List, Dict, Any, Optional
from src.sparse_search import SparseSearcher
from src.dense_search import DenseSearcher
from src.hybrid_search import reciprocal_rank_fusion, linear_score_fusion
from src.reranker import CrossEncoderReranker


class HybridSearchPipeline:

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker: bool = True
    ):
        self.documents = documents
        self.sparse_searcher = SparseSearcher(documents)
        self.dense_searcher = DenseSearcher(documents, model_name=dense_model_name)
        
        self.enable_reranker = enable_reranker
        if enable_reranker:
            self.reranker = CrossEncoderReranker(model_name=reranker_model_name)
        else:
            self.reranker = None

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidates_k: int = 50,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        use_reranker: bool = True
    ) -> Dict[str, Any]:
        profiling = {}

        # 1. Sparse Search
        t0 = time.perf_counter()
        sparse_res = self.sparse_searcher.search(query, top_k=candidates_k)
        profiling["sparse_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Dense Search
        t0 = time.perf_counter()
        dense_res = self.dense_searcher.search(query, top_k=candidates_k)
        profiling["dense_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Hybrid Fusion
        t0 = time.perf_counter()
        if fusion_method == "rrf":
            hybrid_res = reciprocal_rank_fusion(
                sparse_res, dense_res, k=rrf_k, top_k=candidates_k
            )
        else:
            hybrid_res = linear_score_fusion(
                sparse_res, dense_res, alpha=alpha, top_k=candidates_k
            )
        profiling["fusion_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Cross-Encoder Re-ranking
        reranked_res = []
        if use_reranker and self.reranker is not None:
            t0 = time.perf_counter()
            reranked_res = self.reranker.rerank(
                query, hybrid_res[:candidates_k], top_k=top_k
            )
            profiling["rerank_ms"] = (time.perf_counter() - t0) * 1000
        else:
            profiling["rerank_ms"] = 0.0

        profiling["total_ms"] = sum(profiling.values())

        return {
            "query": query,
            "sparse_results": sparse_res[:top_k],
            "dense_results": dense_res[:top_k],
            "hybrid_results": hybrid_res[:top_k],
            "reranked_results": reranked_res if reranked_res else hybrid_res[:top_k],
            "profiling": profiling
        }
