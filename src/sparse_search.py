import re
import math
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi


def simple_tokenizer(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class SparseSearcher:

    def __init__(self, documents: List[Dict[str, Any]]):
        self.doc_ids = [doc["id"] for doc in documents]
        self.documents = {doc["id"]: doc for doc in documents}
        
        # Tokenize corpus for BM25
        print(f"[SparseSearcher] Tokenizing {len(documents)} documents for BM25...")
        self.tokenized_corpus = [simple_tokenizer(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("[SparseSearcher] BM25 index built successfully.")

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        tokenized_query = simple_tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair with doc IDs and sort
        indexed_scores = list(zip(self.doc_ids, scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_id, score) in enumerate(indexed_scores[:top_k], start=1):
            if score <= 0:
                continue
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "score": float(score),
                "rank": rank,
                "method": "BM25"
            })

        return results
