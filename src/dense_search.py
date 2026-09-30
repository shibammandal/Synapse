import numpy as np
import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class DenseSearcher:

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64
    ):
        self.doc_ids = [doc["id"] for doc in documents]
        self.documents = {doc["id"]: doc for doc in documents}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DenseSearcher] Loading embedding model '{model_name}' on device '{device}'...")
        self.model = SentenceTransformer(model_name, device=device)
        
        texts = [doc["text"] for doc in documents]
        print(f"[DenseSearcher] Encoding {len(texts)} documents into dense vectors...")
        # Normalize embeddings to enable fast cosine similarity via dot product
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print(f"[DenseSearcher] Encoded matrix shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "score": float(scores[idx]),
                "rank": rank,
                "method": "Dense"
            })

        return results
