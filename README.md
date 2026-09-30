# ⚡ Synapse
### Dual-Stage Hybrid Neural Search & Cross-Encoder Re-Ranking

An end-to-end modern Information Retrieval (IR) system implementing dual-index retrieval (sparse BM25 + dense semantic bi-encoder), rank fusion (Reciprocal Rank Fusion & linear blending), and cross-encoder re-ranking evaluated on an authentic real-world **MS-MARCO** benchmark (sampled from Microsoft Bing search queries and passages).

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    Query(["🔍 User Query"])

    subgraph S1["Stage 1: Dual First-Stage Retrieval"]
        direction LR
        Sparse["Sparse Lexical Retriever\n• Algorithm: BM25Okapi\n• Strengths: Exact keywords, IDs, acronyms\n• Pool: Top-K candidates"]
        Dense["Dense Semantic Retriever\n• Model: sentence-transformers/all-MiniLM-L6-v2\n• Strengths: Conceptual similarity, paraphrases\n• Pool: Top-K candidates"]
    end

    subgraph S2["Stage 2: Hybrid Rank Fusion"]
        Fusion["Hybrid Fusion Layer\n• Reciprocal Rank Fusion (RRF, k=60)\n• Linear Score Normalization & Blending\n• Pool: Top-M fused candidates"]
    end

    subgraph S3["Stage 3: Cross-Encoder Re-Ranking"]
        Reranker["Cross-Encoder Re-Ranker\n• Model: cross-encoder/ms-marco-MiniLM-L-6-v2\n• Full query-passage joint cross-attention\n• Pool: Top-N re-ranked results"]
    end

    Output(["🏆 Final Ranked Results"])

    Query -->|"Query text"| Sparse
    Query -->|"Query text"| Dense

    Sparse -->|"Top-K Sparse Pool"| Fusion
    Dense -->|"Top-K Dense Pool"| Fusion

    Fusion -->|"Top-M Candidates"| Reranker
    Reranker -->|"Top-N Sorted"| Output

    style Query fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
    style Output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style Sparse fill:#fff8e1,stroke:#f57f17,stroke-width:1.5px,color:#212121
    style Dense fill:#ede7f6,stroke:#512da8,stroke-width:1.5px,color:#212121
    style Fusion fill:#fce4ec,stroke:#c2185b,stroke-width:1.5px,color:#212121
    style Reranker fill:#e0f2f1,stroke:#00796b,stroke-width:1.5px,color:#212121
```

---

## 📐 Mathematical Formulation

### 1. BM25 Okapi Scoring
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### 2. Dense Semantic Cosine Similarity
$$\text{Score}_{\text{Dense}}(d, q) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}$$

### 3. Reciprocal Rank Fusion (RRF)
$$\text{RRF Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
Where $k=60$ is the standard smoothing constant preventing high-rank domination.

### 4. Evaluation Metrics
- **MRR@K (Mean Reciprocal Rank)**: Measures how early the first relevant passage appears.
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Measures graded relevance with logarithmic rank discounting.
- **Recall@K & Precision@K**: Measures coverage and cleanliness of top results.

---

## 🚀 Quickstart & Setup

### 1. Activate Environment
```powershell
cd C:\..\Neural-IR
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run CLI Benchmark & Ablation Study
To run the automated IR benchmark comparing all retrieval stages:
```powershell
python run_eval.py
```

### 3. Launch the Interactive Web Dashboard
To explore queries, view latency profiling, compare rankings, and inspect ground truth matches interactively:
```powershell
streamlit run app.py
```

---
