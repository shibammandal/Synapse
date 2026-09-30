import streamlit as st
import time
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.benchmark_loader import load_benchmark
from src.pipeline import HybridSearchPipeline
from src.evaluation import evaluate_retrieval_run, compute_ndcg_at_k, compute_mrr_at_k

st.set_page_config(
    page_title="SynapseIR | Hybrid Neural Search",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 8px;
    }
    .doc-card {
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #3b82f6;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .relevant-badge {
        background-color: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    .score-badge {
        background-color: #6366f1;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models & building indices...")
def get_pipeline_and_data():
    corpus, queries, qrels = load_benchmark()
    pipeline = HybridSearchPipeline(
        documents=corpus,
        dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker=True
    )
    return pipeline, corpus, queries, qrels


def main():
    st.title("⚡ SynapseIR: Hybrid Neural Search & Deep Re-Ranking")
    st.caption("BM25 Lexical + Dense Semantic (all-MiniLM-L6-v2) + Reciprocal Rank Fusion (RRF) + Cross-Encoder Re-Ranker | Real MS-MARCO Benchmark")

    pipeline, corpus, queries, qrels = get_pipeline_and_data()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    top_k = st.sidebar.slider("Top K Results", min_value=3, max_value=20, value=5)
    candidates_k = st.sidebar.slider("Candidate Pool (K)", min_value=10, max_value=50, value=30)
    fusion_method = st.sidebar.selectbox("Fusion Method", ["rrf", "linear"], format_func=lambda x: "Reciprocal Rank Fusion (RRF)" if x == "rrf" else "Linear Score Blending")
    
    rrf_k = 60
    alpha = 0.5
    if fusion_method == "rrf":
        rrf_k = st.sidebar.number_input("RRF Smoothing Parameter (k)", min_value=1, max_value=100, value=60)
    else:
        alpha = st.sidebar.slider("Dense Weight (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    app_mode = st.sidebar.radio("Mode", ["Interactive Search", "System Evaluation Benchmark"])

    if app_mode == "Interactive Search":
        st.subheader("Query Explorer")
        
        # Query selection or custom input
        query_options = ["Custom query..."] + [f"{q['id']}: {q['text']}" for q in queries]
        selected_option = st.selectbox("Choose a benchmark query or enter your own:", query_options)

        if selected_option == "Custom query...":
            query_input = st.text_input("Enter search query:", value="what organelle generates ATP energy in cells")
            active_qid = None
        else:
            active_qid = selected_option.split(":")[0]
            query_input = selected_option.split(": ", 1)[1]
            st.text_input("Active Query:", value=query_input, disabled=True)

        if query_input.strip():
            with st.spinner("Executing retrieval across all stages..."):
                res = pipeline.search(
                    query=query_input,
                    top_k=top_k,
                    candidates_k=candidates_k,
                    fusion_method=fusion_method,
                    rrf_k=rrf_k,
                    alpha=alpha,
                    use_reranker=True
                )

            # Latency Bar
            p = res["profiling"]
            col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
            col_l1.metric("BM25 Latency", f"{p['sparse_ms']:.1f} ms")
            col_l2.metric("Dense Latency", f"{p['dense_ms']:.1f} ms")
            col_l3.metric("Fusion Latency", f"{p['fusion_ms']:.2f} ms")
            col_l4.metric("Rerank Latency", f"{p['rerank_ms']:.1f} ms")
            col_l5.metric("Total Latency", f"{p['total_ms']:.1f} ms")

            st.write("---")

            # Ground truth relevance set for active query if available
            gt_relevance = qrels.get(active_qid, {}) if active_qid else {}

            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Hybrid + Re-ranker (Final)",
                "🔀 Hybrid Fusion (RRF)",
                "🧠 Dense Retrieval",
                "🔤 BM25 Sparse Retrieval"
            ])

            def render_results(results, is_reranked=False):
                if not results:
                    st.info("No documents retrieved.")
                    return
                for doc in results:
                    doc_id = doc["id"]
                    is_rel = doc_id in gt_relevance
                    rel_grade = gt_relevance.get(doc_id, 0)
                    
                    st.markdown(f"""
                    <div class="doc-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                            <strong>Rank {doc.get('final_rank', doc.get('rank'))} &mdash; <code>{doc_id}</code></strong>
                            <div>
                                {'<span class="relevant-badge">★ Relevant (Grade ' + str(rel_grade) + ')</span>' if is_rel else ''}
                                <span class="score-badge">Score: {doc.get('rerank_score', doc.get('score')):.4f}</span>
                            </div>
                        </div>
                        <div style="color: #334155; font-size: 14px; line-height: 1.5;">
                            {doc['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with tab1:
                st.markdown("### Top Candidates after Cross-Encoder Joint Re-ranking")
                render_results(res["reranked_results"], is_reranked=True)

            with tab2:
                st.markdown(f"### Top Candidates from Fusion ({fusion_method.upper()})")
                render_results(res["hybrid_results"])

            with tab3:
                st.markdown("### Top Candidates from Dense Semantic Embeddings")
                render_results(res["dense_results"])

            with tab4:
                st.markdown("### Top Candidates from BM25 Lexical Matching")
                render_results(res["sparse_results"])

    else:
        st.subheader("📊 Quantitative Benchmark Evaluation (Real MS-MARCO Benchmark)")
        st.write("Evaluate all 4 retrieval strategies across the benchmark queries against ground truth relevance judgements.")

        if st.button("🚀 Run Full Evaluation", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            runs = {
                "BM25 (Sparse)": {},
                "Dense (MiniLM)": {},
                "Hybrid (RRF)": {},
                "Hybrid + Cross-Encoder": {}
            }
            latencies = {m: [] for m in runs.keys()}

            for i, q in enumerate(queries):
                qid = q["id"]
                status_text.text(f"Evaluating query {i+1}/{len(queries)}: '{q['text']}'")
                out = pipeline.search(q["text"], top_k=10, candidates_k=candidates_k, fusion_method=fusion_method, rrf_k=rrf_k, alpha=alpha, use_reranker=True)

                runs["BM25 (Sparse)"][qid] = [doc["id"] for doc in out["sparse_results"]]
                runs["Dense (MiniLM)"][qid] = [doc["id"] for doc in out["dense_results"]]
                runs["Hybrid (RRF)"][qid] = [doc["id"] for doc in out["hybrid_results"]]
                runs["Hybrid + Cross-Encoder"][qid] = [doc["id"] for doc in out["reranked_results"]]

                latencies["BM25 (Sparse)"].append(out["profiling"]["sparse_ms"])
                latencies["Dense (MiniLM)"].append(out["profiling"]["dense_ms"])
                latencies["Hybrid (RRF)"].append(out["profiling"]["sparse_ms"] + out["profiling"]["dense_ms"] + out["profiling"]["fusion_ms"])
                latencies["Hybrid + Cross-Encoder"].append(out["profiling"]["total_ms"])

                progress_bar.progress((i + 1) / len(queries))

            status_text.text("Evaluation complete! Aggregating metrics...")

            records = []
            for method_name, run_dict in runs.items():
                m = evaluate_retrieval_run(run_dict, qrels, k_values=[1, 5, 10])
                avg_lat = sum(latencies[method_name]) / len(latencies[method_name])
                records.append({
                    "Retrieval Strategy": method_name,
                    "NDCG@1": f"{m['NDCG@1']:.4f}",
                    "NDCG@5": f"{m['NDCG@5']:.4f}",
                    "NDCG@10": f"{m['NDCG@10']:.4f}",
                    "MRR@10": f"{m['MRR@10']:.4f}",
                    "Recall@10": f"{m['Recall@10']:.4f}",
                    "Avg Latency": f"{avg_lat:.2f} ms"
                })

            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
            st.success("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
