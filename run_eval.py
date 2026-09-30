import os
import sys
import time
from rich.console import Console
from rich.table import Table

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.benchmark_loader import load_benchmark
from src.pipeline import HybridSearchPipeline
from src.evaluation import evaluate_retrieval_run


def main():
    console = Console()
    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold cyan]       ⚡ SynapseIR: REAL MS-MARCO BENCHMARK EVALUATION      [/bold cyan]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")

    # 1. Load Data
    corpus, queries, qrels = load_benchmark()
    console.print(f"[green][OK][/green] Loaded benchmark: {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels sets.")

    # 2. Initialize Pipeline
    console.print("\n[yellow]Initializing Hybrid Retrieval Pipeline (BM25 + Dense + Reranker)...[/yellow]")
    pipeline = HybridSearchPipeline(
        documents=corpus,
        dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker=True
    )
    console.print("[green][OK][/green] Pipeline ready!\n")

    # 3. Run Benchmark across methods
    runs = {
        "BM25 (Sparse)": {},
        "Dense (MiniLM)": {},
        "Hybrid (RRF k=60)": {},
        "Hybrid + Cross-Encoder": {}
    }

    latencies = {m: [] for m in runs.keys()}

    console.print(f"[yellow]Evaluating {len(queries)} queries across 4 retrieval configurations...[/yellow]")
    for q in queries:
        qid = q["id"]
        qtext = q["text"]

        t0 = time.perf_counter()
        out = pipeline.search(qtext, top_k=10, candidates_k=30, use_reranker=True)

        runs["BM25 (Sparse)"][qid] = [doc["id"] for doc in out["sparse_results"]]
        runs["Dense (MiniLM)"][qid] = [doc["id"] for doc in out["dense_results"]]
        runs["Hybrid (RRF k=60)"][qid] = [doc["id"] for doc in out["hybrid_results"]]
        runs["Hybrid + Cross-Encoder"][qid] = [doc["id"] for doc in out["reranked_results"]]

        latencies["BM25 (Sparse)"].append(out["profiling"]["sparse_ms"])
        latencies["Dense (MiniLM)"].append(out["profiling"]["dense_ms"])
        latencies["Hybrid (RRF k=60)"].append(out["profiling"]["sparse_ms"] + out["profiling"]["dense_ms"] + out["profiling"]["fusion_ms"])
        latencies["Hybrid + Cross-Encoder"].append(out["profiling"]["total_ms"])

    # 4. Compute Metrics
    results_table = Table(title="[bold green]⚡ SynapseIR Benchmark Results (Real MS-MARCO Dataset)[/bold green]", show_header=True, header_style="bold magenta")
    results_table.add_column("Retrieval Strategy", style="cyan", width=26)
    results_table.add_column("NDCG@1", justify="right")
    results_table.add_column("NDCG@5", justify="right")
    results_table.add_column("NDCG@10", justify="right")
    results_table.add_column("MRR@10", justify="right")
    results_table.add_column("Recall@10", justify="right")
    results_table.add_column("Avg Latency (ms)", justify="right")

    for method_name, run_dict in runs.items():
        metrics = evaluate_retrieval_run(run_dict, qrels, k_values=[1, 5, 10])
        avg_lat = sum(latencies[method_name]) / len(latencies[method_name])
        
        results_table.add_row(
            method_name,
            f"{metrics['NDCG@1']:.4f}",
            f"{metrics['NDCG@5']:.4f}",
            f"{metrics['NDCG@10']:.4f}",
            f"{metrics['MRR@10']:.4f}",
            f"{metrics['Recall@10']:.4f}",
            f"{avg_lat:.2f} ms"
        )

    console.print("\n")
    console.print(results_table)
    console.print("\n[bold cyan]Key Insights:[/bold cyan]")
    console.print(" - [bold]BM25[/bold] provides fast keyword filtering but struggles with vocabulary mismatch.")
    console.print(" - [bold]Dense Retrieval[/bold] captures semantic context and synonyms.")
    console.print(" - [bold]Hybrid RRF[/bold] unites keyword precision and semantic recall without hyperparameter tuning.")
    console.print(" - [bold]Cross-Encoder Re-ranker[/bold] achieves top NDCG & MRR via deep joint self-attention.\n")


if __name__ == "__main__":
    main()
