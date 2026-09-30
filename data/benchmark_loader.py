import json
import os
from typing import Dict, List, Tuple, Any

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(DATA_DIR, "msmarco_benchmark.json")


def load_benchmark() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    if not os.path.exists(BENCHMARK_PATH):
        raise FileNotFoundError(f"Benchmark dataset not found at: {BENCHMARK_PATH}")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["corpus"], data["queries"], data["qrels"]


if __name__ == "__main__":
    corpus, queries, qrels = load_benchmark()
    print(f"Loaded: {len(corpus)} passages, {len(queries)} queries, {len(qrels)} qrels.")
