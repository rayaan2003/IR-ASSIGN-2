"""Evaluate Stage 1 retrieval: BM25, Dense, and Hybrid on full FiQA test set.

This is a quick benchmark to validate that the hybrid pipeline is working
correctly.


"""

from __future__ import annotations

import sys
import time

from src.data_loader import load_all
from src.evaluation import (
    evaluate_end_to_end,
    evaluate_stage1,
    format_results_table,
)
from src.fusion import rrf_fuse
from src.retrieval_bm25 import BM25Retriever
from src.retrieval_dense import DenseRetriever


def main() -> int:
    print("=" * 70)
    print("Stage 1 Evaluation: BM25 / Dense / Hybrid")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data ...")
    corpus, queries, qrels = load_all()
    print(f"      {len(corpus):,} docs, {len(queries):,} queries, "
          f"{sum(len(d) for d in qrels.values()):,} judgements")

    # Initialise retrievers
    print("\n[2/4] Loading retrievers ...")
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    dense.load_index()
    print("      BM25 and Dense retrievers ready.")

    results_table = {}

    #  BM25 
    print("\n[3/4] Running retrieval ...")
    print("      BM25 ...", end=" ", flush=True)
    t0 = time.perf_counter()
    bm25_results = bm25.retrieve_batch(queries)
    bm25_time = time.perf_counter() - t0
    print(f"done ({bm25_time:.1f}s, {len(bm25_results):,} rows)")

    s1 = evaluate_stage1(bm25_results, qrels)
    e2e = evaluate_end_to_end(bm25_results, qrels)
    results_table["BM25"] = {**s1, **e2e}

    #  Dense 
    print("      Dense ...", end=" ", flush=True)
    t0 = time.perf_counter()
    dense_results = dense.retrieve_batch(queries)
    dense_time = time.perf_counter() - t0
    print(f"done ({dense_time:.1f}s, {len(dense_results):,} rows)")

    s1 = evaluate_stage1(dense_results, qrels)
    e2e = evaluate_end_to_end(dense_results, qrels)
    results_table["Dense"] = {**s1, **e2e}

    #  Hybrid (RRF) 
    print("      Hybrid (RRF) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    hybrid_results = rrf_fuse([bm25_results, dense_results])
    hybrid_time = time.perf_counter() - t0
    print(f"done ({hybrid_time:.1f}s, {len(hybrid_results):,} rows)")

    s1 = evaluate_stage1(hybrid_results, qrels)
    e2e = evaluate_end_to_end(hybrid_results, qrels)
    results_table["Hybrid"] = {**s1, **e2e}

    #  Results 
    print("\n[4/4] Results")
    print("=" * 70)
    print(format_results_table(results_table))
    print("=" * 70)

    # Сheck
    beir_bm25 = 0.236
    our_bm25 = results_table["BM25"].get("ndcg_cut_10", 0)
    delta = abs(our_bm25 - beir_bm25)
    if delta < 0.03:
        print(f"\nBM25 nDCG@10 = {our_bm25:.4f} (BEIR published: {beir_bm25})")
        print("PASS: within expected range.")
    else:
        print(f"\nWARNING: BM25 nDCG@10 = {our_bm25:.4f} differs from BEIR "
              f"published baseline ({beir_bm25}) by {delta:.4f}.")
        print("Check BM25 parameters and indexing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
