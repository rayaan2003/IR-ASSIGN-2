# IR Search Engine — ECS736P/U CW2

Multi-stage hybrid retrieval pipeline over the FiQA financial passage corpus.

## Architecture

Two competing pipelines that share a cross-encoder reranker:

- **Pipeline A (Hybrid):** BM25 + Dense Bi-Encoder → Reciprocal Rank Fusion → Cross-Encoder
- **Pipeline B (Query Expansion):** BM25 → Bo1 PRF → BM25 (2nd pass) → Cross-Encoder

## Setup



## Project Layout

```
IR_Search_Engine/
├── config.py              # All hyperparameters
├── prepare_dataset.py     # Step 1: dataset preparation
├── build_indexes.py       # Step 2: index construction
├── run_experiments.py     # Step 3: full evaluation
├── demo.py                # Step 4: Streamlit UI
├── src/
│   ├── data_loader.py     # FiQA loading utilities
│   ├── indexing.py        # BM25 index builder
│   ├── retrieval_bm25.py  # BM25 retriever
│   ├── retrieval_dense.py # Dense bi-encoder retriever
│   ├── fusion.py          # Reciprocal Rank Fusion
│   ├── query_expansion.py # Bo1 PRF
│   ├── reranker.py        # Cross-encoder reranker
│   ├── pipeline.py        # Pipeline orchestrator
│   └── evaluation.py      # pytrec_eval metrics
├── data/                  # FiQA corpus, queries, qrels
├── indexes/               # BM25 and FAISS indexes
└── results/               # Evaluation outputs
```

## Team

| Member | Responsibility |
|---|---|
| Artem Zeleniuk | Hybrid retrieval (BM25 + Dense + RRF) |
| Samuel Edwards | Dense retriever support, fusion |
| Rayaan Sheikh | Cross-encoder reranker |
| Hardik Mathur | Evaluation pipeline |
