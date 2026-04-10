# FinLens: Multi-Stage Hybrid Search Engine for Financial Passage Retrieval

ECS736P/U Information Retrieval, CW2 — Group 2

## Overview

A multi-stage hybrid retrieval system evaluated on the FiQA-2018 corpus (57,638 financial passages, 648 test queries). Two competing pipelines share a cross-encoder reranker at Stage 2:

- **Pipeline A (Hybrid):** BM25 + Dense Bi-Encoder (all-MiniLM-L6-v2) → RRF Fusion → Cross-Encoder
- **Pipeline B (Query Expansion):** BM25 → Bo1 PRF → BM25 (2nd pass) → Cross-Encoder

Best configuration: **Dense + Cross-Encoder** achieves nDCG@10 = 0.380 (+50% over BM25 baseline).

## Setup

### Prerequisites

- Python 3.10+
- Java 11+ (required by PyTerrier)
- conda (recommended)

### Installation

```bash
# Create and activate environment
conda create -n ir_search python=3.10 -y
conda activate ir_search

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional, requires CUDA)
pip install faiss-gpu
```

### Data Preparation and Indexing

```bash
# Step 1: Download FiQA dataset via ir_datasets
python prepare_dataset.py

# Step 2: Build BM25 inverted index and FAISS dense vector index
python build_indexes.py
```

This creates:
- `indexes/bm25_index/` - PyTerrier inverted index
- `indexes/faiss_index/` - FAISS IndexFlatIP (384-dim, exact search)

## Usage

### Run Evaluation (all 10 configurations)

```bash
python run_experiments.py
```

Evaluates all configurations and prints a results table with Recall@100, nDCG@10, MAP, MRR@10.

### Interactive Demo

```bash
conda activate ir_search
streamlit run demo.py
```

Opens the FinLens web interface at `http://localhost:8501`. Features:
- Pipeline A / Pipeline B selection
- Compare mode (side-by-side)
- Rank journey visualisation per document
- RAG-powered AI summaries (optional, requires your own OpenAI API key entered in sidebar)

## Configurations Evaluated

| Config | Stage 1 | Stage 2 | Recall@100 | nDCG@10 |
|--------|---------|---------|-----------|---------|
| BM25 | Lexical | — | 0.559 | 0.253 |
| Dense | Dense | — | 0.706 | 0.369 |
| Hybrid (RRF) | BM25 + Dense | — | 0.706 | 0.359 |
| BM25+Bo1 | Expanded lexical | — | 0.563 | 0.244 |
| DenseBo1 | Dense feedback + Bo1 | — | 0.560 | 0.253 |
| BM25+CE | Lexical | Cross-encoder | — | 0.347 |
| Dense+CE | Dense | Cross-encoder | — | 0.380 |
| Hybrid+CE | BM25 + Dense | Cross-encoder | — | 0.370 |
| BM25+Bo1+CE | Expanded lexical | Cross-encoder | — | 0.345 |
| DenseBo1+CE | Dense feedback + Bo1 | Cross-encoder | — | 0.347 |

## Project Structure

```
IR_Search_Engine/
├── config.py              # All hyperparameters (BM25 k1/b, models, paths)
├── prepare_dataset.py     # Download and prepare FiQA dataset
├── build_indexes.py       # Build BM25 + FAISS indexes
├── run_experiments.py     # Evaluate all 10 configurations
├── demo.py                # Streamlit web interface
├── src/
│   ├── data_loader.py     # FiQA corpus/queries/qrels loading
│   ├── indexing.py        # BM25 index builder (PyTerrier)
│   ├── retrieval_bm25.py  # BM25 retriever
│   ├── retrieval_dense.py # Dense bi-encoder + FAISS retriever
│   ├── fusion.py          # Reciprocal Rank Fusion (k=60)
│   ├── query_expansion.py # Bo1 PRF + DenseFeedbackExpander
│   ├── reranker.py        # Cross-encoder reranker
│   ├── pipeline.py        # Pipeline orchestrator
│   └── evaluation.py      # pytrec_eval metrics computation
├── data/                  # FiQA corpus, queries, qrels (generated)
├── indexes/               # BM25 and FAISS indexes (generated)
└── results/               # Evaluation outputs, presentation files
```

## Models

| Component | Model | Details |
|-----------|-------|---------|
| Bi-encoder | `all-MiniLM-L6-v2` | 384-dim, 6 layers, zero-shot |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` | Zero-shot, same MiniLM family |
| BM25 | PyTerrier | k1=1.2, b=0.75 |

All neural components are used zero-shot without fine-tuning on financial data.

## Team

| Member | Responsibility |
|--------|----------------|
| Artem Zeleniuk | Hybrid retrieval (BM25 + Dense + RRF) |
| Samuel Edwards | Dense retriever, fusion |
| Rayaan Sheikh | Cross-encoder reranker |
| Hardik Mathur | Evaluation pipeline |
