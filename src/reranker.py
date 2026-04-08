"""Cross-encoder reranker (Stage 2).

A cross-encoder takes a *pair* ``(query, document)`` as a single concatenated
input and produces a single relevance score. Because it attends jointly to
both the query and the document tokens it is far more accurate than a
bi-encoder, but also far slower -- O(|candidates|) forward passes per query.

That is why the reranker sits *after* Stage 1: Stage 1 narrows the corpus
from ~57 k documents down to ``TOP_K_STAGE1 = 100`` candidates; the
cross-encoder then scores only those 100 pairs.

Model
-----
``cross-encoder/ms-marco-MiniLM-L-6-v2`` is a 6-layer MiniLM fine-tuned on
the MS MARCO passage-ranking task (Nguyen et al., 2016).  It outputs a raw
logit (not a probability) where higher == more relevant. The model is
downloaded automatically from the HuggingFace Hub on first use.

Interface contract
------------------
Both Pipeline A (BM25 + Dense + RRF → rerank) and Pipeline B
(BM25 → Bo1 → BM25 → rerank) feed the reranker with the *same* DataFrame
schema: ``[qid, docno, score, rank]``.  The reranker returns a DataFrame with
the same schema, re-scored and re-ranked to ``TOP_K_FINAL`` results.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sentence_transformers import CrossEncoder

from config import (
    CE_BATCH_SIZE,
    CE_MAX_LENGTH,
    CROSS_ENCODER_MODEL,
    TOP_K_FINAL,
    TOP_K_STAGE1,
)


class CrossEncoderReranker:
    """Stage-2 reranker based on a cross-encoder.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the MiniLM cross-encoder
        fine-tuned on MS MARCO.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    max_length:
        Maximum token length fed to the transformer.  Pairs that exceed this
        are silently truncated -- 512 is the limit for most BERT-family models.
    batch_size:
        Number of (query, doc) pairs scored per forward pass.  Larger values
        are faster on GPU but increase VRAM usage.
    """

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        device: str | None = None,
        max_length: int = CE_MAX_LENGTH,
        batch_size: int = CE_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"[reranker] Loading cross-encoder '{model_name}' ...")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        print("[reranker] Cross-encoder ready.")

    # ------------------------------------------------------------------
    # Single-query reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        corpus: Dict[str, str],
        top_k: int = TOP_K_FINAL,
    ) -> List[Dict]:
        """Rerank a list of candidate documents for a single query.

        Parameters
        ----------
        query:
            Free-text query string.
        candidates:
            List of dicts, each with at least ``{"docno": str, ...}``.
            Typically the output of a Stage-1 retriever's ``.retrieve()``
            method, which produces ``{"docno", "score", "rank"}``.
        corpus:
            Mapping from ``docno`` to raw document text.  Used to build
            the (query, doc) pairs the cross-encoder scores.
        top_k:
            Number of results to return after reranking.

        Returns
        -------
        List of dicts ``{"docno", "score", "rank"}`` sorted by descending
        cross-encoder score, truncated to ``top_k``.
        """
        if not candidates:
            return []

        # Build (query, document_text) pairs; skip any docno missing from corpus.
        pairs: List[tuple[str, str]] = []
        valid_candidates: List[Dict] = []
        for cand in candidates:
            docno = cand["docno"]
            text = corpus.get(docno)
            if text is None:
                continue
            pairs.append((query, text))
            valid_candidates.append(cand)

        if not pairs:
            return []

        # Score all pairs in one batched call.
        ce_scores: List[float] = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        ).tolist()

        # Attach scores back to candidates and sort descending.
        scored = [
            {"docno": cand["docno"], "score": float(score)}
            for cand, score in zip(valid_candidates, ce_scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Re-assign ranks (0-indexed, matching the rest of the pipeline).
        return [
            {"docno": item["docno"], "score": item["score"], "rank": rank}
            for rank, item in enumerate(scored[:top_k])
        ]

    # ------------------------------------------------------------------
    # Batch reranking (many queries at once)
    # ------------------------------------------------------------------

    def rerank_batch(
        self,
        queries: Dict[str, str],
        candidates_df: pd.DataFrame,
        corpus: Dict[str, str],
        top_k: int = TOP_K_FINAL,
    ) -> pd.DataFrame:
        """Rerank Stage-1 candidates for many queries at once.

        This method iterates over each query individually.  The cross-encoder
        cannot be batched *across* queries (each query has a different text),
        but within a single query all ``(query, doc)`` pairs are scored in one
        batched forward pass of size ``CE_BATCH_SIZE``.

        Parameters
        ----------
        queries:
            ``{qid: query_text}`` mapping.
        candidates_df:
            PyTerrier-style DataFrame with columns ``[qid, docno, score, rank]``.
            Typically contains ``TOP_K_STAGE1 = 100`` rows per query.
        corpus:
            ``{docno: text}`` mapping for the full corpus.
        top_k:
            Number of results to retain per query after reranking.

        Returns
        -------
        pd.DataFrame
            Columns ``[qid, docno, score, rank]``, one row per
            (query, top-k document) pair, sorted by ``(qid, rank)``.
        """
        if candidates_df.empty:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

        all_rows: List[Dict] = []

        for qid, query_text in queries.items():
            # Filter candidates for this query.
            qdf = candidates_df[candidates_df["qid"] == qid]
            if qdf.empty:
                continue

            candidates = qdf[["docno", "score", "rank"]].to_dict("records")

            reranked = self.rerank(
                query=query_text,
                candidates=candidates,
                corpus=corpus,
                top_k=top_k,
            )

            for row in reranked:
                all_rows.append(
                    {
                        "qid": qid,
                        "docno": row["docno"],
                        "score": row["score"],
                        "rank": row["rank"],
                    }
                )

        if not all_rows:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

        result = pd.DataFrame(all_rows)
        result = result.sort_values(["qid", "rank"]).reset_index(drop=True)
        return result
