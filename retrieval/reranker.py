"""
reranker.py - Step 5 of the retrieval track. Stage 2 of two-stage search.

Takes the ~20 candidates the bi-encoder shortlisted and re-scores them
properly, returning the best 3-5.

The difference from stage 1 is where the comparison happens. A bi-encoder
turns the query into one vector and each chunk into another, then measures
the angle between them - it never sees the two texts together. A
cross-encoder concatenates them:

    [CLS] why did Greater China net sales decline [SEP] Greater China net
    sales decreased during 2025 ... [SEP]

and runs attention across the pair, so every query word can attend to every
chunk word. Far more accurate, and far more expensive: one model call per
candidate instead of one per query. That is the whole reason for the funnel
- 637 chunks through the cheap stage, 20 through the expensive one.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2, ~80 MB, CPU-friendly.

Scores here are raw logits, roughly -11 to +11. They are NOT comparable to
the 0-1 cosine scores from indexer.py and must never be averaged with them.

Usage:
    python -m retrieval.reranker aapl-2025 "why did Greater China net sales decline"
"""

from __future__ import annotations

import sys

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def get_model():
    """Load the cross-encoder once. First call downloads ~80 MB."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        print(f"loading {MODEL_NAME} ...")
        _model = CrossEncoder(MODEL_NAME, max_length=512)
    return _model


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    batch_size: int = 16,
) -> list[dict]:
    """
    Re-score candidates against the query and return the best top_k.

    Each result keeps both scores: rerank_score decides the order,
    bi_score is retained so evaluation can measure what stage 2 actually
    contributed over stage 1.
    """
    if not candidates:
        return []

    pairs = [
        (
            query,
            f"{c['section']}. {c['text']}"
            if c.get("source_type") == "table"
            else c["text"],
        )
        for c in candidates
    ]

    scores = get_model().predict(pairs, batch_size=batch_size, show_progress_bar=False)

    scored = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["bi_score"] = candidate.get("score")
        item["rerank_score"] = float(score)
        scored.append(item)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
    return scored[:top_k]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    from retrieval.indexer import Index

    filing_id, query = sys.argv[1], " ".join(sys.argv[2:])

    candidates = Index(filing_id).search(query, top_k=20)
    results = rerank(query, candidates, top_k=5)

    print(f'\nquery: "{query}"')
    print(f"{len(candidates)} candidates from stage 1, reranked to {len(results)}\n")

    before = {c["chunk_id"]: c["rank"] for c in candidates}
    for r in results:
        moved = before[r["chunk_id"]] - r["rank"]
        arrow = f"up {moved}" if moved > 0 else ("same" if moved == 0 else f"down {-moved}")
        tag = "TABLE" if r["source_type"] == "table" else "text "
        print(
            f"  {r['rank']}. rerank {r['rerank_score']:>7.3f} | bi {r['bi_score']:.3f} "
            f"| was #{before[r['chunk_id']]:<2} ({arrow})"
        )
        print(f"     [{tag}] p{r['page']:<3} {r['section'][:40]}")
        print(f"     {r['text'][:150]}\n")


if __name__ == "__main__":
    main()