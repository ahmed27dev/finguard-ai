"""
indexer.py - Step 4 of the retrieval track. Stage 1 of two-stage search.

Embeds every chunk with a bi-encoder and stores the vectors in FAISS.
Given a query, returns the top ~20 candidates in milliseconds.

Why two stages at all: the bi-encoder embeds queries and chunks
*separately*, so chunk vectors are computed once and reused forever. That
makes it fast enough to scan thousands of chunks, but it never actually
compares a query against a chunk directly - it only compares two summaries
of them. That costs precision. reranker.py fixes the precision on the
small candidate set this stage hands over.

Model: all-MiniLM-L6-v2. 384 dimensions, ~80 MB, runs fine on CPU. It
truncates at 256 word-pieces, which is exactly why chunker.py caps chunks
at 1,000 characters.

Usage:
    python -m retrieval.indexer build aapl-2025
    python -m retrieval.indexer search aapl-2025 "why did operating income fall"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED = Path("data/processed")
_model = None


def get_model():
    """Load the encoder once and keep it. First call downloads ~80 MB."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"loading {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed texts as L2-normalised float32 vectors.

    Normalising matters: with unit vectors, FAISS inner product IS cosine
    similarity, so scores land in a comparable 0-1 range across queries.
    Without it, longer chunks score higher merely for being longer.
    """
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype("float32")


def build_index(filing_id: str) -> None:
    import faiss

    chunks = json.loads(
        (PROCESSED / f"{filing_id}.chunks.json").read_text(encoding="utf-8")
    )
    if not chunks:
        raise SystemExit("no chunks - run retrieval.chunker first")

    # Table chunks are terse ("Greater China (in millions): 2025: 64,377").
    # Prefixing the section gives the encoder some context to work with,
    # so a bare row of numbers is not stranded without a topic.
    texts = [
        f"{c['section']}. {c['text']}" if c["source_type"] == "table" else c["text"]
        for c in chunks
    ]

    print(f"embedding {len(texts)} chunks ...")
    vectors = embed(texts)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(PROCESSED / f"{filing_id}.faiss"))
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "source_type": c["source_type"],
            "page": c["page"],
            "section": c["section"],
            "origin_ids": c.get("origin_ids", []),
        }
        for c in chunks
    ]
    (PROCESSED / f"{filing_id}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"indexed {index.ntotal} vectors, dim {vectors.shape[1]}")
    print(f"wrote {PROCESSED / f'{filing_id}.faiss'}")


class Index:
    """A loaded index plus its metadata. Load once, query many times."""

    def __init__(self, filing_id: str):
        import faiss

        path = PROCESSED / f"{filing_id}.faiss"
        if not path.exists():
            raise SystemExit(f"no index for {filing_id} - run 'build' first")
        self.filing_id = filing_id
        self.index = faiss.read_index(str(path))
        self.meta = json.loads(
            (PROCESSED / f"{filing_id}.meta.json").read_text(encoding="utf-8")
        )

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Return the top_k candidates for a query.

        top_k is deliberately generous here. This stage is recall-oriented:
        its job is to make sure the right chunk is somewhere in the shortlist,
        not to rank it first. The reranker decides the final order.
        """
        vector = embed([query])
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            item = dict(self.meta[idx])
            item["score"] = float(score)
            item["rank"] = rank + 1
            results.append(item)
        return results


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command, filing_id = sys.argv[1], sys.argv[2]

    if command == "build":
        build_index(filing_id)

    elif command == "search":
        if len(sys.argv) < 4:
            raise SystemExit('usage: search <filing_id> "your query"')
        query = " ".join(sys.argv[3:])
        results = Index(filing_id).search(query, top_k=10)
        print(f'\nquery: "{query}"\n')
        for r in results:
            tag = "TABLE" if r["source_type"] == "table" else "text "
            print(
                f"  {r['rank']:>2}. {r['score']:.3f} [{tag}] p{r['page']:<3} "
                f"{r['section'][:34]:<34}"
            )
            print(f"      {r['text'][:150]}")
    else:
        raise SystemExit("command must be 'build' or 'search'")


if __name__ == "__main__":
    main()