"""
retrieve.py - Step 6 of the retrieval track. The public interface.

This is the only file Person B imports. Everything behind it - parsing,
linearization, chunking, FAISS, the cross-encoder - is an implementation
detail that can change without breaking his code.

    from retrieval.retrieve import retrieve_evidence
    evidence = retrieve_evidence("why did Greater China net sales decline",
                                 "aapl-2025")

Two stages run underneath:
    stage 1  bi-encoder over all chunks     -> 20 candidates  (recall)
    stage 2  cross-encoder over those 20    -> top 3          (precision)

The support threshold
---------------------
Cross-encoder scores are calibrated enough to be useful on their own.
Observed on real queries: a correct answer scores +7 to +9, a plausible
but wrong passage lands near 0, and an unrelated one sits at -4.

So "nothing scored above the threshold" is a real answer, not a failure.
It means the filing contains no passage explaining this anomaly - which
is precisely the case an analyst most wants flagged, and the second
condition in our detection rule: a metric moved sharply AND management
did not explain it.

MIN_SUPPORT_SCORE is a starting value. Calibrate it in eval_retrieval.py
against labelled queries before trusting it in the report.
"""

from __future__ import annotations

import re
import sys

from retrieval.indexer import Index
from retrieval.reranker import rerank

CANDIDATES = 20          # stage 1 shortlist - tuned for recall
TOP_K = 3                # stage 2 output - what the generator sees
MIN_SUPPORT_SCORE = 1.0  # necessary but not sufficient - see must_mention

# Words too common in a 10-K to anchor anything.
_STOPWORDS = {
    "the", "and", "for", "was", "were", "did", "why", "how", "what", "its",
    "this", "that", "with", "from", "have", "has", "had", "company", "during",
    "compared", "increase", "decrease", "increased", "decreased", "change",
    "changed", "year", "years", "net", "total", "millions", "billions",
}

_indexes: dict[str, Index] = {}


# Phrases management uses when accounting for a change. Required only when
# the anchor is a single term and therefore cannot discriminate on its own:
# "Revenue" matched a geographic disclosure, "Net income" matched a tax
# reconciliation fragment. Both lack any causal phrasing, while every real
# explanation in these filings carries one.
EXPLANATORY_RE = re.compile(
    r"due to|primarily|driven by|because|as a result|attributable to|"
    r"reflect(s|ing)?|resulting from|offset by|impacted by|owing to|"
    r"stemming from|caused by|as our|led by|contributed to",
    re.I,
)


def _normalise(text: str) -> str:
    """Lowercase, drop punctuation and possessives, collapse whitespace."""
    text = text.lower().replace("\u2019", "'").replace("'s", "").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _anchor_phrase(label: str) -> str | None:
    """
    The label as a contiguous phrase, when it is short enough to demand one.

    A short label made of common words cannot be matched term by term:
    "Net income" reduces to {income}, and a 10-K says "income" in "foreign
    income before tax", "other income (expense)" and "income taxes". None of
    those is net income, and all of them matched. Requiring the two words
    adjacent rules them out, while "gross margin", "Greater China" and
    "effective tax rate" all appear verbatim when management discusses them.

    Long labels keep the term-by-term test, because a filing may write
    "cash provided by operating activities" where the table says "generated".
    """
    phrase = _normalise(label)
    phrase = re.sub(r"^total ", "", phrase)
    return phrase if 1 <= len(phrase.split()) <= 3 else None


def _anchor_terms(label: str) -> set[str]:
    """Distinctive words from a metric label, lowercased."""
    words = re.findall(r"[a-z]{4,}", label.lower())
    return {w for w in words if w not in _STOPWORDS}


def _mentions(text: str, anchors: set[str]) -> bool:
    """
    True if the passage contains EVERY anchor term (prefix match for plurals).

    Requiring any one term was too weak. "Total shareholders' equity" matched
    a stock performance graph on the word "shareholder" alone, and "Operating
    income" matched "cash provided by operating activities". Both terms
    together rule those out while keeping genuine matches, because a passage
    that really is about operating income says both words.
    """
    low = text.lower()
    return all(a[:-1] in low if a.endswith("s") else a in low for a in anchors)


def _is_supporting(result: dict, threshold: float, anchors: set[str],
                   phrase: str | None = None) -> bool:
    """
    Whether one passage counts as explaining the anomaly.

    Three conditions, each rejecting a failure the others let through:
      score      the reranker thinks it is relevant
      text       a table row restates the number, it never explains it
      anchors    the metric is named - as a contiguous phrase for short
                 labels, term by term for long ones - so the passage is
                 about this metric and not a lexical lookalike

    A one-term anchor cannot do that last job - "Revenue" appears in half
    the filing - so those additionally require causal phrasing.
    """
    if result["rerank_score"] < threshold:
        return False
    if result["source_type"] != "text":
        return False
    if phrase:
        if phrase not in _normalise(result["text"]):
            return False
    elif anchors and not _mentions(result["text"], anchors):
        return False
    if len(anchors) <= 1 and not EXPLANATORY_RE.search(result["text"]):
        return False
    return True


def _get_index(filing_id: str) -> Index:
    """Cache loaded indexes: reading FAISS on every call would be wasteful."""
    if filing_id not in _indexes:
        _indexes[filing_id] = Index(filing_id)
    return _indexes[filing_id]


def retrieve_evidence(
    query: str,
    filing_id: str,
    top_k: int = TOP_K,
    min_score: float | None = None,
    must_mention: str | None = None,
) -> list[dict]:
    """
    Return the best evidence passages for a query.

    This is the contract agreed with Person B. Each item:

        {"text":        str,    the passage itself
         "source_type": str,    "text" or "table"
         "chunk_id":    str,    stable id for this chunk
         "page":        int,    printed page in the filing
         "section":     str,    e.g. "Item 7 - Management's Discussion..."
         "score":       float,  cross-encoder score, higher is better
         "supported":   bool}   passage is genuinely about the query

    must_mention is the anomaly's metric label, e.g. "Greater China" or
    "Research and development". When supplied, a passage must contain one
    of its distinctive words to count as supported, no matter how high it
    scores.

    This exists because the score alone is not trustworthy. Measured on
    real queries, a correct answer scored 1.03 while a passage about export
    restrictions scored 3.34 for a restructuring query - it had matched
    "restructure" to "restrictive". Cross-encoder scores are not comparable
    across queries, so no single cutoff separates those two cases. The
    lexical anchor does, because the label is drawn from the filing's own
    table rows and therefore provably occurs in it.

    A table row can never count as support. It restates the number the
    anomaly was computed from - "Cash generated by operating activities:
    2025: 111,482 | 2024: 118,254" - which is not an account of why the
    number moved. Tables stay in the results as citable sources; they just
    do not make an anomaly explained.

    Results below the bar are still returned, flagged supported=False, so
    the caller can tell "the filing does not explain this" apart from
    "retrieval broke".
    """
    threshold = MIN_SUPPORT_SCORE if min_score is None else min_score
    anchors = _anchor_terms(must_mention) if must_mention else set()
    phrase = _anchor_phrase(must_mention) if must_mention else None

    candidates = _get_index(filing_id).search(query, top_k=CANDIDATES)
    if not candidates:
        return []

    ranked = rerank(query, candidates, top_k=top_k)

    return [
        {
            "text": r["text"],
            "source_type": r["source_type"],
            "chunk_id": r["chunk_id"],
            "page": r["page"],
            "section": r["section"],
            "score": round(r["rerank_score"], 4),
            "supported": _is_supporting(r, threshold, anchors, phrase),
        }
        for r in ranked
    ]


def has_support(evidence: list[dict]) -> bool:
    """
    True if at least one passage cleared the support threshold.

    When this is False the anomaly is unexplained by the filing. Person B
    should surface that as its own finding rather than asking the generator
    to write an explanation from evidence that does not support one - that
    is exactly how a model ends up inventing a cause.
    """
    return any(e["supported"] for e in evidence)


def main() -> None:
    if len(sys.argv) < 3:
        print('usage: python -m retrieval.retrieve <filing_id> "your query"')
        sys.exit(1)

    filing_id, query = sys.argv[1], " ".join(sys.argv[2:])
    evidence = retrieve_evidence(query, filing_id)

    print(f'\nquery: "{query}"')
    print(f"supported: {has_support(evidence)}  "
          f"(threshold {MIN_SUPPORT_SCORE})\n")

    for i, e in enumerate(evidence, start=1):
        mark = "OK " if e["supported"] else "weak"
        tag = "TABLE" if e["source_type"] == "table" else "text "
        print(f"  {i}. [{mark}] {e['score']:>7.3f} [{tag}] "
              f"p{e['page']:<3} {e['section'][:38]}")
        print(f"     {e['text'][:160]}\n")

    if not has_support(evidence):
        print("  -> no passage in this filing explains the query.")
        print("     For an anomaly, that absence is itself the finding.")


if __name__ == "__main__":
    main()