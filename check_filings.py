"""
check_filings.py - does the pipeline work on filings it has never seen?

Everything in retrieval/ was built against one Apple 10-K. Several
decisions are quietly Apple-shaped, and the riskiest is page detection:

    PAGE_FOOTER_RE matches "Apple Inc. | 2025 Form 10-K | 21"

If another company formats its footer differently, no match ever fires,
the page counter never advances, and every chunk is silently labelled
page 1. Nothing errors. The demo just cites page 1 for everything.

This script runs parse -> linearize -> chunk over several filings and
prints the signals that would expose that kind of failure.

Usage:
    python check_filings.py aapl-2025 msft-2025 nvda-2025
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrieval.parser import parse_filing
from retrieval.linearizer import linearize_filing
from retrieval.chunker import build_chunks

FILINGS = Path("data/filings")


def check(filing_id: str) -> dict:
    path = FILINGS / f"{filing_id}.htm"
    if not path.exists():
        return {"filing_id": filing_id, "error": f"missing {path}"}

    try:
        parsed = parse_filing(path)
        linear = linearize_filing(parsed)
        chunks = build_chunks(parsed, linear)
    except Exception as exc:  # noqa: BLE001 - we want the reason, not a traceback
        return {"filing_id": filing_id, "error": f"{type(exc).__name__}: {exc}"}

    pages = [b["page"] for b in parsed["text_blocks"]] or [0]
    layouts: dict[str, int] = {}
    for r in linear:
        layouts[r["layout"]] = layouts.get(r["layout"], 0) + 1

    return {
        "filing_id": filing_id,
        "text_blocks": len(parsed["text_blocks"]),
        "tables": len(parsed["tables"]),
        "sections": len(parsed["_sections_found"]),
        "page_min": min(pages),
        "page_max": max(pages),
        "distinct_pages": len(set(pages)),
        "front_matter": sum(
            1 for b in parsed["text_blocks"] if b["section"] == "Front Matter"
        ),
        "unknown_tables": sum(1 for t in parsed["tables"] if t["section"] == "Unknown"),
        "page_method": parsed.get("_page_method", "?"),
        "linear_rows": len(linear),
        "merged_series": layouts.get("merged_series", 0),
        "stacked": layouts.get("stacked", 0),
        "with_scale": sum(1 for r in linear if r["scale"]),
        "chunks": len(chunks),
    }


def verdict(r: dict) -> list[str]:
    """Flag the specific failures that would not raise an exception."""
    problems = []
    if r["distinct_pages"] <= 2:
        problems.append(f"PAGE DETECTION FAILED (method: {r.get('page_method')})")
    if r["sections"] < 10:
        problems.append(f"only {r['sections']} sections - Item heading regex missed")
    if r["tables"] < 10:
        problems.append(f"only {r['tables']} tables - numeric filter too strict")
    if r["merged_series"] == 0 and r.get("stacked", 0) > 3:
        problems.append("stacked tables found but none merged into series")
    if r["with_scale"] == 0:
        problems.append("no scale detected - every figure is unitless")
    if r["front_matter"] > 60:
        problems.append(f"{r['front_matter']} blocks in front matter - body start missed")
    return problems


def main() -> None:
    ids = sys.argv[1:]
    if not ids:
        print(__doc__)
        sys.exit(1)

    results = []
    for filing_id in ids:
        print(f"processing {filing_id} ...")
        results.append(check(filing_id))

    print("\n" + "=" * 78)
    header = (f"{'filing':<14}{'blocks':>7}{'tables':>7}{'sects':>6}"
              f"{'pages':>12}{'linear':>7}{'merged':>7}{'chunks':>7}")
    print(header)
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['filing_id']:<14} ERROR: {r['error']}")
            continue
        pages = f"{r['page_min']}-{r['page_max']} ({r['distinct_pages']})"
        print(f"{r['filing_id']:<14}{r['text_blocks']:>7}{r['tables']:>7}"
              f"{r['sections']:>6}{pages:>12}"
              f"{r['linear_rows']:>7}{r['merged_series']:>7}{r['chunks']:>7}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for r in results:
        if "error" in r:
            print(f"\n{r['filing_id']}: could not process")
            continue
        problems = verdict(r)
        if problems:
            print(f"\n{r['filing_id']}:")
            for p in problems:
                print(f"   {p}")
        else:
            print(f"\n{r['filing_id']}: looks healthy")


if __name__ == "__main__":
    main()