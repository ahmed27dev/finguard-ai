"""
pipeline.py - run the whole retrieval chain for a filing.

    parser -> linearizer -> chunker -> indexer

Each stage reads the previous stage's JSON off disk, so running one stage
without re-running an earlier one silently processes stale data. That has
already cost us once: scale hints are written during parsing, and starting
from the linearizer produced a filing with no units at all and no error to
say why.

This runs the stages in order and skips any whose output is already newer
than its input. Use --force after changing parser or linearizer code.

    python pipeline.py aapl-2025
    python pipeline.py aapl-2025 msft-2026 nvda-2026 --force
    python pipeline.py --all

Person B's stages (metric extraction, anomaly detection, generation,
verification) plug in after build_retrieval_artifacts() in week 6 - see
run_filing() at the bottom.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

FILINGS = Path("data/filings")
PROCESSED = Path("data/processed")


def _is_stale(output: Path, source: Path) -> bool:
    """True if output is missing or older than the thing it was built from."""
    if not output.exists():
        return True
    return output.stat().st_mtime < source.stat().st_mtime


def _code_mtime(*modules: str) -> float:
    """
    Newest modification time across the given source files.

    Editing parser.py invalidates every downstream artifact just as surely
    as replacing the filing does, so the code counts as an input.
    """
    times = []
    for name in modules:
        path = Path("retrieval") / f"{name}.py"
        if path.exists():
            times.append(path.stat().st_mtime)
    return max(times) if times else 0.0


def build_retrieval_artifacts(filing_id: str, force: bool = False) -> dict:
    """
    Bring every retrieval artifact for one filing up to date.

    Returns a summary of what ran and what each stage produced.
    """
    from retrieval.parser import parse_filing
    from retrieval.linearizer import linearize_filing
    from retrieval.chunker import build_chunks
    from retrieval.indexer import build_index

    source = FILINGS / f"{filing_id}.htm"
    if not source.exists():
        raise SystemExit(
            f"missing {source}\n"
            f"  python -m retrieval.fetch_filing <TICKER>"
        )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    parsed_p = PROCESSED / f"{filing_id}.parsed.json"
    linear_p = PROCESSED / f"{filing_id}.linear.json"
    chunks_p = PROCESSED / f"{filing_id}.chunks.json"
    index_p = PROCESSED / f"{filing_id}.faiss"

    summary: dict = {"filing_id": filing_id, "ran": [], "skipped": []}
    start = time.time()

    # Stage 1 - parse. Invalidated by a new filing or edited parser code.
    code_t = _code_mtime("parser")
    stale = force or _is_stale(parsed_p, source) or (
        parsed_p.exists() and parsed_p.stat().st_mtime < code_t)
    if stale:
        parsed = parse_filing(source)
        parsed_p.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        summary["ran"].append("parse")
    else:
        parsed = json.loads(parsed_p.read_text(encoding="utf-8"))
        summary["skipped"].append("parse")

    summary["text_blocks"] = len(parsed["text_blocks"])
    summary["tables"] = len(parsed["tables"])
    summary["page_method"] = parsed.get("_page_method", "?")
    pages = [b["page"] for b in parsed["text_blocks"]] or [0]
    summary["pages"] = f"{min(pages)}-{max(pages)}"

    # Stage 2 - linearize.
    code_t = _code_mtime("linearizer")
    stale = force or "parse" in summary["ran"] or _is_stale(linear_p, parsed_p) or (
        linear_p.exists() and linear_p.stat().st_mtime < code_t)
    if stale:
        linear = linearize_filing(parsed)
        linear_p.write_text(json.dumps(linear, indent=2), encoding="utf-8")
        summary["ran"].append("linearize")
    else:
        linear = json.loads(linear_p.read_text(encoding="utf-8"))
        summary["skipped"].append("linearize")

    summary["linear_rows"] = len(linear)
    summary["unscaled"] = sum(1 for r in linear if not r.get("scale"))

    # Stage 3 - chunk.
    code_t = _code_mtime("chunker")
    stale = force or "linearize" in summary["ran"] or _is_stale(chunks_p, linear_p) or (
        chunks_p.exists() and chunks_p.stat().st_mtime < code_t)
    if stale:
        chunks = build_chunks(parsed, linear)
        chunks_p.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        summary["ran"].append("chunk")
    else:
        chunks = json.loads(chunks_p.read_text(encoding="utf-8"))
        summary["skipped"].append("chunk")

    summary["chunks"] = len(chunks)

    # Stage 4 - index. Slowest stage, so only rebuild when chunks changed.
    stale = force or "chunk" in summary["ran"] or _is_stale(index_p, chunks_p)
    if stale:
        build_index(filing_id)
        summary["ran"].append("index")
    else:
        summary["skipped"].append("index")

    summary["seconds"] = round(time.time() - start, 1)
    return summary


def investigate(filing_id: str, force: bool = False) -> list[dict]:
    """
    Detect anomalies and look for an explanation of each one in the filing.

    This is the join between the two halves: arithmetic decides what is
    worth flagging, then retrieval goes looking for management's account of
    it. Each anomaly comes back with its evidence and a `supported` flag.

    `supported` being False is a result, not a failure. It means the filing
    contains no passage explaining a change our arithmetic says is real -
    which is exactly the case an analyst most wants surfaced, and the
    second half of the detection rule: the metric moved sharply AND
    management did not explain it.

    Generation must not run on unsupported evidence. Asking a model why
    something happened while handing it passages that do not say is how it
    invents a cause.
    """
    from flagging.metric_extractor import extract_metrics
    from flagging.anomaly_detector import detect_anomalies
    from retrieval.retrieve import retrieve_evidence, has_support

    build_retrieval_artifacts(filing_id, force)

    linear = json.loads(
        (PROCESSED / f"{filing_id}.linear.json").read_text(encoding="utf-8"))
    metrics = extract_metrics(linear)
    anomalies = detect_anomalies(metrics)

    for a in anomalies:
        evidence = retrieve_evidence(
            a["query"], filing_id, must_mention=a["label"])
        a["evidence"] = evidence
        a["supported"] = has_support(evidence)

    out = PROCESSED / f"{filing_id}.investigated.json"
    out.write_text(json.dumps(anomalies, indent=2), encoding="utf-8")
    return anomalies


def run_filing(filing_id: str, force: bool = False) -> dict:
    """Retrieval artifacts only. Use investigate() for the full chain."""
    return build_retrieval_artifacts(filing_id, force)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filings", nargs="*", help="filing ids, e.g. aapl-2025")
    ap.add_argument("--all", action="store_true",
                    help="every .htm in data/filings")
    ap.add_argument("--force", action="store_true",
                    help="rebuild every stage regardless of timestamps")
    ap.add_argument("--investigate", action="store_true",
                    help="also detect anomalies and retrieve evidence for each")
    args = ap.parse_args()

    ids = args.filings
    if args.all:
        ids = sorted(p.stem for p in FILINGS.glob("*.htm"))
    if not ids:
        print(__doc__)
        sys.exit(1)

    if args.investigate:
        for filing_id in ids:
            print(f"\n{'=' * 78}\n{filing_id}\n{'=' * 78}")
            anomalies = investigate(filing_id, args.force)
            supported = sum(1 for a in anomalies if a["supported"])
            print(f"{len(anomalies)} anomalies, {supported} with supporting "
                  f"evidence, {len(anomalies) - supported} unexplained\n")
            for a in anomalies:
                mark = "EXPLAINED  " if a["supported"] else "UNEXPLAINED"
                print(f"[{mark}] {a['severity']:.2f} {a['rule']}")
                print(f"  {a['summary']}")
                print(f"  query: {a['query']}")
                for e in a["evidence"][:2]:
                    flag = "OK  " if e["supported"] else "weak"
                    print(f"    [{flag}] {e['score']:>6.2f} p{e['page']:<4} "
                          f"{e['text'][:105]}")
                print()
        return

    results = []
    for filing_id in ids:
        print(f"\n--- {filing_id} ---")
        summary = build_retrieval_artifacts(filing_id, args.force)
        ran = ", ".join(summary["ran"]) or "nothing"
        skipped = ", ".join(summary["skipped"]) or "nothing"
        print(f"  ran     : {ran}")
        print(f"  skipped : {skipped}  (already current)")
        results.append(summary)

    print("\n" + "=" * 76)
    header = (f"{'filing':<14}{'blocks':>7}{'tables':>7}{'linear':>7}"
              f"{'unscaled':>9}{'chunks':>7}{'pages':>10}{'secs':>7}")
    print(header)
    print("-" * 76)
    for r in results:
        print(f"{r['filing_id']:<14}{r['text_blocks']:>7}{r['tables']:>7}"
              f"{r['linear_rows']:>7}{r['unscaled']:>9}{r['chunks']:>7}"
              f"{r['pages']:>10}{r['seconds']:>7}")

    for r in results:
        if r["page_method"].startswith("NONE"):
            print(f"\n  {r['filing_id']}: no page structure found - "
                  "citations for this filing cannot name a page.")


if __name__ == "__main__":
    main()