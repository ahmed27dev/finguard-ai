"""
inspect_parse.py — eyeball the parser output before trusting it.

Run from the repo root:
    python inspect_parse.py aapl-2025
    python inspect_parse.py aapl-2025 --table tb_0012
    python inspect_parse.py aapl-2025 --find "Greater China"
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load(filing_id: str) -> dict:
    p = Path("data/processed") / f"{filing_id}.parsed.json"
    if not p.exists():
        raise SystemExit(f"not found: {p}  (run retrieval.parser first)")
    return json.loads(p.read_text(encoding="utf-8"))


def overview(d: dict) -> None:
    blocks, tables = d["text_blocks"], d["tables"]

    print("=" * 70)
    print("TEXT BLOCKS PER SECTION")
    print("=" * 70)
    for section, n in Counter(b["section"] for b in blocks).most_common():
        print(f"  {n:>4}  {section[:60]}")

    print()
    print("=" * 70)
    print("TABLES  (id | page | rows x cols | section)")
    print("=" * 70)
    for t in tables:
        rows = t["rows"]
        cols = max((len(r) for r in rows), default=0)
        first = " / ".join(c for c in rows[0][:4] if c) if rows else ""
        print(
            f"  {t['table_id']} | p{t['page']:>3} | {len(rows):>2}x{cols:<2} | "
            f"{t['section'][:30]:<30} | {first[:40]}"
        )

    print()
    print("=" * 70)
    print("LENGTH SANITY")
    print("=" * 70)
    lens = sorted(len(b["text"]) for b in blocks)
    print(f"  shortest block : {lens[0]} chars")
    print(f"  median block   : {lens[len(lens) // 2]} chars")
    print(f"  longest block  : {lens[-1]} chars")
    unknown = sum(1 for t in tables if t["section"] == "Unknown")
    front = sum(1 for b in blocks if b["section"] == "Front Matter")
    print(f"  tables with no section : {unknown}   (want 0)")
    print(f"  blocks in front matter : {front}   (want < 30)")


def show_table(d: dict, table_id: str) -> None:
    for t in d["tables"]:
        if t["table_id"] == table_id:
            print(f"{t['table_id']}  page {t['page']}  |  {t['section']}\n")
            for r in t["rows"]:
                print("  " + " | ".join(f"{c:<18}" for c in r))
            return
    print(f"no table {table_id}")


def find(d: dict, needle: str) -> None:
    n = needle.lower()
    print("--- TEXT BLOCKS ---")
    for b in d["text_blocks"]:
        if n in b["text"].lower():
            print(f"\n[{b['block_id']}] p{b['page']} | {b['section'][:45]}")
            print(f"  {b['text'][:300]}")
    print("\n--- TABLES ---")
    for t in d["tables"]:
        if any(n in c.lower() for r in t["rows"] for c in r):
            print(f"\n[{t['table_id']}] p{t['page']} | {t['section'][:45]}")
            for r in t["rows"][:8]:
                print("  " + " | ".join(f"{c:<16}" for c in r))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data = load(sys.argv[1])
    if "--table" in sys.argv:
        show_table(data, sys.argv[sys.argv.index("--table") + 1])
    elif "--find" in sys.argv:
        find(data, sys.argv[sys.argv.index("--find") + 1])
    else:
        overview(data)