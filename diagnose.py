"""
diagnose.py - show what a filing actually looks like, before changing any code.

Three parsing assumptions were taken from a single Apple 10-K and all three
break on other filers. Rather than guess at replacements, this dumps the raw
evidence for each one:

  1. page markers   - what do footers look like here, if they exist at all
  2. table headers  - are year columns bare years, or dates, or "Fiscal 2026"
  3. scale text     - where does "(in millions)" live: prose, or inside the table

Usage:
    python diagnose.py msft-2026
    python diagnose.py nvda-2026 --tables 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup

from retrieval.parser import _clean, _table_to_rows, _own_text, BLOCK_TAGS

FILINGS = Path("data/filings")


def load(filing_id: str) -> BeautifulSoup:
    path = FILINGS / f"{filing_id}.htm"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return soup


def show_page_markers(soup: BeautifulSoup) -> None:
    """
    Short text lines that end in a number are the usual page-footer shape.
    Print the most common patterns so we can see what this filer uses.
    """
    print("=" * 74)
    print("1. CANDIDATE PAGE MARKERS  (short lines ending in a number)")
    print("=" * 74)

    seen: dict[str, int] = {}
    examples: dict[str, str] = {}
    for el in soup.find_all(True):
        if el.name not in BLOCK_TAGS:
            continue
        text = _clean(_own_text(el))
        if not text or len(text) > 70:
            continue
        if not re.search(r"\d\s*$", text):
            continue
        # Generalise: digits -> #, so "... 10-K | 21" and "| 22" collapse together
        shape = re.sub(r"\d+", "#", text)
        seen[shape] = seen.get(shape, 0) + 1
        examples.setdefault(shape, text)

    ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:12]
    if not ranked:
        print("  none found - this filing may have no textual page footers")
    for shape, count in ranked:
        print(f"  {count:>4}x  {shape[:60]:<60}")
        print(f"        e.g. {examples[shape][:60]}")

    print("\n  page-break CSS markers:",
          len(soup.select('[style*="page-break"], [style*="break-before"], [style*="break-after"]')))
    print("  <hr> elements:", len(soup.find_all("hr")))


def show_table_headers(soup: BeautifulSoup, limit: int) -> None:
    """First two rows of the largest tables - that is where year labels live."""
    print("\n" + "=" * 74)
    print("2. TABLE HEADERS  (largest tables, first 2 rows each)")
    print("=" * 74)

    tables = []
    for t in soup.find_all("table"):
        rows = _table_to_rows(t)
        if len(rows) >= 3:
            tables.append(rows)
    tables.sort(key=len, reverse=True)

    for i, rows in enumerate(tables[:limit]):
        print(f"\n  --- table {i} ({len(rows)} rows) ---")
        for row in rows[:2]:
            print("    " + " | ".join(f"{c[:20]:<20}" for c in row[:7]))
        print("    data: " + " | ".join(f"{c[:20]:<20}" for c in rows[2][:7]))


def show_scale_text(soup: BeautifulSoup) -> None:
    """Where do the units appear: in prose, or inside table markup?"""
    print("\n" + "=" * 74)
    print("3. SCALE / UNIT TEXT")
    print("=" * 74)

    pattern = re.compile(r"in\s+(thousands|millions|billions)", re.I)

    in_prose, in_tables = [], []
    for el in soup.find_all(True):
        if el.name not in BLOCK_TAGS:
            continue
        text = _clean(_own_text(el))
        if text and pattern.search(text):
            inside_table = el.find_parent("table") is not None
            (in_tables if inside_table else in_prose).append(text[:90])

    print(f"\n  in prose  ({len(in_prose)} hits):")
    for t in in_prose[:6]:
        print(f"    {t}")
    print(f"\n  inside tables ({len(in_tables)} hits):")
    for t in in_tables[:6]:
        print(f"    {t}")

    if in_tables and not in_prose:
        print("\n  -> units live inside the table markup, not the surrounding prose.")
        print("     detect_scale() only reads text_blocks, so it finds nothing.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filing_id")
    ap.add_argument("--tables", type=int, default=5)
    args = ap.parse_args()

    soup = load(args.filing_id)
    print(f"\n### {args.filing_id} ###\n")
    show_page_markers(soup)
    show_table_headers(soup, args.tables)
    show_scale_text(soup)


if __name__ == "__main__":
    main()