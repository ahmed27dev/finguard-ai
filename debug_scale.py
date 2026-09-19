"""
debug_scale.py - show what the parser sees just before a table.

Three attempts at scale detection have failed, each on a different wrong
assumption about where units are declared. This prints the actual walk
order around a chosen table so the next attempt is based on the document
rather than a guess.

Usage:
    python debug_scale.py aapl-2025 "Operating income"
"""

from __future__ import annotations

import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from retrieval.parser import _clean, _own_text, BLOCK_TAGS, SCALE_DECL_RE

FILINGS = Path("data/filings")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    filing_id, needle = sys.argv[1], sys.argv[2]
    path = FILINGS / f"{filing_id}.htm"
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # Find the first table containing the needle.
    target = None
    for table in soup.find_all("table"):
        if needle.lower() in table.get_text(" ").lower():
            target = table
            break
    if target is None:
        print(f"no table containing {needle!r}")
        return

    # Walk in document order, keeping the last 25 text-bearing elements
    # seen before we reach that table.
    history: list[tuple[str, str, str]] = []
    for el in soup.find_all(True):
        if el is target:
            break
        if el.name not in BLOCK_TAGS:
            continue
        own = _clean(_own_text(el))
        full = _clean(el.get_text(" "))
        if not own and not full:
            continue
        history.append((el.name, own, full))
        history = history[-25:]

    print(f"\n=== last 25 elements before the {needle!r} table ===")
    print("own  = text the parser actually collects (direct + inline tags)")
    print("full = everything under this element, including nested blocks\n")
    for i, (name, own, full) in enumerate(history):
        marker = " <== UNITS" if (own and SCALE_DECL_RE.search(own)) else (
            " <== units only in full text" if SCALE_DECL_RE.search(full) else "")
        print(f"[-{len(history) - i:>2}] <{name}>{marker}")
        print(f"      own : {own[:110] if own else '(none)'}")
        if full != own:
            print(f"      full: {full[:110]}")

    print("\n=== units inside the table's own first rows ===")
    for tr in target.find_all("tr")[:5]:
        cells = [_clean(td.get_text(' ')) for td in tr.find_all(['td', 'th'])]
        line = " | ".join(c for c in cells if c)
        if line:
            hit = " <== UNITS" if SCALE_DECL_RE.search(line) else ""
            print(f"  {line[:110]}{hit}")


if __name__ == "__main__":
    main()