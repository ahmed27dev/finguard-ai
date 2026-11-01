"""
parser.py — Step 1 of the retrieval track.

Reads a 10-K HTML filing and produces the structure defined in our
interface contract: text blocks and tables, each labelled with the
page number and the Item section it came from.

Everything downstream (chunking, indexing, citations, and Person B's
metric extraction) depends on this output being correct.

Usage:
    python -m retrieval.parser data/filings/aapl-2025.htm
"""

from __future__ import annotations

import json
import re
import warnings
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Page footers look like:  "Apple Inc. | 2025 Form 10-K | 21"
# The trailing integer is the printed page number. This is the most reliable
# page signal in an SEC HTML filing — far better than chasing CSS page breaks.
PAGE_FOOTER_RE = re.compile(r"\|\s*\d{4}\s+Form\s+10-K\s*\|\s*(\d{1,3})\s*$", re.I)

# Section headings look like:  "Item 7.    Management's Discussion and Analysis"
# Item numbers can carry a letter suffix: 1A, 1B, 1C, 7A, 9A, 9B, 9C.
ITEM_HEADING_RE = re.compile(
    r"^item\s+(\d{1,2}[A-C]?)\s*[.:\u2014-]?\s*(.{0,120})$", re.I
)

# Units are declared immediately before the table they describe, either in
# a caption ("The following table shows ... (dollars in millions):") or in a
# statement title ("CONSOLIDATED STATEMENTS OF OPERATIONS (In millions...)").
# Page proximity is a poor proxy for that: it picks up unrelated paragraphs
# and misses nothing that document order would not catch.
SCALE_DECL_RE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)
SCALE_LOOKBACK = 3  # blocks


# Filers that print no footer line still print the page number on its own,
# as a bare block containing nothing but digits. Counting page-break elements
# gives a correct sequence but starts at the cover page, so it runs ahead of
# the printed numbering by however many pages precede page 1. These marks let
# that offset be measured and removed instead of guessed.
BARE_PAGE_RE = re.compile(r"\d{1,3}")
PAGE_MARK_MIN = 5       # marks needed before an offset is trusted
PAGE_MARK_WINDOW = 30   # a mark further than this from the count is not a page number


# A heading line that is really a table-of-contents row, e.g. "Item 1. Business 1"
TOC_TAIL_RE = re.compile(r"\s\d{1,3}$")

# Tags whose text we treat as a candidate block.
BLOCK_TAGS = {"p", "div", "span", "td", "th", "li", "h1", "h2", "h3", "h4"}

# Inline XBRL wraps every number appearing in prose:
#     was $<ix:nonFraction>17.6</ix:nonFraction> billion
# These are child tags, not text nodes, so collecting only direct strings
# silently drops every figure from the narrative. Evidence sentences would
# arrive with their numbers missing, which the faithfulness check depends on.
INLINE_TAGS = {
    "ix:nonfraction", "ix:nonnumeric", "b", "i", "em", "strong",
    "u", "sub", "sup", "a", "font", "small",
}


def _own_text(el: Tag) -> str:
    """
    Text belonging to this element: its direct strings plus any inline
    formatting or XBRL tags nested in it. Nested block tags are excluded so
    the same paragraph is not recorded once per wrapper div.
    """
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name.lower() in INLINE_TAGS:
            parts.append(child.get_text(" "))
    return "".join(parts)


def _page_break_count(soup: BeautifulSoup) -> int:
    """How many page-break elements this filing uses, if any."""
    return len(soup.find_all("hr"))


def _is_page_break(el: Tag) -> bool:
    """
    True if this element marks a page boundary.

    Most filers emit <hr> for page breaks; some style a div instead. Apple
    also prints a text footer, but that is an Apple habit - Microsoft and
    Nvidia print no footer at all, so <hr> is the general mechanism and the
    footer regex is the special case.
    """
    if el.name == "hr":
        return True
    style = (el.get("style") or "").lower()
    return "page-break-before" in style or "page-break-after" in style


def _clean(text: str) -> str:
    """Normalise whitespace and the unicode SEC filings are full of."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Section tracking
# ---------------------------------------------------------------------------

class SectionTracker:
    """
    Watches text blocks go by and remembers which Item we are currently inside.

    The hard part is the table of contents at the front of every filing: it
    contains every Item heading in order, which would otherwise make us think
    we had walked the whole document in the first two pages. We skip headings
    until we see the "PART I" marker that begins the real body.
    """

    def __init__(self) -> None:
        self.current = "Front Matter"
        self._body_started = False
        self._seen: list[str] = []

    def update(self, text: str) -> None:
        if not self._body_started:
            # The real body opens with a standalone "PART I" line.
            if re.fullmatch(r"part\s+i", text, re.I):
                self._body_started = True
            return

        if len(text) > 140:
            return  # too long to be a heading

        m = ITEM_HEADING_RE.match(text)
        if not m:
            return

        number, title = m.group(1), _clean(m.group(2))
        # A TOC row ends in a page number; a real heading does not.
        if TOC_TAIL_RE.search(title):
            title = TOC_TAIL_RE.sub("", title).strip()

        label = f"Item {number.upper()}"
        if title:
            label = f"{label} - {title}"
        self.current = label
        self._seen.append(label)

    @property
    def sections_found(self) -> list[str]:
        return self._seen


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = {"$", "\u20ac", "\u00a3", "\u00a5", "\u20b9"}


def _norm_cell(text: str) -> str:
    """Clean a single cell and tidy the spacing SEC puts inside parentheses."""
    text = _clean(text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def _collapse_spacers(cells: list[str]) -> list[str]:
    """
    SEC HTML splits every figure across several cells and pads with blanks:

        'Americas' | '$' | '178,353' | '' | '' | '7' | '%' | '' | '$' | ...

    We rejoin currency symbols with the number that follows and percent
    signs with the number before, then drop the empty padding. This turns
    an 18-column mess into the 6 real columns a human sees.
    """
    out: list = []
    for c in cells:
        if not c:
            continue
        if c in CURRENCY_SYMBOLS:
            out.append(["PENDING", c])          # waiting for its number
            continue
        if c == "%" and out:
            prev = out[-1]
            if isinstance(prev, list):
                out[-1] = prev[1] + "%"
            else:
                out[-1] = prev + "%"
            continue
        if out and isinstance(out[-1], list):
            out[-1] = out[-1][1] + c            # attach number to its symbol
            continue
        out.append(c)
    return [x[1] if isinstance(x, list) else x for x in out]


def _table_to_rows(table: Tag) -> list[list[str]]:
    """
    Flatten a <table> into aligned string rows.

    The first cell is always kept, even when blank, because it is the row
    label column - dropping it would shift header cells out of alignment
    with the data beneath them.
    """
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        raw = [_norm_cell(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        if not raw:
            continue
        cells = [raw[0]] + _collapse_spacers(raw[1:])
        if any(c for c in cells):
            rows.append(cells)
    return rows


def _looks_like_toc(rows: list[list[str]]) -> bool:
    """The table of contents is a table too. We don't want it."""
    joined = " ".join(" ".join(r) for r in rows[:6]).lower()
    return joined.count("item") >= 4


def _is_numeric_table(rows: list[list[str]]) -> bool:
    """
    True if the table carries financial figures rather than being a layout
    device. We require several cells that parse as numbers.
    """
    numeric = 0
    for row in rows:
        for cell in row:
            if re.fullmatch(r"\(?\$?\s*-?[\d,]+(\.\d+)?\s*\)?%?", cell) and any(
                ch.isdigit() for ch in cell
            ):
                numeric += 1
    return numeric >= 6


# ---------------------------------------------------------------------------
# Main parse
# ---------------------------------------------------------------------------

def parse_filing(source: str | Path | bytes, filing_id: str | None = None) -> dict:
    """
    Parse a 10-K HTML filing.

    Returns the contract shape:
        {"filing_id": str,
         "text_blocks": [{"block_id", "text", "page", "section"}],
         "tables": [{"table_id", "page", "section", "rows"}]}

    Chunking is deliberately NOT done here — that is chunker.py's job.
    This function's only responsibility is faithful extraction with
    correct page and section labels.
    """
    # Accepts a path or raw bytes, because Streamlit hands over an upload in
    # memory and never writes it to disk.
    if isinstance(source, (bytes, bytearray)):
        html = bytes(source).decode("utf-8", errors="ignore")
        filing_id = filing_id or "uploaded"
    else:
        path = Path(source)
        html = path.read_text(encoding="utf-8", errors="ignore")
        filing_id = filing_id or path.stem

    soup = BeautifulSoup(html, "lxml")

    # Strip things that only add noise.
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # Tables are pulled out first and replaced with a placeholder so they
    # don't get mixed into the prose stream.
    tables: list[dict] = []
    for i, table in enumerate(soup.find_all("table")):
        rows = _table_to_rows(table)
        placeholder = soup.new_tag("div")
        if rows and not _looks_like_toc(rows) and _is_numeric_table(rows):
            table_id = f"tb_{len(tables):04d}"
            tables.append({"table_id": table_id, "rows": rows})
            placeholder["data-table-id"] = table_id
        else:
            placeholder["data-table-id"] = "__skip__"
        table.replace_with(placeholder)

    tables_by_id = {t["table_id"]: t for t in tables}

    # Two ways to find page boundaries, and filings use one or the other.
    # A text footer is authoritative when present because it states the
    # printed page number outright; otherwise we count page-break elements,
    # which gives correct sequential numbering.
    footer_hits = len(PAGE_FOOTER_RE.findall(soup.get_text("\n")))
    use_footers = footer_hits >= 5
    break_elements = _page_break_count(soup)

    # Walk the document in order, tracking page and section as we go.
    page = 1
    tracker = SectionTracker()
    text_blocks: list[dict] = []
    seen_text: set[str] = set()
    last_scale: str | None = None
    last_scale_at = -99
    block_no = 0
    # counted page -> the printed number seen on it
    page_marks: dict[int, int] = {}

    for el in soup.find_all(True):
        if not use_footers and _is_page_break(el):
            page += 1
            continue

        # Table placeholder: stamp the table with where we currently are.
        tid = el.get("data-table-id")
        if tid and tid in tables_by_id:
            tables_by_id[tid]["page"] = page
            tables_by_id[tid]["section"] = tracker.current
            if last_scale and block_no - last_scale_at <= SCALE_LOOKBACK:
                tables_by_id[tid]["scale_hint"] = last_scale
            continue

        if el.name not in BLOCK_TAGS:
            continue

        text = _clean(_own_text(el))
        if not text:
            continue

        # A block that is nothing but a number is the printed page number.
        # Recorded, not acted on: one stray figure must not move the count,
        # so the offset is decided from all of them together after the walk.
        if not use_footers and BARE_PAGE_RE.fullmatch(text):
            printed = int(text)
            if 0 < printed <= 400 and abs(printed - page) <= PAGE_MARK_WINDOW:
                page_marks.setdefault(page, printed)

        # Page footer: authoritative page number for everything that follows.
        if use_footers:
            footer = PAGE_FOOTER_RE.search(text)
            if footer:
                page = int(footer.group(1)) + 1
                continue

        tracker.update(text)

        # Track the most recent units declaration, whatever its length -
        # a bare "(In millions)" line is shorter than the block threshold
        # below but is exactly the declaration we need.
        decl = SCALE_DECL_RE.search(text)
        if decl:
            last_scale = decl.group(1).lower()
            last_scale_at = block_no
        block_no += 1

        if len(text) < 30:
            continue  # headings, labels, stray cells

        key = (text[:120], page)
        if key in seen_text:
            continue
        seen_text.add(key)

        text_blocks.append(
            {
                "block_id": f"t_{len(text_blocks):05d}",
                "text": text,
                "page": page,
                "section": tracker.current,
            }
        )

    # Any table we never reached in the walk gets a fallback label.
    for t in tables:
        t.setdefault("page", 0)
        t.setdefault("section", "Unknown")

    # Align the counted pages with the printed ones. A single offset is used
    # rather than per-page correction: the gap is the unnumbered front matter
    # and so is constant, and an agreeing majority is required before any
    # correction is applied. With too few marks, or marks that disagree,
    # nothing changes and the counted sequence stands.
    offset = 0
    if len(page_marks) >= PAGE_MARK_MIN:
        diffs = sorted(printed - counted for counted, printed in page_marks.items())
        median = diffs[len(diffs) // 2]
        if sum(1 for d in diffs if d == median) >= max(PAGE_MARK_MIN,
                                                       int(0.6 * len(diffs))):
            offset = median

    if offset:
        for b in text_blocks:
            b["page"] = max(1, b["page"] + offset)
        for t in tables:
            if t.get("page"):
                t["page"] = max(1, t["page"] + offset)

    return {
        "filing_id": filing_id,
        "text_blocks": text_blocks,
        "tables": tables,
        "_sections_found": tracker.sections_found,
        "_page_method": (
            f"footer ({footer_hits})" if use_footers
            else (f"page-break ({break_elements})"
                  + (f", aligned {offset:+d} against {len(page_marks)} "
                     f"printed numbers" if offset else
                     f", {len(page_marks)} printed numbers found"
                     if page_marks else "")) if break_elements
            else "NONE - pages unreliable"),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    parsed = parse_filing(src)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{parsed['filing_id']}.parsed.json"
    out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    print(f"filing_id     : {parsed['filing_id']}")
    print(f"text blocks   : {len(parsed['text_blocks'])}")
    print(f"tables        : {len(parsed['tables'])}")
    pages = [b["page"] for b in parsed["text_blocks"]]
    if pages:
        print(f"page range    : {min(pages)} - {max(pages)}")
    print(f"sections      : {len(parsed['_sections_found'])} headings hit")
    for s in parsed["_sections_found"][:20]:
        print(f"   {s}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()