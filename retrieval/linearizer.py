"""
linearizer.py - Step 2 of the retrieval track.

Turns the tables produced by parser.py into two things at once:

  1. a searchable sentence per row, so tables can go into the same
     embedding index as prose (this is what retrieval needs)
  2. a structured {period: value} mapping, so Person B can run
     arithmetic on the numbers without re-parsing anything

A 10-K uses two table shapes and both must be handled:

  WIDE            years are columns, one table holds the series
                  | Americas | $178,353 | 7% | $167,045 | 3% | $162,560 |

  STACKED         one table per year, columns are categories
                  tb_0036 = 2025, tb_0037 = 2024, tb_0038 = 2023

Stacked tables are joined back into series by merge_stacked(), because
the most useful anomalies (segment operating income over time) are
invisible otherwise.

Usage:
    python -m retrieval.linearizer aapl-2025
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

YEAR_RE = re.compile(r"^(19|20)\d{2}$")
# Header cells are not always bare years. Nvidia's fiscal calendar gives
# "Jan 25, 2026"; a bare-year pattern matches none of its tables and the
# whole filing yields zero rows. Pull the year out of any short date cell.
PERIOD_IN_CELL_RE = re.compile(r"\b((?:19|20)\d{2})\b")
# A period cell is either a bare year or a date. Requiring a month name
# rejects data labels that merely contain a year, such as the rows of a
# debt maturity table ("3.20% Notes Due 2026").
MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
SCALE_RE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)

# Columns that hold derived figures rather than reported ones. We keep the
# reported values only - the whole point of the project is that our own
# arithmetic produces the changes, not the filing's.
DERIVED_HEADERS = {"change", "%change", "% change", ""}


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def parse_value(cell: str) -> tuple[float | None, str]:
    """
    Convert one cell to (number, kind).

    kind is one of: "number", "percent", "currency", "empty", "text".
    Accounting negatives use parentheses: (95,699) is -95699.
    A bare dash means "not applicable" and returns None, NOT zero -
    treating it as zero would invent a data point.
    """
    c = (cell or "").strip()
    if not c or c in {"-", "\u2014", "\u2013", "$-", "$\u2014", "N/A", "n/a"}:
        return None, "empty"

    negative = c.startswith("(") and c.endswith(")")
    if negative:
        c = c[1:-1].strip()

    is_percent = c.endswith("%")
    if is_percent:
        c = c[:-1].strip()

    is_currency = bool(c) and c[0] in "$\u20ac\u00a3\u00a5\u20b9"
    if is_currency:
        c = c[1:].strip()

    # A second parenthesis layer appears in cells like $(42,627)
    if c.startswith("(") and c.endswith(")"):
        negative = True
        c = c[1:-1].strip()

    c = c.replace(",", "")
    if not re.fullmatch(r"-?\d*\.?\d+", c):
        return None, "text"

    value = float(c)
    if negative:
        value = -value

    if is_percent:
        return value, "percent"
    if is_currency:
        return value, "currency"
    return value, "number"


def _fmt(value: float) -> str:
    """Render a number the way the filing would print it."""
    if value < 0:
        return f"({abs(value):,.0f})" if value == int(value) else f"({abs(value):,.2f})"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def _year_positions(row: list[str]) -> dict[int, str]:
    """
    Column index -> year, for cells that denote a reporting period.

    Accepts a bare year ("2025") or a date carrying one ("Jan 25, 2026").
    Long cells are ignored so prose like "3.20% Notes Due 2026" in a debt
    table is not mistaken for a column header.
    """
    out: dict[int, str] = {}
    for i, cell in enumerate(row):
        c = cell.strip()
        if not c or len(c) > 30:
            continue
        if YEAR_RE.fullmatch(c):
            out[i] = c
            continue
        m = PERIOD_IN_CELL_RE.search(c)
        # Only treat it as a period if the cell is a date. A year alone is
        # not enough - plenty of data labels contain one.
        if m and MONTH_RE.search(c) and "%" not in c:
            out[i] = m.group(1)
    return out


def detect_layout(rows: list[str]) -> dict:
    """
    Work out how the table is arranged.

    Returns one of:
      {"kind": "wide",    "header_idx": int, "columns": {col_idx: year}}
      {"kind": "stacked", "period": "2025", "header_idx": int,
       "columns": {col_idx: category}}
      {"kind": "unknown"}
    """
    for i, row in enumerate(rows[:4]):
        years = _year_positions(row)

        # Two or more years across one row -> years are the columns.
        if len(years) >= 2:
            return {"kind": "wide", "header_idx": i, "columns": years}

        # Exactly one year, alone on its row -> it labels the whole table,
        # and the next row carries the real column headers.
        filled = [c for c in row if c.strip()]
        if len(years) == 1 and len(filled) == 1:
            for j in range(i + 1, min(i + 3, len(rows))):
                cand = rows[j]
                labels = {
                    k: c.strip()
                    for k, c in enumerate(cand)
                    if c.strip() and parse_value(c)[1] in {"text", "empty"}
                }
                if len(labels) >= 2:
                    return {
                        "kind": "stacked",
                        "period": next(iter(years.values())),
                        "header_idx": j,
                        "columns": labels,
                    }
    return {"kind": "unknown"}


# A caption states the units for the table directly beneath it. Any other
# paragraph on the page is just prose that happens to mention millions, and
# trusting it mislabels tables that carry no units of their own - the stock
# performance graph, indexed to 100, came out tagged "in millions".
CAPTION_HINT_RE = re.compile(r"following table|as follows|table (shows|sets forth|presents)", re.I)


def detect_scale(table: dict, text_blocks: list[dict]) -> tuple[str | None, str | None]:
    """
    Find whether figures are in thousands / millions / billions.

    Returns (scale, source) where source is "table" (units stated inside
    the table) or "preceding" (declared in the caption or statement title
    immediately above it, captured by the parser in document order). A table
    with no units of its own returns (None, None) rather than borrowing
    units from unrelated prose - an unlabelled number is honest, a wrongly
    labelled one is not.
    """
    # Look inside the table first. Microsoft and Nvidia put "(In millions)"
    # in the table's own first rows; only Apple states it in the paragraph
    # above. Reading prose alone finds nothing for most filers.
    for row in table.get("rows", [])[:6]:
        for cell in row:
            m = SCALE_RE.search(cell)
            if m:
                return m.group(1).lower(), "table"

    hint = table.get("scale_hint")
    if hint:
        return hint, "preceding"
    return None, None


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

def linearize_table(table: dict, scale: str | None = None,
                    scale_source: str | None = None) -> list[dict]:
    """Produce one linearized record per data row of a single table."""
    rows = table["rows"]
    layout = detect_layout(rows)
    if layout["kind"] == "unknown":
        return []

    out: list[dict] = []
    start = layout["header_idx"] + 1
    scale_note = f" (in {scale})" if scale else ""

    for r_i, row in enumerate(rows[start:], start=start):
        label = row[0].strip() if row else ""
        if not label:
            continue

        values: dict[str, float] = {}
        for col_idx, key in layout["columns"].items():
            if str(key).strip().lower() in DERIVED_HEADERS:
                continue
            if col_idx >= len(row):
                continue
            val, kind = parse_value(row[col_idx])
            if val is not None and kind in {"number", "currency"}:
                values[str(key)] = val

        if not values:
            continue  # section heading row such as "Gross margin:"

        parts = " | ".join(f"{k}: {_fmt(v)}" for k, v in values.items())
        if layout["kind"] == "wide":
            sentence = f"{label}{scale_note}: {parts}"
            period = None
        else:
            period = layout["period"]
            sentence = f"{label} for {period}{scale_note}: {parts}"

        out.append(
            {
                "row_id": f"{table['table_id']}_r{r_i}",
                "table_id": table["table_id"],
                "page": table["page"],
                "section": table["section"],
                "label": label,
                "layout": layout["kind"],
                "period": period,
                "scale": scale,
                "scale_source": scale_source,
                "values": values,
                "linear": sentence,
            }
        )
    return out


def merge_stacked(records: list[dict]) -> list[dict]:
    """
    Join per-year tables back into series.

    tb_0036/37/38 are the same segment table for 2025, 2024 and 2023. On
    their own each shows a single year, so no year-over-year comparison is
    possible. Grouping by (row label, column category) across those tables
    rebuilds the series Person B needs.

    Emits additional records; the per-year ones are kept as well.
    """
    groups: dict[tuple, dict] = {}
    for rec in records:
        if rec["layout"] != "stacked" or not rec["period"]:
            continue
        for category, value in rec["values"].items():
            key = (rec["label"], category, rec["section"])
            g = groups.setdefault(
                key,
                {
                    "label": rec["label"],
                    "category": category,
                    "section": rec["section"],
                    "page": rec["page"],
                    "scale": rec["scale"],
                    "scale_source": rec.get("scale_source"),
                    "values": {},
                    "source_row_ids": [],
                },
            )
            g["values"][rec["period"]] = value
            g["source_row_ids"].append(rec["row_id"])

    merged: list[dict] = []
    for i, g in enumerate(groups.values()):
        if len(g["values"]) < 2:
            continue  # a single year is not a series
        years = sorted(g["values"], reverse=True)
        scale_note = f" (in {g['scale']})" if g["scale"] else ""
        parts = " | ".join(f"{y}: {_fmt(g['values'][y])}" for y in years)
        merged.append(
            {
                "row_id": f"merged_{i:04d}",
                "table_id": "merged",
                "page": g["page"],
                "section": g["section"],
                "label": f"{g['label']} - {g['category']}",
                "layout": "merged_series",
                "period": None,
                "scale": g["scale"],
                "scale_source": g.get("scale_source"),
                "values": {y: g["values"][y] for y in years},
                "linear": f"{g['label']} - {g['category']}{scale_note}: {parts}",
                "source_row_ids": g["source_row_ids"],
            }
        )
    return merged


def linearize_filing(parsed: dict) -> list[dict]:
    """Linearize every table in a parsed filing, then add merged series."""
    records: list[dict] = []
    for table in parsed["tables"]:
        scale, source = detect_scale(table, parsed["text_blocks"])
        records.extend(linearize_table(table, scale, source))
    records.extend(merge_stacked(records))
    return records


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filing_id = sys.argv[1]
    src = Path("data/processed") / f"{filing_id}.parsed.json"
    parsed = json.loads(src.read_text(encoding="utf-8"))

    records = linearize_filing(parsed)
    out = Path("data/processed") / f"{filing_id}.linear.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    by_layout: dict[str, int] = {}
    for r in records:
        by_layout[r["layout"]] = by_layout.get(r["layout"], 0) + 1

    print(f"linearized rows : {len(records)}")
    for k, v in sorted(by_layout.items()):
        print(f"   {k:<16} {v}")
    from_table = sum(1 for r in records if r.get("scale_source") == "table")
    from_prec = sum(1 for r in records if r.get("scale_source") == "preceding")
    unscaled = sum(1 for r in records if not r["scale"])
    print(f"scale  in-table: {from_table}   preceding: {from_prec}   "
          f"none: {unscaled}")
    if from_prec == 0 and unscaled > len(records) // 2:
        print("  WARNING: no scale hints found. Re-run retrieval.parser first -"
              " hints are written during parsing, not linearization.")
    print(f"\nwrote {out}\n")
    print("--- sample ---")
    for r in records[:3]:
        print(f"  {r['linear'][:110]}")
    for r in records:
        if r["layout"] == "merged_series":
            print(f"  {r['linear'][:110]}")
            break


if __name__ == "__main__":
    main()