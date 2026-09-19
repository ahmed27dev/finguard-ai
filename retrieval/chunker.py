"""
chunker.py - Step 3 of the retrieval track.

Turns parsed text blocks and linearized table rows into the uniform
"chunks" that go into the search index.

Two jobs:

  SPLIT   Some blocks run to 3,600 characters. The embedding model
          (all-MiniLM-L6-v2) truncates at 256 word-pieces, roughly 1,000
          characters. Anything past that is silently discarded - the model
          does not warn you, it just never sees the text. So long blocks
          must be split, on sentence boundaries, with a little overlap so
          a fact sitting on the seam survives in at least one chunk.

  MERGE   Other blocks are one short sentence. Embedded alone they carry
          almost no context and match noisily. Consecutive short blocks
          from the same page and section get merged up to target size.

Table rows are passed through as their own chunks: they are already one
self-contained sentence each, and splitting or merging them would break
the row_id link back to the source cell that citations depend on.

Usage:
    python -m retrieval.chunker aapl-2025
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# all-MiniLM-L6-v2 truncates at 256 word-pieces. Roughly 4 characters per
# token gives ~1,000 characters of usable input; we target below that so
# nothing is lost, and cap hard at the ceiling.
TARGET_CHARS = 800
MAX_CHARS = 1000
OVERLAP_CHARS = 120
MIN_CHARS = 40

# Only genuinely short fragments get merged. A block that is already a
# complete statement of reasonable length stands on its own: merging the
# five per-segment explanations on page 22 into one chunk made every one
# of them harder to retrieve and impossible to cite precisely.
MERGE_MAX_CHARS = 120
# A block ending in sentence punctuation and longer than this is a complete
# statement, so it stands alone however short it is. Length alone is not a
# safe test: two per-segment explanations of 110 characters each are still
# two separate answers and merging them makes both harder to retrieve.
STANDALONE_MIN_CHARS = 70

# Abbreviations that end in a period but do not end a sentence. Without
# these a naive splitter cuts "U.S. Tariffs" into two fragments, and 10-K
# prose is thick with them.
ABBREVIATIONS = [
    "U.S", "U.K", "E.U", "Inc", "Corp", "Ltd", "Co", "plc", "LLC", "L.P",
    "No", "Nos", "vs", "etc", "e.g", "i.e", "approx", "Mr", "Ms", "Mrs",
    "Dr", "Jr", "Sr", "St", "Sec", "Art", "Fig", "Ref", "Sept", "Jan",
    "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Oct", "Nov", "Dec",
]
_ABBR_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\.\s",
    re.IGNORECASE,
)
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    """
    Split into sentences, protecting known abbreviations.

    Abbreviation periods are swapped for a placeholder before splitting
    and restored afterwards, so "sales in the U.S. increased" stays whole.
    """
    guarded = _ABBR_RE.sub(lambda m: f"{m.group(1)}\x00 ", text)
    parts = _SENT_END_RE.split(guarded)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _split_long_text(text: str) -> list[str]:
    """Break one over-long block into overlapping, sentence-aligned pieces."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than the ceiling has to be cut on words.
        if len(sentence) > MAX_CHARS:
            if current:
                chunks.append(current.strip())
                current = ""
            words, buf = sentence.split(), ""
            for w in words:
                if len(buf) + len(w) + 1 > MAX_CHARS:
                    chunks.append(buf.strip())
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                current = buf
            continue

        if len(current) + len(sentence) + 1 <= TARGET_CHARS:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            # Carry the tail of the previous chunk forward so a fact on the
            # boundary appears in both neighbours.
            tail = current[-OVERLAP_CHARS:] if current else ""
            if tail:
                tail = tail[tail.find(" ") + 1:] if " " in tail else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= MIN_CHARS]


def _is_standalone(text: str) -> bool:
    """A complete sentence of reasonable length is its own retrievable unit."""
    return len(text) >= STANDALONE_MIN_CHARS and text.rstrip()[-1:] in ".!?"


def chunk_text_blocks(blocks: list[dict]) -> list[dict]:
    """
    Split long blocks and merge short ones.

    Merging never crosses a page or section boundary, because a chunk's
    citation must name exactly one page and one section. A chunk spanning
    two pages could not be cited honestly.
    """
    chunks: list[dict] = []
    pending: list[dict] = []

    def flush() -> None:
        if not pending:
            return
        text = " ".join(b["text"] for b in pending)
        first = pending[0]
        for piece in (_split_long_text(text) if len(text) > MAX_CHARS else [text]):
            chunks.append(
                {
                    "chunk_id": f"c_{len(chunks):05d}",
                    "text": piece,
                    "source_type": "text",
                    "page": first["page"],
                    "section": first["section"],
                    "origin_ids": [b["block_id"] for b in pending],
                }
            )
        pending.clear()

    for block in blocks:
        text = block["text"].strip()
        if len(text) < MIN_CHARS:
            continue

        if pending:
            same_place = (
                block["page"] == pending[0]["page"]
                and block["section"] == pending[0]["section"]
            )
            running = sum(len(b["text"]) for b in pending)
            if not same_place or running + len(text) > TARGET_CHARS:
                flush()

        if len(text) > MAX_CHARS:
            flush()
            for piece in _split_long_text(text):
                chunks.append(
                    {
                        "chunk_id": f"c_{len(chunks):05d}",
                        "text": piece,
                        "source_type": "text",
                        "page": block["page"],
                        "section": block["section"],
                        "origin_ids": [block["block_id"]],
                    }
                )
        elif _is_standalone(text):
            flush()
            chunks.append(
                {
                    "chunk_id": f"c_{len(chunks):05d}",
                    "text": text,
                    "source_type": "text",
                    "page": block["page"],
                    "section": block["section"],
                    "origin_ids": [block["block_id"]],
                }
            )
        elif len(text) < MERGE_MAX_CHARS:
            pending.append(block)
        else:
            flush()
            chunks.append(
                {
                    "chunk_id": f"c_{len(chunks):05d}",
                    "text": text,
                    "source_type": "text",
                    "page": block["page"],
                    "section": block["section"],
                    "origin_ids": [block["block_id"]],
                }
            )

    flush()
    return chunks


def chunk_table_rows(records: list[dict], start_index: int) -> list[dict]:
    """
    Pass linearized rows through as chunks, one row each.

    Single-year "stacked" rows are skipped: their merged_series counterpart
    already carries the same numbers with the full time span, so indexing
    both would put near-duplicate text in the index and let a single row
    crowd out genuinely different evidence in the top-k.
    """
    chunks: list[dict] = []
    for rec in records:
        if rec["layout"] == "stacked":
            continue
        text = rec["linear"]
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rsplit("|", 1)[0].strip()
        chunks.append(
            {
                "chunk_id": f"c_{start_index + len(chunks):05d}",
                "text": text,
                "source_type": "table",
                "page": rec["page"],
                "section": rec["section"],
                "origin_ids": [rec["row_id"]],
                "table_id": rec["table_id"],
                "label": rec["label"],
            }
        )
    return chunks


def build_chunks(parsed: dict, linear: list[dict]) -> list[dict]:
    text_chunks = chunk_text_blocks(parsed["text_blocks"])
    table_chunks = chunk_table_rows(linear, len(text_chunks))
    return text_chunks + table_chunks


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filing_id = sys.argv[1]
    proc = Path("data/processed")
    parsed = json.loads((proc / f"{filing_id}.parsed.json").read_text(encoding="utf-8"))
    linear = json.loads((proc / f"{filing_id}.linear.json").read_text(encoding="utf-8"))

    chunks = build_chunks(parsed, linear)
    out = proc / f"{filing_id}.chunks.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    text_n = sum(1 for c in chunks if c["source_type"] == "text")
    table_n = len(chunks) - text_n
    lens = sorted(len(c["text"]) for c in chunks)
    over = sum(1 for n in lens if n > MAX_CHARS)

    print(f"chunks total   : {len(chunks)}  (text {text_n}, table {table_n})")
    print(f"length min/med/max : {lens[0]} / {lens[len(lens) // 2]} / {lens[-1]}")
    print(f"over {MAX_CHARS} chars : {over}   (want 0)")
    print(f"sections covered   : {len({c['section'] for c in chunks})}")
    print(f"\nwrote {out}\n")
    print("--- sample text chunk ---")
    print("  " + next(c["text"] for c in chunks if c["source_type"] == "text")[:200])
    print("--- sample table chunk ---")
    print("  " + next(c["text"] for c in chunks if c["source_type"] == "table")[:200])


if __name__ == "__main__":
    main()