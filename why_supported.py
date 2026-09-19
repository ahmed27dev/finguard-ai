"""
why_supported.py - show why a passage was accepted or rejected as support.

The investigate output truncates passages at ~105 characters, so a match on
a causal phrase 200 characters in is invisible. This prints each anomaly's
evidence in full alongside the four checks in _is_supporting, so a wrong
verdict can be attributed to the right rule.

Usage:
    python why_supported.py nvda-2026
    python why_supported.py nvda-2026 --only revenue
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrieval.retrieve import (
    _anchor_terms, _mentions, EXPLANATORY_RE, MIN_SUPPORT_SCORE,
)

PROCESSED = Path("data/processed")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filing_id = sys.argv[1]
    needle = None
    if "--only" in sys.argv:
        needle = sys.argv[sys.argv.index("--only") + 1].lower()

    path = PROCESSED / f"{filing_id}.investigated.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run pipeline.py {filing_id} --investigate")

    anomalies = json.loads(path.read_text(encoding="utf-8"))

    for a in anomalies:
        if needle and needle not in a["summary"].lower():
            continue

        anchors = _anchor_terms(a["label"])
        print("=" * 78)
        print(f"{a['rule']}  |  {a['summary'][:90]}")
        print(f"anchor {a['label']!r} -> {sorted(anchors) or '(none)'} "
              f"({len(anchors)} term{'s' if len(anchors) != 1 else ''})")
        print(f"verdict: {'EXPLAINED' if a['supported'] else 'UNEXPLAINED'}")
        print("=" * 78)

        for i, e in enumerate(a["evidence"], start=1):
            score_ok = e["score"] >= MIN_SUPPORT_SCORE
            text_ok = e["source_type"] == "text"
            anchor_ok = (not anchors) or _mentions(e["text"], anchors)
            cue = EXPLANATORY_RE.search(e["text"])
            cue_needed = len(anchors) <= 1
            cue_ok = (not cue_needed) or bool(cue)

            print(f"\n  [{i}] score {e['score']:>6.2f}  p{e['page']}  "
                  f"({e['source_type']})")
            print(f"      score  {'pass' if score_ok else 'FAIL'}"
                  f"   text {'pass' if text_ok else 'FAIL'}"
                  f"   anchor {'pass' if anchor_ok else 'FAIL'}"
                  f"   cue {'pass' if cue_ok else 'FAIL'}"
                  f"{'' if cue_needed else ' (not required)'}")
            if cue:
                start = max(cue.start() - 45, 0)
                print(f"      cue matched {cue.group(0)!r} in: "
                      f"...{e['text'][start:cue.end() + 45]}...")
            if anchors and not anchor_ok:
                missing = [t for t in sorted(anchors)
                           if not _mentions(e["text"], {t})]
                print(f"      missing anchor terms: {missing}")
            print(f"      FULL: {e['text']}")
        print()


if __name__ == "__main__":
    main()