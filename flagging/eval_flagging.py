"""
eval_flagging.py - Step 10. Numbers for the detection and explanation track.

Retrieval can be scored automatically because FinQA annotates which passages
are correct. Nothing annotates which financial changes are worth flagging,
so this track needs hand labels. This file collects them and computes what
they support.

What can and cannot be measured
-------------------------------
PRECISION is measurable. Every flag the system raises can be judged: is this
a change an analyst would want to look at, or noise?

RECALL is not. Measuring it would mean reading a 60-page filing in full and
listing every concern the system should have raised, for every filing, which
is beyond what this project can do honestly. Reporting precision alone and
saying why is better than reporting a recall number built on a guess.

EVIDENCE ACCURACY is measurable for the flags marked explained: does the
cited passage actually account for the change?

FAITHFULNESS is measured by the canary in faithfulness_check.py, not here.
Our generator is extractive, so a pass rate on its own output is close to
guaranteed and says little.

Usage:
    python -m flagging.eval_flagging --label aapl-2025     # label by hand
    python -m flagging.eval_flagging --report              # compute metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROCESSED = Path("data/processed")
LABELS = Path("data/labels")


def _ask(prompt: str, options: str) -> str:
    """
    Ask until one of the allowed answers is given.

    An empty line must never count as an answer. The first version treated
    it as "no", so pressing Enter through the prompts silently recorded
    every flag as a false positive and produced a precision of 0.07 that
    meant nothing.
    """
    allowed = set(options)
    while True:
        answer = input(f"{prompt} [{'/'.join(sorted(allowed))}] ").strip().lower()
        if answer in allowed:
            return answer
        print(f"  Please answer one of: {', '.join(sorted(allowed))}")


def _label_path(filing_id: str) -> Path:
    return LABELS / f"{filing_id}.labels.json"


def load_labels(filing_id: str) -> dict:
    path = _label_path(filing_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def label_filing(filing_id: str) -> None:
    """
    Walk through each flag and record a judgement.

    Evidence is shown in full, never truncated. A 105-character preview
    caused a passage to be misjudged twice during development - once as a
    false positive when the explanation was two lines further in.
    """
    src = PROCESSED / f"{filing_id}.verified.json"
    if not src.exists():
        raise SystemExit(f"missing {src} - run the pipeline for {filing_id} first")

    items = json.loads(src.read_text(encoding="utf-8"))
    labels = load_labels(filing_id)
    LABELS.mkdir(parents=True, exist_ok=True)

    print(f"\nLabelling {filing_id}: {len(items)} flags.")
    print("  y = genuine concern    n = false positive")
    print("  s = skip this one      q = save and quit")
    print("  Blank answers are not accepted.\n")

    for item in items:
        aid = item.get("anomaly_id") or item["finding"][:40]
        if aid in labels:
            continue

        print("=" * 76)
        print(f"RULE      {item.get('rule')}   severity {item.get('severity')}")
        print(f"FINDING   {item['finding']}")
        if item.get("detail"):
            print(f"FIGURES   {json.dumps(item['detail'])}")
        print(f"EXPLAINED {bool(item.get('supported'))}")
        if item.get("citation"):
            c = item["citation"]
            print(f"CITATION  page {c['page']} - {c['section']}")
            print(f"PASSAGE   {c['text_span']}")
        print("=" * 76)

        answer = _ask("Is this a genuine concern an analyst would want "
                      "to see?", "ynsq")
        if answer == "q":
            break
        if answer == "s":
            continue

        entry = {"genuine": answer == "y",
                 "rule": item.get("rule"),
                 "supported": bool(item.get("supported"))}

        if item.get("supported"):
            ev = _ask("Does the cited passage actually explain it?", "yn")
            entry["evidence_correct"] = ev == "y"

        note = input("Note (optional, press Enter to skip): ").strip()
        if note and note.lower() not in {"y", "n"}:
            entry["note"] = note

        labels[aid] = entry
        _label_path(filing_id).write_text(
            json.dumps(labels, indent=2), encoding="utf-8")
        print()

    print(f"saved {len(labels)} labels to {_label_path(filing_id)}")


def report() -> None:
    """Aggregate every label file into the numbers for the report."""
    if not LABELS.exists() or not any(LABELS.glob("*.labels.json")):
        raise SystemExit("no labels yet - run --label <filing_id> first")

    rows: list[dict] = []
    per_filing: dict[str, list[dict]] = {}
    for path in sorted(LABELS.glob("*.labels.json")):
        filing = path.stem.replace(".labels", "")
        entries = list(json.loads(path.read_text(encoding="utf-8")).values())
        per_filing[filing] = entries
        rows.extend(entries)

    if not rows:
        raise SystemExit("label files are empty")

    genuine = sum(1 for r in rows if r["genuine"])
    print("=" * 70)
    print("FLAGGING EVALUATION")
    print("=" * 70)
    print(f"flags labelled : {len(rows)} across {len(per_filing)} filings")
    print(f"precision      : {genuine}/{len(rows)} = {genuine / len(rows):.3f}")
    print("  (share of raised flags judged worth an analyst's attention)")

    print("\nprecision by rule")
    by_rule: dict[str, list[dict]] = {}
    for r in rows:
        by_rule.setdefault(r.get("rule") or "?", []).append(r)
    for rule, group in sorted(by_rule.items(),
                              key=lambda kv: -len(kv[1])):
        hits = sum(1 for g in group if g["genuine"])
        print(f"  {rule:<18} {hits}/{len(group)} = {hits / len(group):.3f}")

    print("\nby filing")
    for filing, entries in per_filing.items():
        hits = sum(1 for e in entries if e["genuine"])
        print(f"  {filing:<14} {hits}/{len(entries)} = {hits / len(entries):.3f}")

    checked = [r for r in rows if "evidence_correct" in r]
    if checked:
        correct = sum(1 for r in checked if r["evidence_correct"])
        print(f"\nevidence accuracy : {correct}/{len(checked)} = "
              f"{correct / len(checked):.3f}")
        print("  (of flags the system marked explained, share whose cited "
              "passage genuinely accounts for the change)")

    explained = sum(1 for r in rows if r["supported"])
    print(f"\nexplained by filing : {explained}/{len(rows)} = "
          f"{explained / len(rows):.3f}")
    print("  Not an accuracy figure. It is how often management narrates a "
          "change our arithmetic flagged;")
    print("  the remainder are the concerns a reader would otherwise have "
          "to find unaided.")

    print("\nRecall is not reported. Establishing it would require reading "
          "each filing in full")
    print("and enumerating every concern the system should have raised, "
          "which was beyond")
    print("the scope of this project. Precision is measured; recall is not "
          "estimated.")

    notes = [r for r in rows if r.get("note")]
    if notes:
        print("\nnotes")
        for r in notes:
            verdict = "genuine" if r["genuine"] else "false positive"
            print(f"  [{verdict}] {r['note']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", metavar="FILING_ID",
                    help="label the flags for one filing by hand")
    ap.add_argument("--report", action="store_true",
                    help="compute metrics from every label file")
    args = ap.parse_args()

    if args.label:
        label_filing(args.label)
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()