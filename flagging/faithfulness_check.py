"""
faithfulness_check.py - Step 9. Verifies the explanation against its citation.

Asks one question of every generated explanation: does the passage we cited
actually support what we wrote? An entailment model reads the evidence as
premise and the explanation as hypothesis and returns entailment,
contradiction or neutral.

What is checked, and what is not
--------------------------------
An explanation has two halves and only one of them is the model's:

    "gross margin moved 75.0% to 71.1% (-3.92pp), 2025 to 2026.
     The filing states: as our business model transitioned from Hopper HGX
     to Blackwell full-scale datacenter solutions."
     |------------ finding -----------|----------- claim -----------|

The finding is arithmetic. It came from the metric extractor and is
verified by construction - recomputable from the filing's own tables. The
evidence passage does not entail it and was never meant to; running NLI on
it would fail every item for the wrong reason.

So only the claim is checked. Splitting the two is what makes the check
mean something: it isolates the part that could have been invented.

Three tests, not one
--------------------
    ENTAILMENT   the NLI model's judgement
    NUMBERS      every figure in the claim appears in the evidence, since
                 NLI models are unreliable on numeric agreement
    HEDGING      the claim asserts no cause the evidence did not state

A claim passes only if all three agree.

Honest limitation
-----------------
Our generator is extractive, so its claims are often near-verbatim spans of
the evidence and entailment is close to guaranteed. A high pass rate is
therefore weak evidence that the checker works. The canary test exists for
that reason: it feeds the checker explanations that are deliberately wrong
and reports whether they are caught.

Usage:
    python -m flagging.faithfulness_check aapl-2025
    python -m flagging.faithfulness_check --canary
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROCESSED = Path("data/processed")
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
LABELS = ["contradiction", "entailment", "neutral"]

ENTAILMENT_MIN = 0.50
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
CAUSAL_RE = re.compile(
    r"\b(caused by|because of|due to|driven by|as a result of|resulted from|"
    r"attributable to|owing to|led to)\b", re.I)

# The wording the generator adds around an extracted span. Stripped before
# checking so the check is on the filing's words, not our framing.
FRAME_RE = re.compile(r"^the filing states:\s*", re.I)

_model = None


def get_model():
    """Load the NLI model once. First call downloads ~270 MB."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        print(f"loading {MODEL_NAME} ...")
        _model = CrossEncoder(MODEL_NAME)
    return _model


def split_explanation(explanation: str, finding: str) -> str:
    """
    Return the part of the explanation that the model produced.

    Everything up to and including the finding is arithmetic we computed;
    only what follows is a claim about the filing.
    """
    idx = explanation.find(finding)
    claim = (explanation[idx + len(finding):] if idx >= 0 else explanation)
    claim = claim.lstrip(". ").strip()
    return FRAME_RE.sub("", claim).strip()


def entailment_score(premise: str, hypothesis: str) -> tuple[str, float]:
    """Run NLI and return (label, probability of entailment)."""
    import numpy as np

    scores = get_model().predict([(premise, hypothesis)])[0]
    exp = np.exp(scores - np.max(scores))
    probs = exp / exp.sum()
    return LABELS[int(np.argmax(probs))], float(probs[LABELS.index("entailment")])


def check_numbers(claim: str, evidence_text: str) -> tuple[bool, list[str]]:
    """
    Every figure in the claim must appear in the evidence.

    NLI models judge numeric agreement poorly - swapping one figure for
    another often leaves entailment high - so numbers get their own test
    rather than being left to the model.
    """
    source = {n.replace(",", "") for n in NUMBER_RE.findall(evidence_text)}
    missing = [n for n in NUMBER_RE.findall(claim)
               if n.replace(",", "") not in source]
    return not missing, missing


def check_causation(claim: str, evidence_text: str) -> bool:
    """A causal assertion is faithful only if the evidence made it first."""
    if not CAUSAL_RE.search(claim):
        return True
    return bool(CAUSAL_RE.search(evidence_text))


def verify(item: dict) -> dict:
    """
    Verify one explanation. Returns the item with a verdict attached.

    verdict is one of:
      supported      all three tests pass
      verbatim       template output; quotes the evidence, nothing to check
      unverified     no citation to check against
      not_supported  at least one test failed, with the reason recorded
    """
    result = dict(item)
    citation = item.get("citation")

    # A template explanation quotes the evidence verbatim and states the
    # arithmetic, so there is nothing in it that could have been invented.
    # Verifying it anyway produced a spurious failure: the page number in
    # our own framing - "On page 44 (Item 7) the filing states:" - was read
    # as a figure absent from the evidence.
    if item.get("source") == "template" and item.get("supported"):
        result.update({"verdict": "verbatim", "faithfulness_score": None,
                       "nli_label": None, "failures": []})
        return result

    if not citation or not item.get("supported"):
        result.update({"verdict": "unverified", "faithfulness_score": None,
                       "nli_label": None,
                       "failures": ["no citation to verify against"]})
        return result

    evidence_text = citation["text_span"]
    claim = split_explanation(item["explanation"], item["finding"])

    if not claim:
        result.update({"verdict": "unverified", "faithfulness_score": None,
                       "nli_label": None, "failures": ["no claim to verify"]})
        return result

    label, score = entailment_score(evidence_text, claim)
    numbers_ok, missing = check_numbers(claim, evidence_text)
    causation_ok = check_causation(claim, evidence_text)

    failures = []
    if label != "entailment" or score < ENTAILMENT_MIN:
        failures.append(f"entailment {label} ({score:.2f})")
    if not numbers_ok:
        failures.append(f"figures absent from evidence: {missing}")
    if not causation_ok:
        failures.append("asserts a cause the evidence does not state")

    result.update({
        "claim": claim,
        "nli_label": label,
        "faithfulness_score": round(score, 4),
        "verdict": "supported" if not failures else "not_supported",
        "failures": failures,
    })
    return result


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

CANARY = [
    # (name, evidence, claim, should_pass)
    ("faithful extraction",
     "Greater China net sales decreased during 2025 compared to 2024 primarily "
     "due to lower net sales of iPhone, partially offset by higher net sales of Mac.",
     "lower net sales of iPhone",
     True),
    ("over-claimed cause",
     "Operating expenses increased due to restructuring costs.",
     "Operating income decreased primarily due to increased operating expenses "
     "associated with restructuring costs.",
     False),
    ("invented figure",
     "The Company's effective tax rate for 2025 was lower compared to 2024 due to "
     "a $10.7 billion year-over-year decrease in the provision for income taxes.",
     "a $22.4 billion decrease in the provision for income taxes",
     False),
    ("contradicts the evidence",
     "Greater China net sales decreased during 2025 compared to 2024.",
     "Greater China net sales grew strongly during 2025",
     False),
    ("unrelated passage",
     "The Company had approximately 166,000 full-time equivalent employees.",
     "lower net sales of iPhone in Greater China",
     False),
    ("cause stated by the filing",
     "Gross margin decreased in fiscal year 2026 due to a shift in product mix.",
     "gross margin fell due to a shift in product mix",
     True),
]


def run_canary() -> None:
    """
    Feed the checker explanations that are known-good and known-bad.

    A checker that passes everything is decorative. This is the test that
    says whether it is doing any work, and it is the number worth reporting
    rather than the pass rate on our own extractive output.
    """
    print("CANARY - known-good and known-bad pairs\n" + "=" * 70)
    passed = 0
    for name, evidence, claim, should_pass in CANARY:
        item = {
            "finding": "FINDING",
            "explanation": f"FINDING. The filing states: {claim}",
            "supported": True,
            "citation": {"page": 0, "section": "-", "chunk_id": "-",
                         "text_span": evidence},
        }
        out = verify(item)
        got_pass = out["verdict"] == "supported"
        ok = got_pass == should_pass
        passed += ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}")
        print(f"        expected {'pass' if should_pass else 'reject'}, "
              f"got {'pass' if got_pass else 'reject'} "
              f"({out['nli_label']}, {out['faithfulness_score']})")
        if out["failures"]:
            print(f"        {'; '.join(out['failures'])}")
    print("=" * 70)
    print(f"{passed}/{len(CANARY)} canary cases handled correctly")
    if passed < len(CANARY):
        print("A checker that cannot separate these is not verifying anything.")


def main() -> None:
    if "--canary" in sys.argv:
        run_canary()
        return

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filing_id = sys.argv[1]
    src = PROCESSED / f"{filing_id}.explained.json"
    if not src.exists():
        raise SystemExit(f"missing {src} - run flagging.generator first")

    items = json.loads(src.read_text(encoding="utf-8"))
    results = [verify(i) for i in items]

    out = PROCESSED / f"{filing_id}.verified.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    print(f"explanations : {len(results)}")
    for verdict, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {n:>3}  {verdict}")
    print()

    for r in results:
        if r["verdict"] == "unverified":
            continue
        mark = {"supported": "PASS", "verbatim": "QUOTE"}.get(
            r["verdict"], "FAIL")
        # verbatim results carry no score - they quote the evidence, so
        # there is nothing an entailment model would be scoring.
        score = (f"{r['faithfulness_score']:.2f} ({r['nli_label']})"
                 if r.get("faithfulness_score") is not None
                 else "quoted verbatim")
        print(f"[{mark}] {score} {r['finding'][:70]}")
        print(f"       claim: {r.get('claim', '')[:120]}")
        for f in r["failures"]:
            print(f"       ! {f}")
        print()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()