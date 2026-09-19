"""
generator.py - Step 7 of the pipeline. Writes the plain-language explanation.

Turns a flagged anomaly plus its retrieved evidence into a sentence or two
an analyst can read.

Written defensively, because the model is the least reliable part of this
system and the only part that can invent things:

  TEMPLATE FIRST   A template built from the arithmetic and the evidence is
                   always produced. It is correct by construction: every
                   number in it came from the metrics, every claim from a
                   retrieved passage. The model's job is to read better than
                   the template, not to add anything to it.

  VALIDATE         Model output is accepted only if it passes checks: it
                   introduces no number absent from the inputs, it is not
                   degenerate or truncated, and it does not assert a cause
                   the evidence did not state. Failing any check, the
                   template is used instead and the fallback is recorded.

  NO CAUSAL INVENTION  The generator is instructed to report what the
                   filing says, not to explain why. "The filing attributes
                   this to X" is checkable; "this was caused by X" is a
                   claim the evidence usually does not support. This matters
                   because faithfulness_check.py should be catching real
                   errors, not ones we introduced here.

Never runs on unsupported evidence. Asking a model why something happened
while handing it passages that do not say is how it invents a cause.

Usage:
    python -m flagging.generator aapl-2025
    python -m flagging.generator aapl-2025 --template-only
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROCESSED = Path("data/processed")
MODEL_NAME = "google/flan-t5-base"

# FLAN-T5 Base truncates at 512 word-pieces and degrades well before that,
# so only the strongest passages are shown to it.
MAX_EVIDENCE = 2
MAX_EVIDENCE_CHARS = 600
MAX_NEW_TOKENS = 80

_model = None
_tokenizer = None

# Phrases that assert a cause. The generator must not introduce one that the
# evidence did not already contain.
CAUSAL_CLAIM_RE = re.compile(
    r"\b(caused by|because of|due to|driven by|as a result of|resulted from|"
    r"attributable to|owing to)\b", re.I)

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def get_model():
    """Load FLAN-T5 once. First call downloads ~1 GB."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"loading {MODEL_NAME} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Template - always produced, correct by construction
# ---------------------------------------------------------------------------

def build_template(anomaly: dict, evidence: list[dict]) -> str:
    """
    Compose an explanation from the arithmetic and the evidence verbatim.

    Nothing here is generated. The finding is the detector's own summary and
    the account is the retrieved sentence quoted as the filing's, so the
    output cannot contain a claim that is not already in its inputs.
    """
    finding = anomaly["summary"]
    supporting = [e for e in evidence if e.get("supported")]

    if not supporting:
        return (f"{finding}. The filing contains no passage explaining this "
                f"change.")

    top = supporting[0]
    quoted = top["text"].strip()
    if len(quoted) > 320:
        quoted = quoted[:320].rsplit(" ", 1)[0] + "..."

    return (f"{finding}. On page {top['page']} ({top['section']}) the filing "
            f"states: \"{quoted}\"")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_prompt(anomaly: dict, evidence: list[dict]) -> str:
    """
    Frame the task as extractive question answering.

    The first version opened with "Finding: <summary>" and asked for a
    summary. FLAN-T5 Base copied that line back word for word on every
    accepted output - it is weak at abstractive summarising and will echo
    the most salient line in the prompt rather than compose anything.

    It is, however, reasonably good at extractive QA, which is what it was
    instruction-tuned on. So the finding is kept out of the prompt entirely
    and the model is asked one question about one passage. The finding is
    prepended to its answer afterwards, where it cannot be copied.
    """
    passages = []
    for e in evidence[:MAX_EVIDENCE]:
        text = e["text"].strip()
        if len(text) > MAX_EVIDENCE_CHARS:
            text = text[:MAX_EVIDENCE_CHARS].rsplit(" ", 1)[0]
        passages.append(text)

    question = anomaly.get("query") or f"what happened to {anomaly['label']}"
    return (
        "Read the context and answer the question using only the context. "
        "Answer in one short sentence.\n\n"
        f"Context: {' '.join(passages)}\n\n"
        f"Question: {question}?\n\nAnswer:"
    )


def generate_with_model(anomaly: dict, evidence: list[dict]) -> str:
    model, tokenizer = get_model()
    prompt = build_prompt(anomaly, evidence)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             num_beams=4, no_repeat_ngram_size=3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _overlap(a: str, b: str) -> float:
    """Jaccard overlap of the word sets of two strings."""
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def validate(text: str, anomaly: dict, evidence: list[dict]) -> tuple[bool, str]:
    """
    Decide whether model output is safe to use.

    Returns (ok, reason). Each check corresponds to a way a small model
    fails on this task, and any failure sends the answer back to the
    template rather than to the user.
    """
    if not text or len(text) < 25:
        return False, "too short"
    if len(text) > 700:
        return False, "too long"

    words = text.lower().split()
    if len(set(words)) < len(words) / 3:
        return False, "degenerate repetition"

    # Echoing the finding is the failure mode this model actually has. It
    # passes every other check - unique words, no invented numbers, no
    # causal claim - while adding nothing at all, so it needs its own test.
    if _overlap(text, anomaly["summary"]) > 0.55:
        return False, "echoed the finding"

    # Numbers are the easiest thing for a model to get wrong and the most
    # damaging: a wrong figure in a financial explanation is worse than no
    # explanation. Every number in the output must appear in its inputs.
    source = anomaly["summary"] + " " + " ".join(e["text"] for e in evidence)
    source_numbers = {n.replace(",", "") for n in NUMBER_RE.findall(source)}
    for n in NUMBER_RE.findall(text):
        if n.replace(",", "") not in source_numbers:
            return False, f"invented number {n!r}"

    # A causal assertion is allowed only if the evidence made it first.
    if CAUSAL_CLAIM_RE.search(text):
        evidence_text = " ".join(e["text"] for e in evidence)
        if not CAUSAL_CLAIM_RE.search(evidence_text):
            return False, "asserts a cause the evidence does not state"

    return True, "ok"


def explain(anomaly: dict, evidence: list[dict],
            use_model: bool = True) -> dict:
    """
    Produce the explanation for one anomaly.

    Always returns a usable explanation. `source` records whether it came
    from the model or the template, and `fallback_reason` says why when the
    model's output was rejected - which is the number that belongs in the
    evaluation, not a claim that the generator works.
    """
    supporting = [e for e in evidence if e.get("supported")]
    template = build_template(anomaly, evidence)

    result = {
        "anomaly_id": anomaly.get("anomaly_id"),
        "finding": anomaly["summary"],
        # Carried through so the dashboard can show the figures behind the
        # flag. A concern the reader cannot recompute is one they have to
        # take on trust, which is the opposite of the point.
        "rule": anomaly.get("rule"),
        "severity": anomaly.get("severity"),
        "metric_ids": anomaly.get("metric_ids", []),
        "detail": anomaly.get("detail"),
        "page": anomaly.get("page"),
        "section": anomaly.get("section"),
        "explanation": template,
        "source": "template",
        "fallback_reason": None,
        "citation": None,
        "supported": bool(supporting),
    }

    if supporting:
        top = supporting[0]
        result["citation"] = {
            "page": top["page"],
            "section": top["section"],
            "chunk_id": top["chunk_id"],
            "text_span": top["text"],
        }

    # No supporting evidence means nothing for the model to summarise. The
    # template already states that plainly, and that is the honest answer.
    if not supporting or not use_model:
        if not supporting:
            result["fallback_reason"] = "no supporting evidence"
        return result

    try:
        generated = generate_with_model(anomaly, supporting)
    except Exception as exc:  # noqa: BLE001
        result["fallback_reason"] = f"model error: {type(exc).__name__}"
        return result

    ok, reason = validate(generated, anomaly, supporting)
    result["model_output"] = generated
    if ok:
        # The model returns an extracted span, so it is wrapped in wording
        # that attributes the claim to the filing rather than asserting it.
        answer = generated.rstrip(".")
        result["explanation"] = (
            f"{anomaly['summary']}. The filing states: {answer}.")
        result["source"] = "model"
    else:
        result["fallback_reason"] = reason
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filing_id = sys.argv[1]
    use_model = "--template-only" not in sys.argv

    src = PROCESSED / f"{filing_id}.investigated.json"
    if not src.exists():
        raise SystemExit(f"missing {src} - run "
                         f"'python pipeline.py {filing_id} --investigate' first")

    anomalies = json.loads(src.read_text(encoding="utf-8"))
    results = [explain(a, a.get("evidence", []), use_model) for a in anomalies]

    out = PROCESSED / f"{filing_id}.explained.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    from_model = sum(1 for r in results if r["source"] == "model")
    reasons: dict[str, int] = {}
    for r in results:
        if r["fallback_reason"]:
            reasons[r["fallback_reason"]] = reasons.get(r["fallback_reason"], 0) + 1

    print(f"anomalies      : {len(results)}")
    print(f"model accepted : {from_model}")
    print(f"template used  : {len(results) - from_model}")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {n:>3}  {reason}")
    print()

    for r in results:
        tag = "MODEL   " if r["source"] == "model" else "template"
        print(f"[{tag}] {r['finding'][:88]}")
        print(f"  {r['explanation'][:300]}")
        if r.get("model_output") and r["source"] != "model":
            print(f"  rejected model output: {r['model_output'][:160]}")
        print()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()