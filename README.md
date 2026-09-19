# FinGuard AI

Reads a 10-K filing, finds financial changes worth an analyst's attention,
retrieves management's account of each one from the filing itself, and checks
that the account actually supports the explanation given.

The design constraint that shapes everything: **no language model decides what
is flagged.** Concerns come from arithmetic over the filing's own tables, so
every flag can be recomputed by hand from the figures shown alongside it. The
model's only job is to read the retrieved passage, and a separate verifier
checks what it wrote against that passage.

---

## What it does

```
10-K (HTML)
   │
   ├─ parse ──── text blocks + tables, with page and Item section
   ├─ linearize ─ table rows as sentences AND {year: value} series
   ├─ chunk ───── retrievable units, never spanning a page boundary
   └─ index ───── FAISS over all chunks
   │
   ├─ extract ─── canonical metrics (revenue, operating income, ...)
   ├─ detect ──── ARITHMETIC ONLY: divergence, sustained trend,
   │              ratio shift, magnitude
   ├─ retrieve ── bi-encoder (top 20) → cross-encoder (top 3)
   ├─ generate ── FLAN-T5 extracts the cause from the passage
   └─ verify ──── NLI entailment + numeric + causal checks
```

An anomaly with no supporting passage is reported as **unexplained**, not
hidden. That case is often the most useful output: the arithmetic found a
real change and management never addressed it.

---

## Results

Measured on four filings (Apple FY2025, Microsoft FY2026, Nvidia FY2026,
JPMorgan FY2025) and the FinQA benchmark.

### Retrieval — FinQA test, 200 questions

| Mode   | System     | MRR       | R@1   | R@3   | R@20  |
| ------ | ---------- | --------- | ----- | ----- | ----- |
| easy   | bi-encoder | 0.752     | 0.466 | 0.723 | 0.956 |
| easy   | + reranker | 0.798     | 0.532 | 0.761 | 0.956 |
| pooled | bi-encoder | 0.489     | 0.273 | 0.435 | 0.700 |
| pooled | + reranker | **0.580** | 0.351 | 0.533 | 0.700 |

_easy_ scores against one FinQA example's own candidates, roughly 40–60 items.
_pooled_ scores against every candidate across all 200 examples. The real
system searches 751–4,837 chunks of a whole filing, so **pooled is the honest
figure and easy overstates quality.** Both are reported.

The reranker is worth three times more under realistic distractor density
(+18.6% MRR pooled vs +6.1% easy) — with few candidates the bi-encoder rarely
meets a hard negative.

### Flagging — 16 hand-labelled concerns

| Metric                  | Result |
| ----------------------- | ------ |
| Detection precision     | 0.562  |
| Evidence accuracy       | 0.667  |
| Explained by the filing | 0.375  |

Recall is not reported. Establishing it would mean reading each filing in full
and enumerating every concern the system should have raised. Precision is
measured; recall is not estimated.

_Explained by the filing_ is not an accuracy figure — it is how often
management narrates a change the arithmetic found. The other 62.5% are the
concerns a reader would otherwise have to find unaided.

### Faithfulness — adversarial canary, 6/6

The system's own explanations pass verification at 0.81–0.99 entailment, but
that is weak evidence: the generator is extractive, so its claims are
near-verbatim spans and entailment is close to guaranteed. The checker was
therefore tested against deliberately wrong pairs:

| Case                       | Result                                           |
| -------------------------- | ------------------------------------------------ |
| over-claimed cause         | rejected, 0.006                                  |
| substituted figure         | rejected, 0.001 (and caught by the numeric test) |
| contradiction              | rejected, 0.000                                  |
| unrelated passage          | rejected, neutral                                |
| faithful extraction        | passed, 0.725                                    |
| cause stated by the filing | passed, 0.998                                    |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

SEC rejects requests without a descriptive User-Agent:

```bash
export FINGUARD_UA="Your Project Name your@email.com"
# Windows: $env:FINGUARD_UA = "Your Project Name your@email.com"
```

## Run

```bash
python -m retrieval.fetch_filing AAPL        # download the latest 10-K
python pipeline.py aapl-2025 --investigate   # parse → detect → retrieve
python -m flagging.generator aapl-2025       # explain
python -m flagging.faithfulness_check aapl-2025
python -m streamlit run app.py               # dashboard
```

First run downloads four models, about 1.5 GB total. Everything runs on CPU.

## Reproduce the results

```bash
python -m pytest tests/ -q                            # 84 tests

curl -L -o data/finqa/test.json \
  https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json
python -m retrieval.eval_retrieval --split test --limit 200   # ~25 min

python -m flagging.faithfulness_check --canary
python -m flagging.eval_flagging --report             # uses data/labels/
```

`data/labels/` holds the hand labels the flagging evaluation depends on. They
are the one thing in this repo that cannot be regenerated.

---

## Layout

```
retrieval/     parser · linearizer · chunker · indexer · reranker
               retrieve · eval_retrieval · fetch_filing
flagging/      metric_extractor · anomaly_detector · generator
               faithfulness_check · eval_flagging
pipeline.py    runs every stage, skipping what is already current
app.py         Streamlit dashboard
config.yaml    every threshold, asserted against the code by tests
tests/         84 tests, each tied to a bug that occurred
```

Diagnostics used during development and kept because they are still useful:
`inspect_parse.py`, `check_filings.py`, `diagnose.py`, `debug_scale.py`,
`why_supported.py`.

---

## Limitations

**HTML only.** SEC serves filings as HTML with inline XBRL, which parses far
more reliably than PDF. PDF upload is not implemented.

**The parser generalises, but only after being made to.** It was built against
one Apple filing, looked correct, and then failed silently on three others —
producing 1,078 confidently mislabelled chunks for Microsoft with no error
raised. Apple states units in a statement title, Microsoft inside the table,
Nvidia labels year columns as `Jan 25, 2026` rather than `2026`, and none of
the three besides Apple prints a page footer. Each was found by testing on
unfamiliar filers, not by inspection.

**Relevance scores are not comparable across queries.** Over 200 questions, a
correct top-1 answer scored 1.03 while a false positive scored 3.34, and the
score distributions for correct and incorrect answers overlap. A single
threshold cannot separate them, which is why support requires a lexical anchor
and a source-type check in addition to the score.

**A wider shortlist does not help.** Raising the first-stage shortlist from 20
to 50 lifted pooled Recall@50 to 0.774 from a Recall@20 ceiling of 0.700, but
left Recall@3 unchanged. Since only the top 3 reach the generator, the extra
recall is unusable. The bottleneck is the bi-encoder's representation, not the
funnel width.

**Year-over-year magnitude is a weak detector alone.** A flat 15% threshold
flagged six Microsoft metrics that had all simply grown with the company.
Suppressing metrics that move with revenue reduced those six to two.

**Small samples.** Detection precision rests on 16 labels across three filings,
and one rule (`divergence`) has a single labelled instance.

**Not an audit.** FinGuard flags areas for review. It is not a fraud detector,
an auditor, or investment advice; every concern needs human judgement.

---

## Related work

HC-RAG ([arXiv 2608.12335](https://arxiv.org/abs/2608.12335)) improves evidence
retrieval over filings with a typed document-section-unit graph. CiteCheck
([arXiv 2605.27700](https://arxiv.org/abs/2605.27700)) detects fabricated
references in AI-generated text — verifying that citations point to real works,
explicitly not that they support the surrounding claim. Claim-to-source
alignment is the separate problem this project's verification step addresses.

FinGuard differs in combining arithmetic-driven proactive detection with
claim-level verification of each generated explanation against its citation.
