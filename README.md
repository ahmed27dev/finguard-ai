# FinGuard AI

Explainable Retrieval-Augmented Financial Filing Analysis and Concern Flagging System.

Upload a 10-K filing → the system parses it, detects financial anomalies using
statistical rules, retrieves supporting evidence with a two-stage retrieval
pipeline, generates a plain-language explanation for each flagged concern, and
verifies that the explanation is grounded in its cited evidence.

## Team split

- **Person A** — `retrieval/` — parsing, linearization, chunking, bi-encoder retrieval, cross-encoder reranking, retrieval evaluation.
- **Person B** — `flagging/` — metric extraction, anomaly detection, explanation generation, citation mapping, faithfulness verification.
- **Together** — `pipeline.py`, `app.py`, integration, final evaluation.

## Getting started

\`\`\`bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## Interface contract

\`\`\`python

# retrieval/retrieve.py -> Person A implements this

def retrieve_evidence(query: str, filing_id: str, top_k: int = 5) -> list[dict]:
"""
Returns:
[
{"text": "...", "source_type": "table", "page": 12,
"section": "Item 7 - MD&A", "score": 0.87}, ...
]
"""

# flagging/ -> Person B implements this

def flag_concerns(evidence_list: list[dict]) -> list[dict]:
"""
Returns:
[
{"concern": "...", "explanation": "...",
"citation": {"page": 12, "section": "...", "text_span": "..."},
"faithfulness_score": 0.91}, ...
]
"""
\`\`\`
