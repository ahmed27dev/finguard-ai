"""
eval_retrieval.py - Step 7 of the retrieval track. The numbers for the report.

Measures how often retrieval puts the right evidence at the top, using
FinQA's gold_inds annotations as ground truth.

Two modes, and reporting both is the point
------------------------------------------
EASY    Candidates come from one FinQA example only: roughly 40-60
        sentences and table rows. This is the setting FinQA was designed
        for and what published MRR numbers describe.

POOLED  Candidates are pooled across every example loaded, so a query
        competes against thousands of distractors from other filings.

Our real system retrieves over 637 chunks of a whole 10-K, which is much
closer to POOLED than to EASY. Reporting only the EASY number would
overstate retrieval quality on the actual task. Report both, and say so.

Two systems, so stage 2 has to justify itself
---------------------------------------------
BI      bi-encoder alone
BI+CE   bi-encoder shortlist, cross-encoder rerank

The delta between them is the evidence that the reranker earns its cost.

Metrics
-------
MRR         1/rank of the first gold item. Rewards putting one right
            answer first.
Recall@k    fraction of gold items appearing in the top k. A question with
            three gold sentences needs all three, so this is the metric
            that matters for multi-evidence questions.
NDCG@k      rank-weighted, credits gold items further down the list.

Usage:
    python -m retrieval.eval_retrieval --limit 100
    python -m retrieval.eval_retrieval --limit 300 --split test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FINQA_DIR = Path("data/finqa")
RESULTS_DIR = Path("data/results")
CANDIDATES = 20  # stage 1 shortlist, matches retrieve.py
K_VALUES = (1, 3, 5, 10, 20, 50)


# ---------------------------------------------------------------------------
# FinQA loading
# ---------------------------------------------------------------------------

def linearize_finqa_row(table: list[list[str]], row_idx: int) -> str:
    """
    Reproduce FinQA's own row format exactly.

        {header[0]} the {row[0]} of {header[j]} is {row[j]} ; ...

    It has to match theirs, not ours, because gold_inds is keyed by their
    row indices and we compare against their annotation.
    """
    if row_idx >= len(table) or not table:
        return ""
    header, row = table[0], table[row_idx]
    parts = [f"the {row[0]} of {header[j]} is {row[j]} ;" for j in range(1, min(len(row), len(header)))]
    return f"{header[0]} " + " ".join(parts)


def build_candidates(example: dict) -> list[dict]:
    """
    All retrievable units for one example, keyed the way gold_inds is.

    text_N indexes into pre_text followed by post_text; table_N is row N
    of the table, with row 0 being the header.
    """
    candidates: list[dict] = []

    sentences = list(example.get("pre_text", [])) + list(example.get("post_text", []))
    for i, sentence in enumerate(sentences):
        text = sentence.strip()
        if len(text) > 2:  # FinQA pads with lone "." entries
            candidates.append({"id": f"text_{i}", "text": text})

    table = example.get("table", [])
    for i in range(1, len(table)):
        text = linearize_finqa_row(table, i)
        if text.strip():
            candidates.append({"id": f"table_{i}", "text": text})

    return candidates


def load_examples(split: str, limit: int) -> list[dict]:
    """Load examples that have a question and at least one gold evidence item."""
    path = FINQA_DIR / f"{split}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "  curl -L -o data/finqa/test.json "
            "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for ex in raw:
        qa = ex.get("qa") or {}
        gold = set((qa.get("gold_inds") or {}).keys())
        question = (qa.get("question") or "").strip()
        if not gold or not question:
            continue
        candidates = build_candidates(ex)
        if not any(c["id"] in gold for c in candidates):
            continue  # gold id we could not reconstruct - skip rather than score it wrong
        out.append({"id": ex.get("id", ""), "question": question,
                    "gold": gold, "candidates": candidates})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mrr(ranked: list, gold: set) -> float:
    for rank, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: list, gold: set, k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def ndcg_at_k(ranked: list, gold: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(ranked[:k], start=1)
        if cid in gold
    )
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run(split: str, limit: int, rerank_on: bool) -> dict:
    from retrieval.indexer import embed
    from retrieval.reranker import get_model as get_ce

    examples = load_examples(split, limit)
    if not examples:
        raise SystemExit("no usable examples loaded")

    # One flat pool of every candidate across every example. EASY mode masks
    # it down to the owning example; POOLED searches all of it.
    # Gold is matched on global pool position, never on the FinQA id string:
    # ids like "table_1" repeat in every example, so comparing by id would
    # count another filing's row as a correct hit and inflate every metric.
    pool_texts, pool_ids, owner = [], [], []
    for e_i, ex in enumerate(examples):
        gold_positions = set()
        for c in ex["candidates"]:
            if c["id"] in ex["gold"]:
                gold_positions.add(len(pool_texts))
            pool_texts.append(c["text"])
            pool_ids.append(c["id"])
            owner.append(e_i)
        ex["gold_pos"] = gold_positions
    owner_arr = np.array(owner)

    print(f"{len(examples)} questions, {len(pool_texts)} candidates in pool")
    print("embedding candidates ...")
    pool_vecs = embed(pool_texts)
    query_vecs = embed([ex["question"] for ex in examples])

    ce = get_ce() if rerank_on else None
    results: dict[str, dict] = {}

    for mode in ("easy", "pooled"):
        systems = ["bi"] + (["bi+ce"] if rerank_on else [])
        scores = {s: {"mrr": [], **{f"r@{k}": [] for k in K_VALUES},
                      **{f"ndcg@{k}": [] for k in K_VALUES}} for s in systems}
        gap_when_correct, gap_when_wrong = [], []

        for e_i, ex in enumerate(examples):
            sims = pool_vecs @ query_vecs[e_i]
            if mode == "easy":
                sims = np.where(owner_arr == e_i, sims, -np.inf)

            order = [int(i) for i in np.argsort(-sims)[:CANDIDATES]]
            _record(scores["bi"], order, ex["gold_pos"])

            if ce is not None:
                pairs = [(ex["question"], pool_texts[i]) for i in order]
                ce_scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
                reordered = sorted(zip(order, ce_scores), key=lambda t: -t[1])
                _record(scores["bi+ce"], [i for i, _ in reordered], ex["gold_pos"])

                # Score distribution for calibrating the support threshold
                top_pos, top_score = reordered[0][0], float(reordered[0][1])
                (gap_when_correct if top_pos in ex["gold_pos"]
                 else gap_when_wrong).append(top_score)

            if (e_i + 1) % 25 == 0:
                print(f"  {mode}: {e_i + 1}/{len(examples)}")

        results[mode] = {
            s: {k: float(np.mean(v)) for k, v in m.items()} for s, m in scores.items()
        }
        if gap_when_correct or gap_when_wrong:
            results[mode]["threshold_data"] = {
                "top_score_when_correct": _summary(gap_when_correct),
                "top_score_when_wrong": _summary(gap_when_wrong),
            }

    return {"split": split, "n_questions": len(examples),
            "pool_size": len(pool_texts), "results": results}


def _record(bucket: dict, ranked: list, gold: set) -> None:
    bucket["mrr"].append(mrr(ranked, gold))
    for k in K_VALUES:
        bucket[f"r@{k}"].append(recall_at_k(ranked, gold, k))
        bucket[f"ndcg@{k}"].append(ndcg_at_k(ranked, gold, k))


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values)
    return {"n": len(values), "mean": float(arr.mean()),
            "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90))}


def report(out: dict) -> None:
    print("\n" + "=" * 74)
    print(f"RETRIEVAL EVALUATION - FinQA {out['split']}, "
          f"{out['n_questions']} questions, pool {out['pool_size']}")
    print("=" * 74)

    header = f"{'mode':<8} {'system':<8} {'MRR':>7}" + "".join(
        f"{f'R@{k}':>8}" for k in K_VALUES) + "".join(f"{f'N@{k}':>8}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for mode, systems in out["results"].items():
        for system, m in systems.items():
            if system == "threshold_data":
                continue
            row = f"{mode:<8} {system:<8} {m['mrr']:>7.3f}"
            row += "".join(f"{m[f'r@{k}']:>8.3f}" for k in K_VALUES)
            row += "".join(f"{m[f'ndcg@{k}']:>8.3f}" for k in K_VALUES)
            print(row)

    for mode, systems in out["results"].items():
        td = systems.get("threshold_data")
        if not td:
            continue
        c, w = td["top_score_when_correct"], td["top_score_when_wrong"]
        if c.get("n") and w.get("n"):
            print(f"\n[{mode}] top rerank score when rank-1 was correct: "
                  f"median {c['median']:.2f} (p10 {c['p10']:.2f}, n={c['n']})")
            print(f"[{mode}] top rerank score when rank-1 was wrong:   "
                  f"median {w['median']:.2f} (p90 {w['p90']:.2f}, n={w['n']})")
            if c["p10"] <= w["p90"]:
                print("      -> the distributions overlap: no single threshold "
                      "separates correct from incorrect.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    out = run(args.split, args.limit, rerank_on=not args.no_rerank)
    report(out)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"retrieval_eval_{args.split}_{args.limit}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()