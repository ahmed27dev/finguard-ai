"""
anomaly_detector.py - Step 5 of the pipeline. The project's central claim.

Decides which financial changes are worth flagging, using arithmetic only.
No model is involved at any point in this file, and that is the point: a
reviewer can recompute every flag by hand from the filing, and the same
filing always produces the same flags.

Three rules, in descending order of how much they are worth
-----------------------------------------------------------

DIVERGENCE      Two metrics that normally move together move apart:
                revenue up while operating income falls. Hard to dismiss,
                rarely a false positive, and the most analyst-relevant
                thing here.

SEGMENT_VS_TOTAL  One segment moves against the company. Apple's Greater
                China operating income fell three years running while the
                total rose. Invisible in any single table - it only exists
                because the linearizer rebuilt the series across three
                per-year tables.

RATIO_SHIFT     A derived ratio moves sharply: gross margin, operating
                margin, effective tax rate, current ratio. Catches changes
                that raw magnitudes hide.

MAGNITUDE       A single metric changes by more than a threshold. Kept
                last and deliberately weak, because on its own it fires on
                every fast-growing company. It earns a flag only when the
                change is large AND the metric is a headline one.

Every anomaly carries a `query` for retrieval and a `label` for the
lexical anchor, so the retrieval side can look for an explanation and
report honestly when the filing contains none.

Usage:
    python -m flagging.anomaly_detector aapl-2025
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROCESSED = Path("data/processed")

# Metrics that normally move together. Divergence between a pair is the
# signal; which direction each moved is recorded in the detail.
DIVERGENCE_PAIRS = [
    ("revenue", "operating_income"),
    ("revenue", "gross_profit"),
    ("revenue", "net_income"),
    ("net_income", "operating_cash_flow"),
    ("revenue", "receivables"),
]

# Ratios computed from canonical metrics. (name, numerator, denominator,
# as_percent, threshold in percentage points).
RATIOS = [
    ("gross margin", "gross_profit", "revenue", True, 2.0),
    ("operating margin", "operating_income", "revenue", True, 2.0),
    ("effective tax rate", "tax_provision", "pretax_income", True, 3.0),
    ("R&D intensity", "rnd_expense", "revenue", True, 1.0),
    ("current ratio", "current_assets", "current_liabilities", False, 0.15),
    ("debt to equity", "total_liabilities", "equity", False, 0.50),
]

HEADLINE = {"revenue", "operating_income", "net_income", "gross_profit",
            "operating_cash_flow", "total_assets", "equity"}

DIVERGENCE_MIN_GAP = 10.0    # percentage points between the two changes
MAGNITUDE_THRESHOLD = 15.0   # percent, headline metrics only
SEGMENT_MIN_GAP = 8.0        # percentage points, segment against total
SUSTAINED_MIN_GAP = 12.0     # percentage points over the whole span

# A metric that grew at roughly the rate the company grew is not anomalous,
# it is just growth. Measured on Microsoft FY2026: a bare 15% threshold
# flagged revenue, operating income, gross profit, net income, equity and
# total assets - six flags carrying no information, because the company
# grew uniformly. Magnitude only means something relative to the company's
# own trajectory.
MAGNITUDE_ALIGNED_PP = 8.0   # within this many points of revenue: not a flag
MAGNITUDE_ALIGNED_RATIO = (0.80, 1.25)  # or within this multiple of it
REVENUE_THRESHOLD = 25.0     # revenue has no baseline, so it needs its own

# An absolute points gap alone fails at high growth rates. Nvidia grew
# revenue 65.5% and gross profit 56.8% - an 8.7 point gap, over the cutoff,
# yet gross profit plainly just tracked revenue. Proportion catches that:
# 56.8/65.5 = 0.87, inside the band. Points catch the opposite case, where
# revenue barely moves and a small points gap is still proportionally huge.


def pct_change(new: float, old: float) -> float | None:
    """
    Percent change, or None when it cannot be computed honestly.

    A change from zero is undefined, not infinite, and a change from a
    negative base has no meaningful sign - reporting either would produce
    a flag that cannot be defended.
    """
    if old is None or new is None or old == 0 or old < 0:
        return None
    return (new - old) / abs(old) * 100.0


def latest_two(values: dict[str, float]) -> tuple[str, str] | None:
    """The two most recent periods present, newest first."""
    periods = sorted(values, reverse=True)
    return (periods[0], periods[1]) if len(periods) >= 2 else None


def _by_canonical(metrics: list[dict]) -> dict[str, dict]:
    return {m["canonical"]: m for m in metrics if m.get("canonical")}


def _fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def rule_divergence(metrics: list[dict]) -> list[dict]:
    """Two metrics that should move together moved apart."""
    named = _by_canonical(metrics)
    out = []

    for a_name, b_name in DIVERGENCE_PAIRS:
        a, b = named.get(a_name), named.get(b_name)
        if not a or not b:
            continue

        pa, pb = latest_two(a["values"]), latest_two(b["values"])
        if not pa or not pb or pa != pb:
            continue

        new_p, old_p = pa
        ca = pct_change(a["values"][new_p], a["values"][old_p])
        cb = pct_change(b["values"][new_p], b["values"][old_p])
        if ca is None or cb is None:
            continue

        gap = abs(ca - cb)
        opposite = (ca > 0) != (cb > 0)
        if not (opposite and gap >= DIVERGENCE_MIN_GAP):
            continue

        out.append({
            "rule": "divergence",
            "summary": (f"{a_name.replace('_', ' ')} {_fmt_pct(ca)} while "
                        f"{b_name.replace('_', ' ')} {_fmt_pct(cb)} "
                        f"({old_p} to {new_p})"),
            "severity": min(gap / 50.0, 1.0),
            "label": b["label"],
            "query": (f"why did {b['label']} change during {new_p} "
                      f"compared to {old_p}"),
            "metric_ids": [a["metric_id"], b["metric_id"]],
            "page": b["page"],
            "section": b["section"],
            "detail": {"periods": [old_p, new_p],
                       a_name: round(ca, 2), b_name: round(cb, 2),
                       "gap_pp": round(gap, 2)},
        })
    return out


def rule_segment_vs_total(metrics: list[dict]) -> list[dict]:
    """
    A segment moving against the company total.

    Segment series are labelled "<line item> - <segment>", produced by the
    linearizer's merge of per-year tables. Grouping by line item lets each
    segment be compared against its own Total.
    """
    groups: dict[str, dict[str, dict]] = {}
    for m in metrics:
        label = m["label"]
        if " - " not in label or m["layout"] != "merged_series":
            continue
        base, _, segment = label.rpartition(" - ")
        groups.setdefault(base.strip(), {})[segment.strip()] = m

    out = []
    for base, segments in groups.items():
        total = segments.get("Total")
        if not total:
            continue
        pt = latest_two(total["values"])
        if not pt:
            continue
        new_p, old_p = pt
        ct = pct_change(total["values"][new_p], total["values"][old_p])
        if ct is None:
            continue

        for name, m in segments.items():
            if name in {"Total", "Corporate"}:
                continue
            pm = latest_two(m["values"])
            if not pm or pm != pt:
                continue
            cm = pct_change(m["values"][new_p], m["values"][old_p])
            if cm is None:
                continue

            gap = abs(cm - ct)
            if not ((cm > 0) != (ct > 0) and gap >= SEGMENT_MIN_GAP):
                continue

            out.append({
                "rule": "segment_vs_total",
                "summary": (f"{base} in {name} {_fmt_pct(cm)} while the "
                            f"company total {_fmt_pct(ct)} "
                            f"({old_p} to {new_p})"),
                "severity": min(gap / 40.0, 1.0),
                "label": name,
                "query": (f"why did {name} {base.lower()} change during "
                          f"{new_p} compared to {old_p}"),
                "metric_ids": [m["metric_id"], total["metric_id"]],
                "page": m["page"],
                "section": m["section"],
                "detail": {"periods": [old_p, new_p], "segment": round(cm, 2),
                           "total": round(ct, 2), "gap_pp": round(gap, 2)},
            })
    return out


def _direction(values: dict[str, float]) -> int | None:
    """
    +1 if the series rose in every step, -1 if it fell in every step,
    0 if it changed direction. None if there are fewer than three periods.

    A single down year is noise. Three consecutive down years is a trend,
    and it is the trend that an analyst wants flagged.
    """
    periods = sorted(values)
    if len(periods) < 3:
        return None
    steps = [values[periods[i + 1]] - values[periods[i]]
             for i in range(len(periods) - 1)]
    if all(x > 0 for x in steps):
        return 1
    if all(x < 0 for x in steps):
        return -1
    return 0


def rule_sustained_trend(metrics: list[dict]) -> list[dict]:
    """
    A series moving one way in every period while its parent moves the other.

    Comparing only the two most recent periods understates a multi-year
    slide. Apple's Greater China operating income fell in both available
    steps - 30,328 to 27,082 to 26,917, down 11.2% overall - while the
    company total rose 16.4%. Year on year that reads as -0.6%, which
    ranked it last among seven flags; across the full span it is one of the
    clearest signals in the filing.
    """
    groups: dict[str, dict[str, dict]] = {}
    for m in metrics:
        label = m["label"]
        if " - " not in label or m["layout"] != "merged_series":
            continue
        base, _, segment = label.rpartition(" - ")
        groups.setdefault(base.strip(), {})[segment.strip()] = m

    out = []
    for base, segments in groups.items():
        total = segments.get("Total")
        if not total:
            continue
        total_dir = _direction(total["values"])
        if not total_dir:
            continue
        t_periods = sorted(total["values"])
        total_span = pct_change(total["values"][t_periods[-1]],
                                total["values"][t_periods[0]])
        if total_span is None:
            continue

        for name, m in segments.items():
            if name in {"Total", "Corporate"}:
                continue
            seg_dir = _direction(m["values"])
            if not seg_dir or seg_dir == total_dir:
                continue

            s_periods = sorted(m["values"])
            if s_periods != t_periods:
                continue
            seg_span = pct_change(m["values"][s_periods[-1]],
                                  m["values"][s_periods[0]])
            if seg_span is None:
                continue

            gap = abs(seg_span - total_span)
            if gap < SUSTAINED_MIN_GAP:
                continue

            direction = "declined" if seg_dir < 0 else "rose"
            out.append({
                "rule": "sustained_trend",
                "summary": (f"{base} in {name} {direction} in every period "
                            f"{s_periods[0]} to {s_periods[-1]} "
                            f"({_fmt_pct(seg_span)}) while the company total "
                            f"moved {_fmt_pct(total_span)}"),
                "severity": min(gap / 40.0, 1.0),
                "label": name,
                "query": (f"why did {name} {base.lower()} decline over "
                          f"{s_periods[0]} to {s_periods[-1]}"),
                "metric_ids": [m["metric_id"], total["metric_id"]],
                "page": m["page"],
                "section": m["section"],
                "detail": {"periods": s_periods,
                           "segment_span": round(seg_span, 2),
                           "total_span": round(total_span, 2),
                           "gap_pp": round(gap, 2),
                           "series": {p: m["values"][p] for p in s_periods}},
            })
    return out


def rule_ratio_shift(metrics: list[dict]) -> list[dict]:
    """A derived ratio moved more than its threshold."""
    named = _by_canonical(metrics)
    out = []

    for label, num_name, den_name, as_pct, threshold in RATIOS:
        num, den = named.get(num_name), named.get(den_name)
        if not num or not den:
            continue
        shared = sorted(set(num["values"]) & set(den["values"]), reverse=True)
        if len(shared) < 2:
            continue
        new_p, old_p = shared[0], shared[1]

        def ratio(period: str) -> float | None:
            d = den["values"][period]
            if not d:
                return None
            r = num["values"][period] / d
            return r * 100.0 if as_pct else r

        r_new, r_old = ratio(new_p), ratio(old_p)
        if r_new is None or r_old is None:
            continue

        move = r_new - r_old
        if abs(move) < threshold:
            continue

        unit = "pp" if as_pct else ""
        shown = (f"{r_old:.1f}% to {r_new:.1f}%" if as_pct
                 else f"{r_old:.2f} to {r_new:.2f}")
        out.append({
            "rule": "ratio_shift",
            "summary": f"{label} moved {shown} ({move:+.2f}{unit}), "
                       f"{old_p} to {new_p}",
            "severity": min(abs(move) / (threshold * 4), 1.0),
            # Anchor on the ratio's own name. Using the numerator's label
            # let "operating margin" match "cash provided by operating
            # activities"; "gross margin" and "operating margin" as phrases
            # appear verbatim in the MD&A when management discusses them.
            "label": label,
            "query": f"why did {label} change during {new_p} compared to {old_p}",
            "metric_ids": [num["metric_id"], den["metric_id"]],
            "page": num["page"],
            "section": num["section"],
            "detail": {"periods": [old_p, new_p], "from": round(r_old, 3),
                       "to": round(r_new, 3), "move": round(move, 3)},
        })
    return out


def rule_magnitude(metrics: list[dict]) -> list[dict]:
    """
    A headline metric moved more than the threshold.

    Restricted to headline metrics, and suppressed when the metric simply
    tracked revenue. Without both restrictions this rule flags every routine
    change and drowns the three above - which is the whole reason a bare
    year-over-year threshold is a weak detector on its own.
    """
    named = _by_canonical(metrics)
    revenue = named.get("revenue")
    revenue_change = None
    if revenue:
        rp = latest_two(revenue["values"])
        if rp:
            revenue_change = pct_change(revenue["values"][rp[0]],
                                        revenue["values"][rp[1]])

    out = []
    for m in metrics:
        canonical = m.get("canonical")
        if canonical not in HEADLINE:
            continue
        p = latest_two(m["values"])
        if not p:
            continue
        new_p, old_p = p
        change = pct_change(m["values"][new_p], m["values"][old_p])
        if change is None:
            continue

        if canonical == "revenue":
            # Revenue is the baseline, so it has nothing to be compared
            # against and needs a higher bar of its own.
            if abs(change) < REVENUE_THRESHOLD:
                continue
        else:
            if abs(change) < MAGNITUDE_THRESHOLD:
                continue
            if revenue_change is not None:
                if abs(change - revenue_change) < MAGNITUDE_ALIGNED_PP:
                    continue  # moved with the company, not against it
                lo, hi = MAGNITUDE_ALIGNED_RATIO
                if revenue_change > 0 and change > 0 and \
                        lo <= change / revenue_change <= hi:
                    continue  # same again, in proportional terms

        # Severity measures the gap from the company's own trajectory, not
        # the raw change. Scoring the raw change saturated at 1.00 for three
        # separate Nvidia metrics, leaving them unrankable.
        gap = (abs(change) if revenue_change is None or canonical == "revenue"
               else abs(change - revenue_change))
        divisor = 100.0 if canonical == "revenue" else 40.0

        out.append({
            "rule": "magnitude",
            "summary": (f"{m['canonical'].replace('_', ' ')} {_fmt_pct(change)} "
                        f"({old_p} to {new_p})"),
            "severity": min(gap / divisor, 1.0),
            "label": m["label"],
            "query": (f"why did {m['label']} change during {new_p} "
                      f"compared to {old_p}"),
            "metric_ids": [m["metric_id"]],
            "page": m["page"],
            "section": m["section"],
            "detail": {"periods": [old_p, new_p], "change": round(change, 2),
                       "revenue_change": (round(revenue_change, 2)
                                          if revenue_change is not None else None)},
        })
    return out


# ---------------------------------------------------------------------------

def detect_anomalies(metrics: list[dict]) -> list[dict]:
    """
    Run every rule and return anomalies ranked by severity.

    This is the contract with the retrieval side. Each anomaly carries a
    `query` to search with and a `label` to use as the lexical anchor.
    """
    found: list[dict] = []
    found += rule_divergence(metrics)
    found += rule_sustained_trend(metrics)
    found += rule_segment_vs_total(metrics)
    found += rule_ratio_shift(metrics)
    found += rule_magnitude(metrics)

    # A multi-year trend supersedes the single-year view of the same thing:
    # reporting both says the same finding twice, once understated.
    sustained = {(a["label"], tuple(sorted(a["metric_ids"])))
                 for a in found if a["rule"] == "sustained_trend"}
    found = [a for a in found
             if not (a["rule"] == "segment_vs_total"
                     and (a["label"], tuple(sorted(a["metric_ids"]))) in sustained)]

    found.sort(key=lambda a: -a["severity"])
    for i, a in enumerate(found):
        a["anomaly_id"] = f"a_{i:03d}"
    return found


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filing_id = sys.argv[1]
    src = PROCESSED / f"{filing_id}.metrics.json"
    if not src.exists():
        raise SystemExit(f"missing {src} - run flagging.metric_extractor first")

    metrics = json.loads(src.read_text(encoding="utf-8"))
    anomalies = detect_anomalies(metrics)

    out = PROCESSED / f"{filing_id}.anomalies.json"
    out.write_text(json.dumps(anomalies, indent=2), encoding="utf-8")

    by_rule: dict[str, int] = {}
    for a in anomalies:
        by_rule[a["rule"]] = by_rule.get(a["rule"], 0) + 1

    print(f"metrics in  : {len(metrics)}")
    print(f"anomalies   : {len(anomalies)}")
    for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"   {n:>3}  {rule}")
    print()
    for a in anomalies[:15]:
        print(f"  [{a['severity']:.2f}] {a['rule']:<17} p{a['page']:<4} {a['summary']}")
    if len(anomalies) > 15:
        print(f"  ... and {len(anomalies) - 15} more")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()