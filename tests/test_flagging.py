"""
test_flagging.py - regression tests for the detection and explanation track.

Every test here corresponds to a bug that occurred during development and
would recur silently. None of them raise an exception when they break: a
segment's revenue gets compared against the company's operating income, a
cash-flow change outranks a balance sheet figure, a model echoes its own
prompt back. The output still looks like an answer.

    pytest tests/test_flagging.py -v
"""

from __future__ import annotations

import pytest

from flagging.metric_extractor import (
    canonical_name, is_usable, extract_metrics,
)
from flagging.anomaly_detector import (
    pct_change, _direction, detect_anomalies, rule_magnitude,
    rule_sustained_trend, rule_divergence, rule_ratio_shift,
)
from flagging.generator import validate, build_template, build_prompt, _overlap
from flagging.faithfulness_check import (
    split_explanation, check_numbers, check_causation,
)

from tests.mock_data import (
    linear_row, metric, evidence, APPLE_SEGMENT_OP_INCOME,
    MSFT_UNIFORM_GROWTH, NVDA_HYPERGROWTH, APPLE_TAX_EVIDENCE,
    APPLE_CHINA_EVIDENCE,
)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

class TestCanonicalNames:
    def test_filers_use_different_words_for_revenue(self):
        """Apple says "Total net sales", Microsoft says "Total revenue"."""
        for label in ["Total net sales", "Total revenue", "Net sales", "Revenue"]:
            assert canonical_name(label) == "revenue", label

    def test_footnote_markers_stripped(self):
        """"Services (1)" must not be treated as a different metric."""
        assert canonical_name("Total net sales (1)") == "revenue"

    def test_segment_rows_stay_unnamed(self):
        """A segment is a real metric but not the company-wide one."""
        assert canonical_name("Operating income/(loss) - Greater China") is None

    def test_unknown_label_is_none(self):
        assert canonical_name("Vendor non-trade receivables") is None


class TestUsability:
    def test_stock_performance_index_rejected(self):
        """Indexed to 100, not dollars. Flagging it produces nonsense."""
        row = linear_row("S&P 500 Index", {"2025": 217.0, "2024": 185.0},
                         section="Item 5 - Market for Registrant", scale=None)
        usable, reason = is_usable(row)
        assert not usable
        assert "section" in reason or "units" in reason

    def test_per_share_rejected(self):
        row = linear_row("Basic earnings per share", {"2025": 7.49, "2024": 6.11})
        assert not is_usable(row)[0]

    def test_single_period_rejected(self):
        """Stacked rows hold one year; merged_series carries the span."""
        row = linear_row("Net sales", {"Americas": 178353.0}, layout="stacked")
        assert not is_usable(row)[0]

    def test_unknown_units_rejected(self):
        row = linear_row("Some metric", {"2025": 100.0, "2024": 90.0}, scale=None)
        usable, reason = is_usable(row)
        assert not usable and "units" in reason

    def test_normal_row_accepted(self):
        row = linear_row("Total net sales",
                         {"2025": 416161.0, "2024": 391035.0})
        assert is_usable(row)[0]


class TestPrimarySelection:
    def test_company_total_beats_segments(self):
        """
        Microsoft labels four different series "Total revenue" - the company
        figure and three segments. Without this, a segment's revenue could be
        compared against company operating income.
        """
        rows = [
            linear_row("Total revenue", {"2026": 331839.0, "2025": 281724.0}, page=50),
            linear_row("Total revenue", {"2026": 139996.0, "2025": 120810.0}, page=95),
            linear_row("Total revenue", {"2026": 137791.0, "2025": 106265.0}, page=95),
        ]
        metrics = extract_metrics(rows)
        revenue = [m for m in metrics if m["canonical"] == "revenue"]
        assert len(revenue) == 1, "exactly one series may carry the canonical name"
        assert revenue[0]["values"]["2026"] == 331839.0
        assert all(m["role"] == "component"
                   for m in metrics if m.get("component_of") == "revenue")

    def test_balance_beats_cash_flow_change(self):
        """
        The balance sheet shows two dates, the cash flow statement three, so
        ranking by period count put the *change* in inventory above the
        inventory balance. Magnitude is the reliable signal.
        """
        rows = [
            linear_row("Inventories", {"2025": 5718.0, "2024": 7286.0}, page=33),
            linear_row("Inventories",
                       {"2025": 1400.0, "2024": -1046.0, "2023": -1618.0}, page=35),
        ]
        metrics = extract_metrics(rows)
        inventory = [m for m in metrics if m["canonical"] == "inventory"]
        assert len(inventory) == 1
        assert inventory[0]["values"]["2025"] == 5718.0

    def test_period_subsets_merged(self):
        """The same series in two tables over different spans is one metric."""
        rows = [
            linear_row("Total revenue", {"2026": 331839.0, "2025": 281724.0}),
            linear_row("Total revenue",
                       {"2026": 331839.0, "2025": 281724.0, "2024": 245122.0}),
        ]
        metrics = extract_metrics(rows)
        revenue = [m for m in metrics if m["canonical"] == "revenue"]
        assert len(revenue) == 1
        assert len(revenue[0]["values"]) == 3, "the longer span should survive"


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestPctChange:
    def test_normal(self):
        assert pct_change(110.0, 100.0) == pytest.approx(10.0)

    def test_zero_base_is_none(self):
        """A change from nothing is undefined, not infinite."""
        assert pct_change(50.0, 0.0) is None

    def test_negative_base_is_none(self):
        """From -5 to +3 has no defensible percentage, so none is reported."""
        assert pct_change(3.0, -5.0) is None


class TestDirection:
    def test_monotonic_decline(self):
        assert _direction({"2023": 30328.0, "2024": 27082.0, "2025": 26917.0}) == -1

    def test_monotonic_rise(self):
        assert _direction({"2023": 114301.0, "2024": 123216.0, "2025": 133050.0}) == 1

    def test_wavering_is_zero(self):
        assert _direction({"2023": 100.0, "2024": 90.0, "2025": 120.0}) == 0

    def test_two_periods_is_not_a_trend(self):
        """One step cannot establish direction."""
        assert _direction({"2024": 100.0, "2025": 90.0}) is None


# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------

class TestMagnitudeRule:
    def test_growth_tracking_revenue_is_not_flagged(self):
        """
        Microsoft FY2026: every headline metric grew with the company. A bare
        15% threshold raised six flags carrying no information.
        """
        metrics = [metric(name, name.replace("_", " ").title(), values,
                          metric_id=f"m_{i:04d}")
                   for i, (name, values) in enumerate(MSFT_UNIFORM_GROWTH.items())]
        flagged = {a["metric_ids"][0] for a in rule_magnitude(metrics)}
        by_id = {m["metric_id"]: m["canonical"] for m in metrics}
        names = {by_id[i] for i in flagged}

        assert "operating_income" not in names, "grew 20.8% against revenue 17.8%"
        assert "gross_profit" not in names
        assert "total_assets" not in names
        assert "net_income" in names, "grew 31.3%, nearly double revenue"

    def test_proportional_check_catches_high_growth(self):
        """
        Nvidia: gross profit is 8.7 points from revenue - past an absolute
        cutoff - but 56.8/65.5 = 0.87 proportionally, so it just tracked.
        """
        metrics = [metric(name, name, values, metric_id=f"m_{i:04d}")
                   for i, (name, values) in enumerate(NVDA_HYPERGROWTH.items())]
        by_id = {m["metric_id"]: m["canonical"] for m in metrics}
        names = {by_id[a["metric_ids"][0]] for a in rule_magnitude(metrics)}
        assert "gross_profit" not in names
        assert "equity" in names, "nearly doubled, far beyond revenue growth"

    def test_severity_does_not_saturate(self):
        """Three Nvidia metrics scored exactly 1.00 and became unrankable."""
        metrics = [metric(name, name, values, metric_id=f"m_{i:04d}")
                   for i, (name, values) in enumerate(NVDA_HYPERGROWTH.items())]
        severities = [a["severity"] for a in rule_magnitude(metrics)]
        assert len(set(severities)) == len(severities), "severities must differ"


class TestSustainedTrend:
    def _segment_metrics(self) -> list[dict]:
        return [
            metric(None, f"Operating income/(loss) - {seg}", values,
                   page=47, layout="merged_series", metric_id=f"m_{i:04d}")
            for i, (seg, values) in enumerate(APPLE_SEGMENT_OP_INCOME.items())
        ]

    def test_detects_segment_falling_while_total_rises(self):
        """
        The project's showcase finding. Year over year it reads -0.6% and
        ranks last; across the full span it is -11.2% against +16.4%.
        """
        found = rule_sustained_trend(self._segment_metrics())
        china = [a for a in found if a["label"] == "Greater China"]
        assert len(china) == 1
        assert china[0]["detail"]["segment_span"] == pytest.approx(-11.25, abs=0.1)
        assert china[0]["detail"]["total_span"] == pytest.approx(16.4, abs=0.1)

    def test_segments_moving_with_the_total_are_not_flagged(self):
        found = rule_sustained_trend(self._segment_metrics())
        assert {a["label"] for a in found} == {"Greater China"}

    def test_severity_beats_the_single_year_view(self):
        """The trend must outrank what the one-year rule would have scored."""
        china = rule_sustained_trend(self._segment_metrics())[0]
        assert china["severity"] > 0.5

    def test_single_year_view_is_suppressed(self):
        """Reporting both states one finding twice, once understated."""
        anomalies = detect_anomalies(self._segment_metrics())
        rules = {(a["rule"], a["label"]) for a in anomalies}
        assert ("sustained_trend", "Greater China") in rules
        assert ("segment_vs_total", "Greater China") not in rules


class TestDivergenceAndRatios:
    def test_quality_of_earnings_divergence(self):
        """Apple FY2025: earnings up 19.5%, cash from operations down 5.7%."""
        metrics = [
            metric("net_income", "Net income",
                   {"2025": 112010.0, "2024": 93736.0}, metric_id="m_0000"),
            metric("operating_cash_flow", "Cash generated by operating activities",
                   {"2025": 111482.0, "2024": 118254.0}, metric_id="m_0001"),
        ]
        found = rule_divergence(metrics)
        assert len(found) == 1
        assert found[0]["detail"]["gap_pp"] == pytest.approx(25.2, abs=0.5)

    def test_effective_tax_rate_shift(self):
        metrics = [
            metric("tax_provision", "Provision for income taxes",
                   {"2025": 20719.0, "2024": 29749.0}, metric_id="m_0000"),
            metric("pretax_income", "Income before provision for income taxes",
                   {"2025": 132729.0, "2024": 123485.0}, metric_id="m_0001"),
        ]
        found = [a for a in rule_ratio_shift(metrics)
                 if "tax" in a["summary"].lower()]
        assert len(found) == 1
        assert found[0]["label"] == "effective tax rate", \
            "ratios anchor on their own name, not the numerator's label"

    def test_anomalies_carry_query_and_anchor(self):
        """Retrieval needs both; without them the join silently degrades."""
        metrics = [
            metric("revenue", "Total net sales",
                   {"2025": 416161.0, "2024": 300000.0}, metric_id="m_0000"),
        ]
        for a in detect_anomalies(metrics):
            assert a["query"] and a["label"]
            assert a["anomaly_id"].startswith("a_")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGenerator:
    anomaly = {
        "anomaly_id": "a_000",
        "label": "effective tax rate",
        "summary": "effective tax rate moved 24.1% to 15.6% (-8.48pp), 2024 to 2025",
        "query": "why did effective tax rate change during 2025 compared to 2024",
    }

    def test_template_quotes_the_evidence(self):
        out = build_template(self.anomaly, [evidence(APPLE_TAX_EVIDENCE)])
        assert "page 27" in out
        assert "provision for income taxes" in out

    def test_template_states_absence_plainly(self):
        out = build_template(self.anomaly, [])
        assert "no passage" in out.lower()

    def test_finding_is_kept_out_of_the_prompt(self):
        """
        With "Finding: <summary>" at the top, FLAN-T5 copied that line back
        verbatim on every accepted output.
        """
        prompt = build_prompt(self.anomaly, [evidence(APPLE_TAX_EVIDENCE)])
        assert self.anomaly["summary"] not in prompt
        assert "Question:" in prompt and "Context:" in prompt

    def test_echo_of_the_finding_rejected(self):
        """
        The failure that passed every other check - unique words, no invented
        numbers, no causal claim - while adding nothing at all.
        """
        ok, reason = validate(self.anomaly["summary"], self.anomaly,
                              [evidence(APPLE_TAX_EVIDENCE)])
        assert not ok and "echo" in reason

    def test_invented_number_rejected(self):
        ok, reason = validate(
            "a $22.4 billion decrease in the provision for income taxes",
            self.anomaly, [evidence(APPLE_TAX_EVIDENCE)])
        assert not ok and "22.4" in reason

    def test_invented_cause_rejected(self):
        """The over-claim the project's own design document shipped."""
        anomaly = dict(self.anomaly, summary="operating income fell 75%")
        ev = [evidence("Operating expenses increased.")]
        ok, reason = validate(
            "operating income decreased primarily due to restructuring costs",
            anomaly, ev)
        assert not ok and "cause" in reason

    def test_faithful_extraction_accepted(self):
        ok, reason = validate(
            "a $10.7 billion year-over-year decrease in the provision for "
            "income taxes", self.anomaly, [evidence(APPLE_TAX_EVIDENCE)])
        assert ok, reason

    def test_overlap_measure(self):
        assert _overlap("a b c", "a b c") == pytest.approx(1.0)
        assert _overlap("a b c", "x y z") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

class TestFaithfulnessParts:
    def test_claim_is_separated_from_arithmetic(self):
        """
        Only the model's half may be entailment-checked. The finding is
        arithmetic the evidence was never meant to entail.
        """
        finding = "gross margin moved 75.0% to 71.1% (-3.92pp), 2025 to 2026"
        full = f"{finding}. The filing states: business model transitioned."
        assert split_explanation(full, finding) == "business model transitioned."

    def test_substituted_figure_caught(self):
        ok, missing = check_numbers("a $22.4 billion decrease",
                                    APPLE_TAX_EVIDENCE)
        assert not ok and "22.4" in missing

    def test_matching_figure_passes(self):
        ok, _ = check_numbers("a $10.7 billion decrease", APPLE_TAX_EVIDENCE)
        assert ok

    def test_cause_not_in_evidence_caught(self):
        assert not check_causation("fell due to restructuring costs",
                                   "Operating expenses increased.")

    def test_cause_stated_by_the_filing_passes(self):
        assert check_causation("declined due to lower iPhone sales",
                               APPLE_CHINA_EVIDENCE)

    def test_claim_without_causal_language_passes(self):
        assert check_causation("lower net sales of iPhone", APPLE_CHINA_EVIDENCE)