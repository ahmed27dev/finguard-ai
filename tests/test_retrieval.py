"""
test_retrieval.py - regression tests for the retrieval track.

Every test here corresponds to a bug that was actually hit, or to a
behaviour that would fail silently if it regressed. Silence is the point:
none of these produce an exception when they break. A wrong sign, a lost
page number or a borrowed unit just flows downstream and shows up as a
confidently wrong citation in the demo.

    pytest tests/ -v
"""

from __future__ import annotations

import pytest

from retrieval.parser import (
    _collapse_spacers, _norm_cell, _table_to_rows, parse_filing,
)
from retrieval.linearizer import (
    parse_value, _year_positions, detect_layout, detect_scale,
    linearize_table, merge_stacked,
)
from retrieval.chunker import split_sentences, chunk_text_blocks, MAX_CHARS
from retrieval.eval_retrieval import mrr, recall_at_k, ndcg_at_k
from retrieval.retrieve import _anchor_terms, _mentions


# ---------------------------------------------------------------------------
# Value parsing - a wrong sign silently inverts an anomaly
# ---------------------------------------------------------------------------

class TestParseValue:
    def test_accounting_negative(self):
        """Parentheses mean negative. Reading (95,699) as positive would
        turn a cost into income and flip every derived comparison."""
        assert parse_value("(95,699)") == (-95699.0, "number")

    def test_currency_negative(self):
        assert parse_value("$(42,627)") == (-42627.0, "currency")

    def test_currency_and_thousands_separator(self):
        assert parse_value("$178,353") == (178353.0, "currency")

    def test_percent(self):
        assert parse_value("(4)%") == (-4.0, "percent")

    def test_dash_is_none_not_zero(self):
        """A dash means 'not applicable to this segment'. Recording it as
        zero invents a data point and manufactures fake anomalies."""
        for dash in ("-", "\u2014", "$-", "N/A", ""):
            value, kind = parse_value(dash)
            assert value is None, f"{dash!r} should be None, got {value}"
            assert kind == "empty"

    def test_decimal(self):
        assert parse_value("7.49") == (7.49, "number")

    def test_text_is_not_a_number(self):
        assert parse_value("Greater China") == (None, "text")


# ---------------------------------------------------------------------------
# SEC table markup - spacer cells wrecked every table before this
# ---------------------------------------------------------------------------

class TestTableCells:
    def test_collapse_currency_and_percent(self):
        """SEC splits one figure across several cells with blank padding:
        'Americas' | '$' | '178,353' | '' | '7' | '%' ..."""
        raw = ["$", "178,353", "", "", "7", "%", "", "$", "167,045"]
        assert _collapse_spacers(raw) == ["$178,353", "7%", "$167,045"]

    def test_paren_spacing_normalised(self):
        assert _norm_cell("( 95,699 )") == "(95,699)"

    def test_label_column_kept_when_blank(self):
        """The header's first cell is empty but must survive, or the year
        columns shift out of alignment with the data rows below."""
        from bs4 import BeautifulSoup
        html = ("<table>"
                "<tr><td></td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Revenue</td><td>$100</td><td>$90</td></tr>"
                "</table>")
        rows = _table_to_rows(BeautifulSoup(html, "lxml").find("table"))
        assert len(rows[0]) == len(rows[1]) == 3
        assert rows[0][0] == ""


# ---------------------------------------------------------------------------
# Period headers - Nvidia produced zero rows before this was relaxed
# ---------------------------------------------------------------------------

class TestPeriodDetection:
    def test_bare_years(self):
        assert _year_positions(["", "2025", "2024"]) == {1: "2025", 2: "2024"}

    def test_fiscal_dates(self):
        """Nvidia's fiscal calendar gives 'Jan 25, 2026'. A bare-year
        pattern matched none of its tables and the filing yielded nothing."""
        got = _year_positions(["", "Jan 25, 2026", "Jan 26, 2025"])
        assert got == {1: "2026", 2: "2025"}

    def test_debt_row_is_not_a_header(self):
        """'3.20% Notes Due 2026' contains a year but is a data label."""
        assert _year_positions(["3.20% Notes Due 2026"]) == {}

    def test_long_cell_rejected(self):
        long_cell = "The Company issued notes maturing in 2026 under its program"
        assert _year_positions([long_cell]) == {}


# ---------------------------------------------------------------------------
# Layout and merging - the best anomaly is invisible without this
# ---------------------------------------------------------------------------

class TestLayout:
    def test_wide(self):
        rows = [["", "2025", "2024"], ["Revenue", "100", "90"]]
        layout = detect_layout(rows)
        assert layout["kind"] == "wide"
        assert layout["columns"] == {1: "2025", 2: "2024"}

    def test_stacked(self):
        """One year labelling the whole table, categories as columns."""
        rows = [["", "2025"],
                ["", "Americas", "Europe"],
                ["Net sales", "100", "90"]]
        layout = detect_layout(rows)
        assert layout["kind"] == "stacked"
        assert layout["period"] == "2025"

    def test_change_columns_excluded(self):
        """The filing's own year-over-year percentages must not be imported:
        our claim is that our arithmetic produces the flags."""
        table = {"table_id": "t", "page": 1, "section": "s",
                 "rows": [["", "2025", "Change", "2024"],
                          ["Greater China", "64,377", "(4)%", "66,952"]]}
        rows = linearize_table(table, "millions")
        assert rows[0]["values"] == {"2025": 64377.0, "2024": 66952.0}


class TestMergeStacked:
    def test_rebuilds_series_across_tables(self):
        """Apple splits its segment note into one table per year. Without
        merging, no year-over-year comparison is possible and the Greater
        China divergence - the project's showcase anomaly - is invisible."""
        records = []
        for period, china, total in [("2025", 26917, 133050),
                                     ("2024", 27082, 123216),
                                     ("2023", 30328, 114301)]:
            table = {"table_id": f"tb_{period}", "page": 47, "section": "Item 8",
                     "rows": [["", period],
                              ["", "Greater China", "Total"],
                              ["Operating income", str(china), str(total)]]}
            records.extend(linearize_table(table, "millions", "preceding"))

        merged = merge_stacked(records)
        china_series = next(
            r for r in merged if r["label"] == "Operating income - Greater China")
        assert china_series["values"] == {
            "2025": 26917.0, "2024": 27082.0, "2023": 30328.0}
        assert china_series["layout"] == "merged_series"

    def test_single_period_is_not_a_series(self):
        table = {"table_id": "t", "page": 1, "section": "s",
                 "rows": [["", "2025"], ["", "Americas"], ["Net sales", "100"]]}
        assert merge_stacked(linearize_table(table)) == []


# ---------------------------------------------------------------------------
# Scale - four attempts, three of them wrong in different ways
# ---------------------------------------------------------------------------

class TestScale:
    def test_units_inside_table(self):
        """Microsoft and Nvidia state units in the table's own first rows."""
        table = {"rows": [["(In millions)"], ["June 30,", "2026", "2025"]],
                 "page": 5}
        assert detect_scale(table, []) == ("millions", "table")

    def test_units_from_preceding_block(self):
        """Apple states them in the statement title just above the table.
        The parser records this during the walk as scale_hint."""
        table = {"rows": [["", "2025"]], "page": 5, "scale_hint": "millions"}
        assert detect_scale(table, []) == ("millions", "preceding")

    def test_no_units_stays_unknown(self):
        """The stock performance graph is indexed to 100, not millions.
        Borrowing units from unrelated prose mislabelled it by 10^6."""
        table = {"rows": [["", "2020"], ["Apple Inc.", "100"]], "page": 19}
        assert detect_scale(table, []) == (None, None)


# ---------------------------------------------------------------------------
# Chunking - citations depend on one chunk naming one page
# ---------------------------------------------------------------------------

class TestChunking:
    def test_abbreviations_do_not_split(self):
        text = "Sales in the U.S. increased. The Company, Inc. reported results."
        assert len(split_sentences(text)) == 2

    def test_never_merges_across_pages(self):
        """A chunk spanning two pages could not be cited honestly."""
        blocks = [
            {"block_id": "a", "text": "A short sentence about revenue growth here.",
             "page": 13, "section": "Item 1A"},
            {"block_id": "b", "text": "Another short sentence on the next page.",
             "page": 14, "section": "Item 1A"},
        ]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) == 2
        assert {c["page"] for c in chunks} == {13, 14}

    def test_never_merges_across_sections(self):
        blocks = [
            {"block_id": "a", "text": "A short sentence about revenue growth here.",
             "page": 20, "section": "Item 7"},
            {"block_id": "b", "text": "Another short sentence in a new section.",
             "page": 20, "section": "Item 8"},
        ]
        assert len(chunk_text_blocks(blocks)) == 2

    def test_long_block_split_under_cap(self):
        """MiniLM truncates at ~1000 chars without warning. Anything past
        the cap is silently never seen by the model."""
        long_text = ("The Company is subject to various claims and legal "
                     "proceedings arising in the ordinary course of business. ") * 30
        blocks = [{"block_id": "a", "text": long_text.strip(),
                   "page": 13, "section": "Item 1A"}]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) > 1
        assert all(len(c["text"]) <= MAX_CHARS for c in chunks)

    def test_complete_statements_not_merged(self):
        """Five per-segment explanations were glued into one chunk, making
        each one harder to retrieve and impossible to cite precisely."""
        blocks = [
            {"block_id": "a", "page": 22, "section": "Item 7",
             "text": "Greater China net sales decreased during 2025 compared to "
                     "2024 primarily due to lower net sales of iPhone."},
            {"block_id": "b", "page": 22, "section": "Item 7",
             "text": "Japan net sales increased during 2025 compared to 2024 "
                     "primarily due to higher net sales of iPhone and Services."},
        ]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Retrieval guards and metrics
# ---------------------------------------------------------------------------

class TestAnchors:
    def test_anchor_rejects_lexical_lookalike(self):
        """A passage about export restrictions scored 3.34 for a
        restructuring query - it matched 'restructure' to 'restrictive'.
        Score alone could not reject it; the anchor can."""
        anchors = _anchor_terms("restructuring")
        text = "Restrictive measures can also require the Company to act."
        assert not _mentions(text, anchors)

    def test_anchor_accepts_real_mention(self):
        anchors = _anchor_terms("Greater China")
        text = "Greater China net sales decreased during 2025."
        assert _mentions(text, anchors)

    def test_stopwords_excluded(self):
        assert "company" not in _anchor_terms("Company total net sales")


class TestMetrics:
    def test_mrr(self):
        assert mrr(["a", "g", "c"], {"g"}) == pytest.approx(0.5)
        assert mrr(["a", "b"], {"g"}) == 0.0

    def test_recall(self):
        assert recall_at_k(["a", "g", "c"], {"g", "z"}, 3) == pytest.approx(0.5)

    def test_ndcg_bounded(self):
        """NDCG came out at 1.19 once because FinQA ids like 'table_1'
        repeat across examples and another filing's row counted as gold."""
        for gold in ({"a"}, {"a", "b"}, {"a", "b", "c"}):
            score = ndcg_at_k(["a", "b", "c", "d"], gold, 10)
            assert 0.0 <= score <= 1.0

    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "z"], {"a", "b"}, 10) == pytest.approx(1.0)